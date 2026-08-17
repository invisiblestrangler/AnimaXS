#import "A12ANEBridge.h"
#import <dispatch/dispatch.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <mach/vm_param.h>
#include <stdint.h>
#include <string.h>

// This is intentionally a very small extraction of the A12/H11 mechanism
// validated on-device by AnimaANEProbe v14: Apple-compiled Espresso bundle ->
// _ANEModel -> _ANEClient -> IOSurface-backed request, with the same IOSurface
// simultaneously exposed as a shared MTLBuffer via bytesNoCopy.

typedef CFTypeRef A12IOSurfaceRef;
typedef A12IOSurfaceRef (*IOSurfaceCreateFn)(CFDictionaryRef);
typedef int32_t (*IOSurfaceLockFn)(A12IOSurfaceRef, uint32_t, uint32_t *);
typedef int32_t (*IOSurfaceUnlockFn)(A12IOSurfaceRef, uint32_t, uint32_t *);
typedef void *(*IOSurfaceGetBaseAddressFn)(A12IOSurfaceRef);

typedef struct {
    void *handle;
    IOSurfaceCreateFn create;
    IOSurfaceLockFn lock;
    IOSurfaceUnlockFn unlock;
    IOSurfaceGetBaseAddressFn baseAddress;
    const CFStringRef *widthKey;
    const CFStringRef *heightKey;
    const CFStringRef *bytesPerElementKey;
    const CFStringRef *bytesPerRowKey;
    const CFStringRef *allocSizeKey;
    const CFStringRef *pixelFormatKey;
    BOOL ok;
} A12IOSurfaceAPI;

static NSString *A12String(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (__unused NSException *exception) { return @"<description threw>"; }
}

static NSError *A12Error(NSInteger code, NSString *message) {
    return [NSError errorWithDomain:@"com.invisiblestrangler.AnimaXS.A12ANE"
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey: message ?: @"A12 ANE error"}];
}

static NSUInteger A12Align64(NSUInteger value) { return (value + 63u) & ~63u; }
static NSUInteger A12PlaneStrideBytes(NSUInteger spatial) { return A12Align64(spatial * 2u); }
static NSUInteger A12AllocationBytes(NSUInteger channels, NSUInteger spatial) {
    return channels * A12PlaneStrideBytes(spatial);
}

static double A12Milliseconds(uint64_t delta) {
    static mach_timebase_info_data_t timebase;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{ mach_timebase_info(&timebase); });
    return (double)delta * (double)timebase.numer / (double)timebase.denom / 1e6;
}

static A12IOSurfaceAPI A12IOSurface(void) {
    static A12IOSurfaceAPI api;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        api.handle = dlopen("/System/Library/Frameworks/IOSurface.framework/IOSurface", RTLD_NOW | RTLD_LOCAL);
        if (!api.handle) return;
        api.create = (IOSurfaceCreateFn)dlsym(api.handle, "IOSurfaceCreate");
        api.lock = (IOSurfaceLockFn)dlsym(api.handle, "IOSurfaceLock");
        api.unlock = (IOSurfaceUnlockFn)dlsym(api.handle, "IOSurfaceUnlock");
        api.baseAddress = (IOSurfaceGetBaseAddressFn)dlsym(api.handle, "IOSurfaceGetBaseAddress");
        api.widthKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceWidth");
        api.heightKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceHeight");
        api.bytesPerElementKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerElement");
        api.bytesPerRowKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerRow");
        api.allocSizeKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceAllocSize");
        api.pixelFormatKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfacePixelFormat");
        api.ok = api.create && api.lock && api.unlock && api.baseAddress && api.widthKey &&
                 api.heightKey && api.bytesPerElementKey && api.bytesPerRowKey && api.allocSizeKey &&
                 api.pixelFormatKey;
    });
    return api;
}

static void *A12LoadANE(void) {
    static void *handle;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                        RTLD_NOW | RTLD_LOCAL);
    });
    return handle;
}

static NSDictionary *A12Options(void) {
    return @{ @"kANEModelKeyEspressoTranslationOptions": @{ @"compute_unit_mask": @5 },
              @"kANEFModelIdentityStrKey": @"compiled_0" };
}

BOOL A12ANEIsAvailable(void) {
    if (!A12LoadANE() || !A12IOSurface().ok) return NO;
    return NSClassFromString(@"_ANEModel") && NSClassFromString(@"_ANEClient") &&
           NSClassFromString(@"_ANERequest") && NSClassFromString(@"_ANEIOSurfaceObject");
}

NSString *A12ANERuntimeStatus(void) {
    if (!A12LoadANE()) return @"AppleNeuralEngine private framework unavailable";
    if (!A12IOSurface().ok) return @"IOSurface runtime unavailable";
    NSArray<NSString *> *names = @[@"_ANEModel", @"_ANEClient", @"_ANERequest", @"_ANEIOSurfaceObject"];
    for (NSString *name in names) if (!NSClassFromString(name)) return [@"Missing runtime class " stringByAppendingString:name];
    return @"available";
}

static A12IOSurfaceRef A12MakeRawSurface(NSUInteger bytes) {
    A12IOSurfaceAPI io = A12IOSurface();
    if (!io.ok) return NULL;
    NSDictionary *properties = @{
        (__bridge id)(*io.widthKey): @(bytes),
        (__bridge id)(*io.heightKey): @1,
        (__bridge id)(*io.bytesPerElementKey): @1,
        (__bridge id)(*io.bytesPerRowKey): @(bytes),
        (__bridge id)(*io.allocSizeKey): @(bytes),
        (__bridge id)(*io.pixelFormatKey): @0
    };
    return io.create((__bridge CFDictionaryRef)properties);
}

static id A12SurfaceObject(A12IOSurfaceRef surface) {
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!surfaceClass || !surface) return nil;
    return ((id(*)(Class,SEL,void *))objc_msgSend)(
        surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)surface);
}

static NSString *A12GenericKey(NSUInteger spatial, NSUInteger inputChannels, NSUInteger outputChannels) {
    return [NSString stringWithFormat:
        @"{\"isegment\":0,\"inputs\":{\"x\":{\"shape\":[%lu,1,1,%lu,1]}},\"outputs\":{\"y\":{\"shape\":[%lu,1,1,%lu,1]}}}",
        (unsigned long)spatial, (unsigned long)inputChannels,
        (unsigned long)spatial, (unsigned long)outputChannels];
}

static NSString *A12QKVKey(NSUInteger spatial, NSUInteger channels) {
    return [NSString stringWithFormat:
        @"{\"isegment\":0,\"inputs\":{\"x\":{\"shape\":[%lu,1,1,%lu,1]}},\"outputs\":{\"q\":{\"shape\":[%lu,1,1,%lu,1]},\"k\":{\"shape\":[%lu,1,1,%lu,1]},\"v\":{\"shape\":[%lu,1,1,%lu,1]}}}",
        (unsigned long)spatial, (unsigned long)channels,
        (unsigned long)spatial, (unsigned long)channels,
        (unsigned long)spatial, (unsigned long)channels,
        (unsigned long)spatial, (unsigned long)channels];
}

static NSDictionary *A12ModelAttributes(id model) {
    @try {
        id attrs = ((id(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        return [attrs isKindOfClass:NSDictionary.class] ? attrs : @{};
    } @catch (__unused NSException *exception) { return @{}; }
}

static NSArray<NSString *> *A12ModelOutputSymbols(id model) {
    NSDictionary *attrs = A12ModelAttributes(model);
    id desc = attrs[@"ANEFModelDescription"];
    id symbols = [desc isKindOfClass:NSDictionary.class] ? desc[@"kANEFModelOutputSymbolsArrayKey"] : nil;
    return [symbols isKindOfClass:NSArray.class] ? symbols : @[];
}

static BOOL A12LoadModel(id client, id model, NSError **error, double *milliseconds) {
    uint64_t start = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"loadModel:options:qos:error:"), model, A12Options(), 25, error);
    if (milliseconds) *milliseconds = A12Milliseconds(mach_absolute_time() - start);
    return ok;
}

static BOOL A12UnloadModel(id client, id model, NSError **error) {
    return ((BOOL(*)(id,SEL,id,id,unsigned int,NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"unloadModel:options:qos:error:"), model, A12Options(), 25, error);
}

static BOOL A12Evaluate(id client, id model, id request, NSError **error, double *milliseconds) {
    uint64_t start = mach_absolute_time();
    BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError **))objc_msgSend)(
        client, NSSelectorFromString(@"evaluateWithModel:options:request:qos:error:"),
        model, A12Options(), request, 25, error);
    if (milliseconds) *milliseconds = A12Milliseconds(mach_absolute_time() - start);
    return ok;
}

static NSString *A12SanitizeCacheKey(NSString *value) {
    NSCharacterSet *allowed = [NSCharacterSet characterSetWithCharactersInString:@"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."];
    NSMutableString *out = [NSMutableString stringWithCapacity:value.length];
    for (NSUInteger i = 0; i < value.length; ++i) {
        unichar c = [value characterAtIndex:i];
        [out appendString:[allowed characterIsMember:c] ? [NSString stringWithCharacters:&c length:1] : @"_"];
    }
    return out;
}

static NSURL *A12PreparedModelURL(NSString *cacheKey, NSError **error) {
    NSURL *template = [[NSBundle mainBundle] URLForResource:@"Conv2048W8Template" withExtension:@"bundle"];
    if (!template) {
        if (error) *error = A12Error(1, @"Bundled Conv2048W8Template.bundle is missing");
        return nil;
    }
    NSURL *caches = [NSURL fileURLWithPath:[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject]
                               isDirectory:YES];
    NSURL *base = [caches URLByAppendingPathComponent:@"AnimaXS-ANE" isDirectory:YES];
    if (![[NSFileManager defaultManager] createDirectoryAtURL:base withIntermediateDirectories:YES attributes:nil error:error]) return nil;
    NSString *safe = A12SanitizeCacheKey(cacheKey);
    NSURL *destination = [base URLByAppendingPathComponent:[safe stringByAppendingString:@".mlmodelc"] isDirectory:YES];
    if ([[NSFileManager defaultManager] fileExistsAtPath:destination.path]) return destination;
    NSURL *temporary = [base URLByAppendingPathComponent:[NSString stringWithFormat:@".%@-%@.tmp", safe, NSUUID.UUID.UUIDString] isDirectory:YES];
    if (![[NSFileManager defaultManager] copyItemAtURL:template toURL:temporary error:error]) return nil;
    return temporary; // caller patches then atomically moves it into destination
}

static BOOL A12FinalizePreparedModelURL(NSURL **modelURL, NSString *cacheKey, NSError **error) {
    NSURL *url = *modelURL;
    if (![url.lastPathComponent hasPrefix:@"."]) return YES; // cache hit
    NSURL *base = [url URLByDeletingLastPathComponent];
    NSURL *destination = [base URLByAppendingPathComponent:[A12SanitizeCacheKey(cacheKey) stringByAppendingString:@".mlmodelc"] isDirectory:YES];
    if ([[NSFileManager defaultManager] fileExistsAtPath:destination.path]) {
        [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
        *modelURL = destination;
        return YES;
    }
    if (![[NSFileManager defaultManager] moveItemAtURL:url toURL:destination error:error]) return NO;
    *modelURL = destination;
    return YES;
}

static BOOL A12WriteWeightFile(NSURL *modelURL, NSArray<NSData *> *payloads, NSString **why) {
    uint32_t count = (uint32_t)payloads.count;
    NSMutableData *weights = [NSMutableData dataWithLength:8 + (NSUInteger)count * 16];
    memcpy(weights.mutableBytes, &count, 4);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t section = i, size = (uint32_t)payloads[i].length;
        memcpy((uint8_t *)weights.mutableBytes + 8 + (NSUInteger)i * 16, &section, 4);
        memcpy((uint8_t *)weights.mutableBytes + 8 + (NSUInteger)i * 16 + 8, &size, 4);
    }
    for (NSData *payload in payloads) [weights appendData:payload];
    NSError *error = nil;
    if (![weights writeToURL:[modelURL URLByAppendingPathComponent:@"model.espresso.weights"]
                     options:NSDataWritingAtomic error:&error]) {
        if (why) *why = A12String(error);
        return NO;
    }
    return YES;
}

static BOOL A12WriteW8ConvBundle(NSURL *modelURL, NSString *layerName,
                                 NSUInteger inputChannels, NSUInteger outputChannels,
                                 NSUInteger spatial, NSData *q, NSData *bias, NSData *scale,
                                 NSString **why) {
    if (q.length != inputChannels * outputChannels ||
        bias.length != outputChannels * 4 || scale.length != outputChannels * 4) {
        if (why) *why = [NSString stringWithFormat:@"bad W8 payload %@ q=%lu expected=%lu bias=%lu/%lu scale=%lu/%lu",
            layerName, (unsigned long)q.length, (unsigned long)(inputChannels * outputChannels),
            (unsigned long)bias.length, (unsigned long)(outputChannels * 4),
            (unsigned long)scale.length, (unsigned long)(outputChannels * 4)];
        return NO;
    }
    NSString *net = [NSString stringWithFormat:
        @"{\"storage\":\"model.espresso.weights\",\"analyses\":{},\"properties\":{},\"format_version\":200,\"metadata_in_weights\":[],\"layers\":[{\"pad_r\":0,\"fused_relu\":0,\"fused_tanh\":0,\"debug_info\":\"%@\",\"pad_fill_mode\":0,\"pad_b\":0,\"pad_l\":0,\"top\":\"y\",\"K\":%lu,\"name\":\"%@\",\"has_batch_norm\":0,\"type\":\"convolution\",\"n_groups\":1,\"pad_t\":0,\"has_biases\":0,\"C\":%lu,\"bottom\":\"x\",\"weights\":{\"W_U8\":1,\"per_ch_qscale\":5,\"per_ch_qbias\":3},\"Nx\":1,\"pad_mode\":0,\"pad_value\":0,\"Ny\":1,\"n_parallel\":1,\"attributes\":{\"is_output\":1}}]}",
        layerName, (unsigned long)inputChannels, layerName, (unsigned long)outputChannels];
    NSString *shape = [NSString stringWithFormat:
        @"{\"layer_shapes\":{\"x\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1},\"y\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1}}}",
        (unsigned long)inputChannels, (unsigned long)spatial,
        (unsigned long)outputChannels, (unsigned long)spatial];
    NSError *error = nil;
    if (![[net dataUsingEncoding:NSUTF8StringEncoding] writeToURL:[modelURL URLByAppendingPathComponent:@"model.espresso.net"] options:NSDataWritingAtomic error:&error] ||
        ![[shape dataUsingEncoding:NSUTF8StringEncoding] writeToURL:[modelURL URLByAppendingPathComponent:@"model.espresso.shape"] options:NSDataWritingAtomic error:&error]) {
        if (why) *why = A12String(error);
        return NO;
    }
    return A12WriteWeightFile(modelURL,
        @[[NSMutableData dataWithLength:24], q, [NSData data], bias, [NSData data], scale], why);
}

static BOOL A12WriteFusedQKVBundle(NSURL *modelURL, NSUInteger channels, NSUInteger spatial,
                                   NSData *qQ, NSData *qBias, NSData *qScale,
                                   NSData *kQ, NSData *kBias, NSData *kScale,
                                   NSData *vQ, NSData *vBias, NSData *vScale,
                                   NSString **why) {
    NSUInteger matrixBytes = channels * channels;
    NSUInteger vectorBytes = channels * sizeof(float);
    if (qQ.length != matrixBytes || kQ.length != matrixBytes || vQ.length != matrixBytes ||
        qBias.length != vectorBytes || kBias.length != vectorBytes || vBias.length != vectorBytes ||
        qScale.length != vectorBytes || kScale.length != vectorBytes || vScale.length != vectorBytes) {
        if (why) *why = @"bad fused QKV W8 payload sizes";
        return NO;
    }
    NSString *(^layer)(NSString *, NSString *, int, int, int) =
    ^NSString *(NSString *name, NSString *top, int wi, int bi, int si) {
        return [NSString stringWithFormat:
            @"{\"pad_r\":0,\"fused_relu\":0,\"fused_tanh\":0,\"debug_info\":\"%@\",\"pad_fill_mode\":0,\"pad_b\":0,\"pad_l\":0,\"top\":\"%@\",\"K\":%lu,\"name\":\"%@\",\"has_batch_norm\":0,\"type\":\"convolution\",\"n_groups\":1,\"pad_t\":0,\"has_biases\":0,\"C\":%lu,\"bottom\":\"x\",\"weights\":{\"W_U8\":%d,\"per_ch_qscale\":%d,\"per_ch_qbias\":%d},\"Nx\":1,\"pad_mode\":0,\"pad_value\":0,\"Ny\":1,\"n_parallel\":1,\"attributes\":{\"is_output\":1}}",
            name, top, (unsigned long)channels, name, (unsigned long)channels, wi, si, bi];
    };
    NSString *net = [NSString stringWithFormat:
        @"{\"storage\":\"model.espresso.weights\",\"analyses\":{},\"properties\":{},\"format_version\":200,\"metadata_in_weights\":[],\"layers\":[%@,%@,%@]}",
        layer(@"qconv", @"q", 1, 3, 5), layer(@"kconv", @"k", 7, 9, 11), layer(@"vconv", @"v", 13, 15, 17)];
    NSString *shape = [NSString stringWithFormat:
        @"{\"layer_shapes\":{\"x\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1},\"q\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1},\"k\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1},\"v\":{\"k\":%lu,\"w\":%lu,\"n\":1,\"_rank\":4,\"h\":1}}}",
        (unsigned long)channels, (unsigned long)spatial,
        (unsigned long)channels, (unsigned long)spatial,
        (unsigned long)channels, (unsigned long)spatial,
        (unsigned long)channels, (unsigned long)spatial];
    NSError *error = nil;
    if (![[net dataUsingEncoding:NSUTF8StringEncoding] writeToURL:[modelURL URLByAppendingPathComponent:@"model.espresso.net"] options:NSDataWritingAtomic error:&error] ||
        ![[shape dataUsingEncoding:NSUTF8StringEncoding] writeToURL:[modelURL URLByAppendingPathComponent:@"model.espresso.shape"] options:NSDataWritingAtomic error:&error]) {
        if (why) *why = A12String(error);
        return NO;
    }
    return A12WriteWeightFile(modelURL, @[
        [NSMutableData dataWithLength:24], qQ, [NSData data], qBias, [NSData data], qScale,
        [NSMutableData dataWithLength:24], kQ, [NSData data], kBias, [NSData data], kScale,
        [NSMutableData dataWithLength:24], vQ, [NSData data], vBias, [NSData data], vScale
    ], why);
}

@interface A12ANESurface () {
    A12IOSurfaceRef _surface;
}
- (A12IOSurfaceRef)a12SurfaceRef;
@end

@implementation A12ANESurface
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device channels:(NSUInteger)channels spatial:(NSUInteger)spatial error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
    if (!device || channels == 0 || spatial == 0) {
        if (error) *error = A12Error(20, @"Invalid ANE surface shape/device");
        return nil;
    }
    _channels = channels;
    _spatial = spatial;
    _planeStrideElements = A12PlaneStrideBytes(spatial) / 2;
    _byteCount = A12AllocationBytes(channels, spatial);
    _surface = A12MakeRawSurface(_byteCount);
    if (!_surface) {
        if (error) *error = A12Error(21, @"IOSurface allocation failed");
        return nil;
    }
    A12IOSurfaceAPI io = A12IOSurface();
    io.lock(_surface, 0, NULL);
    void *base = io.baseAddress(_surface);
    memset(base, 0, _byteCount);
    io.unlock(_surface, 0, NULL);
    vm_size_t page = vm_page_size;
    if (((uintptr_t)base % page) != 0 || (_byteCount % page) != 0) {
        if (error) *error = A12Error(22, [NSString stringWithFormat:@"IOSurface is not page aligned: ptr=%p bytes=%lu page=%lu", base, (unsigned long)_byteCount, (unsigned long)page]);
        CFRelease(_surface); _surface = NULL;
        return nil;
    }
    _metalBuffer = [device newBufferWithBytesNoCopy:base length:_byteCount
                                             options:MTLResourceStorageModeShared deallocator:nil];
    if (!_metalBuffer) {
        if (error) *error = A12Error(23, @"MTLBuffer bytesNoCopy rejected ANE IOSurface allocation");
        CFRelease(_surface); _surface = NULL;
        return nil;
    }
    return self;
}
- (A12IOSurfaceRef)a12SurfaceRef { return _surface; }
- (void)dealloc { _metalBuffer = nil; if (_surface) CFRelease(_surface); }
@end

@interface A12ANEProjectionModel ()
@property(nonatomic, strong) id model;
@property(nonatomic, strong) id client;
@property(nonatomic, strong) NSURL *modelURL;
@property(nonatomic) BOOL loaded;
@end

@implementation A12ANEProjectionModel
- (nullable instancetype)initWithQBytes:(NSData *)qBytes biasF32:(NSData *)biasF32 scaleF32:(NSData *)scaleF32
                          inputChannels:(NSUInteger)inputChannels outputChannels:(NSUInteger)outputChannels
                                spatial:(NSUInteger)spatial label:(NSString *)label cacheKey:(NSString *)cacheKey
                                  error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
    if (!A12ANEIsAvailable()) {
        if (error) *error = A12Error(30, A12ANERuntimeStatus());
        return nil;
    }
    _inputChannels = inputChannels; _outputChannels = outputChannels; _spatial = spatial;
    _label = [label copy];
    NSError *fileError = nil;
    NSURL *url = A12PreparedModelURL(cacheKey, &fileError);
    if (!url) { if (error) *error = fileError ?: A12Error(31, @"ANE template preparation failed"); return nil; }
    BOOL isTemporary = [url.lastPathComponent hasPrefix:@"."];
    if (isTemporary) {
        NSString *why = nil;
        if (!A12WriteW8ConvBundle(url, label, inputChannels, outputChannels, spatial, qBytes, biasF32, scaleF32, &why)) {
            [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
            if (error) *error = A12Error(32, why ?: @"ANE model patch failed");
            return nil;
        }
        if (!A12FinalizePreparedModelURL(&url, cacheKey, &fileError)) {
            [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
            if (error) *error = fileError ?: A12Error(33, @"ANE model cache finalize failed");
            return nil;
        }
    }
    Class modelClass = NSClassFromString(@"_ANEModel");
    Class clientClass = NSClassFromString(@"_ANEClient");
    _model = ((id(*)(Class,SEL,id,id))objc_msgSend)(modelClass, NSSelectorFromString(@"modelAtURL:key:"),
                                                    url, A12GenericKey(spatial, inputChannels, outputChannels));
    _client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, NSSelectorFromString(@"sharedConnection"));
    if (!_model || !_client) {
        if (error) *error = A12Error(34, @"ANE model/client construction failed");
        return nil;
    }
    NSError *loadError = nil; double loadMs = 0;
    if (!A12LoadModel(_client, _model, &loadError, &loadMs)) {
        if (error) *error = loadError ?: A12Error(35, @"ANE model load failed");
        return nil;
    }
    _loadMilliseconds = loadMs;
    _loaded = YES;
    _modelURL = url;
    return self;
}

- (BOOL)evaluateInput:(A12ANESurface *)input output:(A12ANESurface *)output milliseconds:(double *)milliseconds error:(NSError **)error {
    if (!_loaded || !_model || !_client) { if (error) *error = A12Error(40, @"ANE model is not loaded"); return NO; }
    if (input.channels != _inputChannels || input.spatial != _spatial ||
        output.channels != _outputChannels || output.spatial != _spatial) {
        if (error) *error = A12Error(41, @"ANE surface shape does not match projection model");
        return NO;
    }
    id inputObject = A12SurfaceObject([input a12SurfaceRef]);
    id outputObject = A12SurfaceObject([output a12SurfaceRef]);
    Class requestClass = NSClassFromString(@"_ANERequest");
    if (!inputObject || !outputObject || !requestClass) { if (error) *error = A12Error(42, @"ANE request surface wrapping failed"); return NO; }
    id request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
        requestClass, NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
        @[inputObject], @[@0], @[outputObject], @[@0], @0);
    if (!request) { if (error) *error = A12Error(43, @"ANE request construction failed"); return NO; }
    NSError *evaluationError = nil;
    BOOL ok = A12Evaluate(_client, _model, request, &evaluationError, milliseconds);
    if (!ok && error) *error = evaluationError ?: A12Error(44, @"ANE evaluation failed");
    return ok;
}

- (void)invalidate {
    if (!_loaded) return;
    NSError *error = nil;
    A12UnloadModel(_client, _model, &error);
    _loaded = NO;
    _model = nil;
}
- (void)dealloc { [self invalidate]; }
@end

static id A12QKVOutputObject(NSString *symbol, id q, id k, id v) {
    if ([symbol isEqualToString:@"q"] || [symbol isEqualToString:@"q@output"]) return q;
    if ([symbol isEqualToString:@"k"] || [symbol isEqualToString:@"k@output"]) return k;
    return v;
}

@interface A12ANEQKVModel ()
@property(nonatomic, strong) id model;
@property(nonatomic, strong) id client;
@property(nonatomic, strong) NSURL *modelURL;
@property(nonatomic) BOOL loaded;
@end

@implementation A12ANEQKVModel
- (nullable instancetype)initWithQBytes:(NSData *)qBytes qBiasF32:(NSData *)qBiasF32 qScaleF32:(NSData *)qScaleF32
                                  kBytes:(NSData *)kBytes kBiasF32:(NSData *)kBiasF32 kScaleF32:(NSData *)kScaleF32
                                  vBytes:(NSData *)vBytes vBiasF32:(NSData *)vBiasF32 vScaleF32:(NSData *)vScaleF32
                                channels:(NSUInteger)channels spatial:(NSUInteger)spatial
                                   label:(NSString *)label cacheKey:(NSString *)cacheKey error:(NSError **)error {
    self = [super init];
    if (!self) return nil;
    if (!A12ANEIsAvailable()) { if (error) *error = A12Error(50, A12ANERuntimeStatus()); return nil; }
    _channels = channels; _spatial = spatial; _label = [label copy];
    NSError *fileError = nil;
    NSURL *url = A12PreparedModelURL(cacheKey, &fileError);
    if (!url) { if (error) *error = fileError ?: A12Error(51, @"ANE QKV template preparation failed"); return nil; }
    BOOL isTemporary = [url.lastPathComponent hasPrefix:@"."];
    if (isTemporary) {
        NSString *why = nil;
        if (!A12WriteFusedQKVBundle(url, channels, spatial,
                                    qBytes, qBiasF32, qScaleF32,
                                    kBytes, kBiasF32, kScaleF32,
                                    vBytes, vBiasF32, vScaleF32, &why)) {
            [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
            if (error) *error = A12Error(52, why ?: @"ANE fused QKV patch failed");
            return nil;
        }
        if (!A12FinalizePreparedModelURL(&url, cacheKey, &fileError)) {
            [[NSFileManager defaultManager] removeItemAtURL:url error:NULL];
            if (error) *error = fileError ?: A12Error(53, @"ANE QKV model cache finalize failed");
            return nil;
        }
    }
    Class modelClass = NSClassFromString(@"_ANEModel");
    Class clientClass = NSClassFromString(@"_ANEClient");
    _model = ((id(*)(Class,SEL,id,id))objc_msgSend)(modelClass, NSSelectorFromString(@"modelAtURL:key:"),
                                                    url, A12QKVKey(spatial, channels));
    _client = ((id(*)(Class,SEL))objc_msgSend)(clientClass, NSSelectorFromString(@"sharedConnection"));
    if (!_model || !_client) { if (error) *error = A12Error(54, @"ANE QKV model/client construction failed"); return nil; }
    NSError *loadError = nil; double loadMs = 0;
    if (!A12LoadModel(_client, _model, &loadError, &loadMs)) {
        if (error) *error = loadError ?: A12Error(55, @"ANE QKV model load failed");
        return nil;
    }
    _loadMilliseconds = loadMs; _loaded = YES; _modelURL = url;
    return self;
}

- (BOOL)evaluateInput:(A12ANESurface *)input qOutput:(A12ANESurface *)qOutput
               kOutput:(A12ANESurface *)kOutput vOutput:(A12ANESurface *)vOutput
          milliseconds:(double *)milliseconds error:(NSError **)error {
    if (!_loaded || !_model || !_client) { if (error) *error = A12Error(60, @"ANE QKV model is not loaded"); return NO; }
    NSArray *surfaces = @[input, qOutput, kOutput, vOutput];
    for (A12ANESurface *surface in surfaces) {
        if (surface.channels != _channels || surface.spatial != _spatial) {
            if (error) *error = A12Error(61, @"ANE QKV surface shape mismatch"); return NO;
        }
    }
    id inObject = A12SurfaceObject([input a12SurfaceRef]);
    id qObject = A12SurfaceObject([qOutput a12SurfaceRef]);
    id kObject = A12SurfaceObject([kOutput a12SurfaceRef]);
    id vObject = A12SurfaceObject([vOutput a12SurfaceRef]);
    NSArray<NSString *> *symbols = A12ModelOutputSymbols(_model);
    NSArray<NSString *> *order = symbols.count == 3 ? symbols : @[@"k@output", @"q@output", @"v@output"];
    NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:3];
    NSMutableArray *indices = [NSMutableArray arrayWithCapacity:3];
    for (NSUInteger i = 0; i < 3; ++i) {
        id object = A12QKVOutputObject(order[i], qObject, kObject, vObject);
        if (object) [outputs addObject:object];
        [indices addObject:@(i)];
    }
    Class requestClass = NSClassFromString(@"_ANERequest");
    if (!inObject || outputs.count != 3 || !requestClass) { if (error) *error = A12Error(62, @"ANE QKV request surface wrapping failed"); return NO; }
    id request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
        requestClass, NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
        @[inObject], @[@0], outputs, indices, @0);
    if (!request) { if (error) *error = A12Error(63, @"ANE QKV request construction failed"); return NO; }
    NSError *evaluationError = nil;
    BOOL ok = A12Evaluate(_client, _model, request, &evaluationError, milliseconds);
    if (!ok && error) *error = evaluationError ?: A12Error(64, @"ANE QKV evaluation failed");
    return ok;
}

- (void)invalidate {
    if (!_loaded) return;
    NSError *error = nil;
    A12UnloadModel(_client, _model, &error);
    _loaded = NO; _model = nil;
}
- (void)dealloc { [self invalidate]; }
@end
