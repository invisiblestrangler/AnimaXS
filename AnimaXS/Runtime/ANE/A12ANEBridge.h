#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>
#import <dispatch/dispatch.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

/// Runtime-only bridge for the A12/H11 Apple Neural Engine path proven by the
/// AnimaANEProbe v14 device harness. No private Apple headers are imported;
/// the private runtime is resolved dynamically and this backend remains an
/// opt-in, device-only experimental path.
FOUNDATION_EXPORT BOOL A12ANEIsAvailable(void);
FOUNDATION_EXPORT NSString *A12ANERuntimeStatus(void);
/// Diagnostic-only selector inventory for the private client.
FOUNDATION_EXPORT NSString *A12ANEClientCapabilitySummary(void);

/// Disk-cache preparation is intentionally separate from private ANE loading.
/// These functions only materialize the proven Espresso bundle under the app's
/// cache directory and are safe to call during explicit model import.
FOUNDATION_EXPORT BOOL A12ANEPreparedModelExists(NSString *cacheKey);
FOUNDATION_EXPORT BOOL A12ANEPrepareProjectionModel(
    NSData *qBytes, NSData *biasF32, NSData *scaleF32,
    NSUInteger inputChannels, NSUInteger outputChannels, NSUInteger spatial,
    NSString *label, NSString *cacheKey, NSError **error);
FOUNDATION_EXPORT BOOL A12ANEPrepareQKVModel(
    NSData *qBytes, NSData *qBiasF32, NSData *qScaleF32,
    NSData *kBytes, NSData *kBiasF32, NSData *kScaleF32,
    NSData *vBytes, NSData *vBiasF32, NSData *vScaleF32,
    NSUInteger channels, NSUInteger spatial, NSString *label,
    NSString *cacheKey, NSError **error);

@interface A12ANESurface : NSObject
@property(nonatomic, readonly) id<MTLBuffer> metalBuffer;
@property(nonatomic, readonly) NSUInteger channels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) NSUInteger planeStrideElements;
@property(nonatomic, readonly) NSUInteger byteCount;
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                               channels:(NSUInteger)channels
                                spatial:(NSUInteger)spatial
                                  error:(NSError **)error;
@end

@interface A12ANEProjectionModel : NSObject
@property(nonatomic, readonly) NSUInteger inputChannels;
@property(nonatomic, readonly) NSUInteger outputChannels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) double loadMilliseconds;
@property(nonatomic, readonly) NSString *label;
/// Diagnostic metadata reported by the loaded ANE program.
@property(nonatomic, readonly) NSUInteger procedureCount;
@property(nonatomic, readonly) NSString *procedureSummary;

/// qBytes is row-major U8 [outputChannels,inputChannels]. bias/scale are one
/// Float32 value per output channel, matching the H11 Espresso W8 ABI.
- (nullable instancetype)initWithQBytes:(NSData *)qBytes
                                biasF32:(NSData *)biasF32
                               scaleF32:(NSData *)scaleF32
                          inputChannels:(NSUInteger)inputChannels
                         outputChannels:(NSUInteger)outputChannels
                                spatial:(NSUInteger)spatial
                                  label:(NSString *)label
                               cacheKey:(NSString *)cacheKey
                                  error:(NSError **)error;

/// Loads an already-prepared model. Never writes model.espresso.weights.
- (nullable instancetype)initPreparedWithInputChannels:(NSUInteger)inputChannels
                                        outputChannels:(NSUInteger)outputChannels
                                               spatial:(NSUInteger)spatial
                                                 label:(NSString *)label
                                              cacheKey:(NSString *)cacheKey
                                                 error:(NSError **)error
    NS_SWIFT_NAME(init(preparedInputChannels:outputChannels:spatial:label:cacheKey:));

- (BOOL)evaluateInput:(A12ANESurface *)input
                output:(A12ANESurface *)output
           milliseconds:(nullable double *)milliseconds
                  error:(NSError **)error;
/// Diagnostic-only: unload ANE residency while retaining the same _ANEModel object.
- (BOOL)diagnosticUnloadKeepingModel;
/// Diagnostic-only: reload the retained _ANEModel without reconstructing it from disk.
/// Returns -1 on private-runtime failure.
- (double)diagnosticReloadMilliseconds;
- (void)invalidate;
@end

@interface A12ANEQKVModel : NSObject
@property(nonatomic, readonly) NSUInteger channels;
@property(nonatomic, readonly) NSUInteger spatial;
@property(nonatomic, readonly) double loadMilliseconds;
@property(nonatomic, readonly) NSString *label;

/// Fused self-attention Q/K/V projection. All three W8 matrices are
/// row-major U8 [channels,channels] with one Float32 bias/scale per row.
- (nullable instancetype)initWithQBytes:(NSData *)qBytes
                                qBiasF32:(NSData *)qBiasF32
                               qScaleF32:(NSData *)qScaleF32
                                  kBytes:(NSData *)kBytes
                                kBiasF32:(NSData *)kBiasF32
                               kScaleF32:(NSData *)kScaleF32
                                  vBytes:(NSData *)vBytes
                                vBiasF32:(NSData *)vBiasF32
                               vScaleF32:(NSData *)vScaleF32
                                channels:(NSUInteger)channels
                                 spatial:(NSUInteger)spatial
                                   label:(NSString *)label
                                cacheKey:(NSString *)cacheKey
                                   error:(NSError **)error;

/// Loads an already-prepared fused QKV model. Never writes weights.
- (nullable instancetype)initPreparedWithChannels:(NSUInteger)channels
                                           spatial:(NSUInteger)spatial
                                             label:(NSString *)label
                                          cacheKey:(NSString *)cacheKey
                                             error:(NSError **)error
    NS_SWIFT_NAME(init(preparedChannels:spatial:label:cacheKey:));

- (BOOL)evaluateInput:(A12ANESurface *)input
               qOutput:(A12ANESurface *)qOutput
               kOutput:(A12ANESurface *)kOutput
               vOutput:(A12ANESurface *)vOutput
          milliseconds:(nullable double *)milliseconds
                 error:(NSError **)error;
/// Diagnostic-only: unload ANE residency while retaining the same _ANEModel object.
- (BOOL)diagnosticUnloadKeepingModel;
/// Diagnostic-only: reload the retained _ANEModel without reconstructing it from disk.
/// Returns -1 on private-runtime failure.
- (double)diagnosticReloadMilliseconds;
- (void)invalidate;
@end

// ---------------------------------------------------------------------------
// EXPERIMENT BRANCH ONLY — private ANE multi-procedure proof-of-concept.
//
// V5 proved that Core ML can hold two public functions but then incorrectly
// tried to load the resulting MLProgram through the Espresso-only _ANEModel
// constructor. This probe uses the iOS 18 _ANEInMemoryModel MIL path instead.
// It reuses the tiny V5 `identity` + `double` MIL/weight asset, patches only
// its spatial shape 64 -> 128 in memory so each fp16 IOSurface is exactly one
// A12 16 KiB page, compiles/loads the model ONCE, and dispatches procedure 0
// and procedure 1 through `_ANERequest ... procedureIndex:`. A successful run
// must observe two private procedures and exact all-ones / all-twos outputs.
//
// It is intentionally a physical-device launch probe rather than production
// runtime code. The constructor is skipped on Simulator and XCTest. It unloads
// the tiny model after the measurement and leaves the daemon's compiled-model
// cache intact so a second launch can distinguish cold compile from warm load.
// Remove this whole section when the 28-program implementation supersedes it.
// ---------------------------------------------------------------------------

#if !TARGET_OS_SIMULATOR

typedef CFTypeRef A12MP_IOSurfaceRef;
typedef A12MP_IOSurfaceRef (*A12MP_IOSurfaceCreateFn)(CFDictionaryRef);
typedef int32_t (*A12MP_IOSurfaceLockFn)(A12MP_IOSurfaceRef, uint32_t, uint32_t *);
typedef int32_t (*A12MP_IOSurfaceUnlockFn)(A12MP_IOSurfaceRef, uint32_t, uint32_t *);
typedef void *(*A12MP_IOSurfaceGetBaseAddressFn)(A12MP_IOSurfaceRef);

typedef struct {
    void *handle;
    A12MP_IOSurfaceCreateFn create;
    A12MP_IOSurfaceLockFn lock;
    A12MP_IOSurfaceUnlockFn unlock;
    A12MP_IOSurfaceGetBaseAddressFn baseAddress;
    const CFStringRef *widthKey;
    const CFStringRef *heightKey;
    const CFStringRef *bytesPerElementKey;
    const CFStringRef *bytesPerRowKey;
    const CFStringRef *allocSizeKey;
    const CFStringRef *pixelFormatKey;
    BOOL ok;
} A12MP_IOSurfaceAPI;

static inline A12MP_IOSurfaceAPI A12MPGetIOSurfaceAPI(void) {
    A12MP_IOSurfaceAPI api = {0};
    api.handle = dlopen("/System/Library/Frameworks/IOSurface.framework/IOSurface",
                        RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) return api;
    api.create = (A12MP_IOSurfaceCreateFn)dlsym(api.handle, "IOSurfaceCreate");
    api.lock = (A12MP_IOSurfaceLockFn)dlsym(api.handle, "IOSurfaceLock");
    api.unlock = (A12MP_IOSurfaceUnlockFn)dlsym(api.handle, "IOSurfaceUnlock");
    api.baseAddress = (A12MP_IOSurfaceGetBaseAddressFn)dlsym(api.handle, "IOSurfaceGetBaseAddress");
    api.widthKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceWidth");
    api.heightKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceHeight");
    api.bytesPerElementKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerElement");
    api.bytesPerRowKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerRow");
    api.allocSizeKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceAllocSize");
    api.pixelFormatKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfacePixelFormat");
    api.ok = api.create && api.lock && api.unlock && api.baseAddress && api.widthKey &&
             api.heightKey && api.bytesPerElementKey && api.bytesPerRowKey &&
             api.allocSizeKey && api.pixelFormatKey;
    return api;
}

static inline A12MP_IOSurfaceRef A12MPMakeSurface(A12MP_IOSurfaceAPI api, NSUInteger bytes) {
    if (!api.ok) return NULL;
    NSDictionary *properties = @{
        (__bridge id)(*api.widthKey): @(bytes),
        (__bridge id)(*api.heightKey): @1,
        (__bridge id)(*api.bytesPerElementKey): @1,
        (__bridge id)(*api.bytesPerRowKey): @(bytes),
        (__bridge id)(*api.allocSizeKey): @(bytes),
        (__bridge id)(*api.pixelFormatKey): @0
    };
    return api.create((__bridge CFDictionaryRef)properties);
}

static inline NSString *A12MPTwoProcedureProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE private multi-procedure MIL POC v1",
            @"source=V5TwoProcedure identity+double; runtime patch spatial 64->128 for 16KiB A12 surfaces",
            nil];

        void *aneHandle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        if (!aneHandle) {
            [lines addObject:@"RESULT=FAIL stage=framework-load"];
            return [lines componentsJoinedByString:@"\n"];
        }

        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class inMemoryClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!descriptorClass || !inMemoryClass || !requestClass || !surfaceClass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL stage=class-discovery descriptor=%@ inMemory=%@ request=%@ surface=%@",
                descriptorClass ? @"yes" : @"no", inMemoryClass ? @"yes" : @"no",
                requestClass ? @"yes" : @"no", surfaceClass ? @"yes" : @"no"]];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *templateURL = [[NSBundle mainBundle] URLForResource:@"Conv2048W8Template"
                                                     withExtension:@"bundle"];
        NSURL *assetURL = [templateURL URLByAppendingPathComponent:@"V5TwoProcedure.mlmodelc"
                                                       isDirectory:YES];
        NSURL *milURL = [assetURL URLByAppendingPathComponent:@"model.mil"];
        NSURL *weightURL = [assetURL URLByAppendingPathComponent:@"weights/weight.bin"];
        NSError *ioError = nil;
        NSString *milText = [NSString stringWithContentsOfURL:milURL
                                                     encoding:NSUTF8StringEncoding
                                                        error:&ioError];
        NSData *weightData = [NSData dataWithContentsOfURL:weightURL options:0 error:&ioError];
        if (!templateURL || !milText || !weightData) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=fixture-read error=%@",
                ioError.localizedDescription ?: @"missing V5 asset"]];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSString *oldShape = @"[1, 64, 1, 64]";
        if ([milText rangeOfString:oldShape].location == NSNotFound) {
            [lines addObject:@"RESULT=FAIL stage=fixture-patch reason=expected V5 shape not found"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSString *patchedText = [milText stringByReplacingOccurrencesOfString:oldShape
                                                                   withString:@"[1, 64, 1, 128]"];
        NSData *patchedMIL = [patchedText dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *weights = @{
            @"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": weightData }
        };

        SEL descriptorSelector = NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:");
        SEL modelSelector = NSSelectorFromString(@"inMemoryModelWithDescriptor:");
        if (![descriptorClass respondsToSelector:descriptorSelector] ||
            ![inMemoryClass respondsToSelector:modelSelector]) {
            [lines addObject:@"RESULT=FAIL stage=in-memory-selector-discovery"];
            return [lines componentsJoinedByString:@"\n"];
        }

        id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            descriptorClass, descriptorSelector, patchedMIL, weights, nil);
        id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
            inMemoryClass, modelSelector, descriptor) : nil;
        if (!descriptor || !memoryModel) {
            [lines addObject:@"RESULT=FAIL stage=in-memory-model-construction"];
            return [lines componentsJoinedByString:@"\n"];
        }

        BOOL compiledCached = NO;
        SEL cachedSelector = NSSelectorFromString(@"compiledModelExists");
        if ([memoryModel respondsToSelector:cachedSelector]) {
            compiledCached = ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, cachedSelector);
        }
        [lines addObject:[NSString stringWithFormat:@"compiledCacheHit=%@",
            compiledCached ? @"yes" : @"no"]];

        NSString *tempDirectory = nil;
        double compileMS = 0.0;
        if (!compiledCached) {
            NSString *hexID = ((id(*)(id,SEL))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"hexStringIdentifier"));
            if (![hexID isKindOfClass:NSString.class] || hexID.length == 0) {
                [lines addObject:@"RESULT=FAIL stage=compiler-cache-key"];
                return [lines componentsJoinedByString:@"\n"];
            }
            tempDirectory = [NSTemporaryDirectory() stringByAppendingPathComponent:hexID];
            NSString *weightDirectory = [tempDirectory stringByAppendingPathComponent:@"weights"];
            NSFileManager *fm = NSFileManager.defaultManager;
            [fm removeItemAtPath:tempDirectory error:NULL];
            if (![fm createDirectoryAtPath:weightDirectory
               withIntermediateDirectories:YES attributes:nil error:&ioError] ||
                ![patchedMIL writeToFile:[tempDirectory stringByAppendingPathComponent:@"model.mil"]
                              options:NSDataWritingAtomic error:&ioError] ||
                ![weightData writeToFile:[weightDirectory stringByAppendingPathComponent:@"weight.bin"]
                              options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=compiler-materialize error=%@",
                    ioError.localizedDescription ?: @"unknown"]];
                return [lines componentsJoinedByString:@"\n"];
            }

            NSError *compileError = nil;
            NSTimeInterval started = NSDate.timeIntervalSinceReferenceDate;
            BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
                25u, @{}, &compileError);
            compileMS = (NSDate.timeIntervalSinceReferenceDate - started) * 1000.0;
            if (!compiled) {
                [lines addObject:[NSString stringWithFormat:
                    @"RESULT=FAIL stage=compile compileMs=%.1f error=%@",
                    compileMS, compileError.localizedDescription ?: @"unknown"]];
                [fm removeItemAtPath:tempDirectory error:NULL];
                return [lines componentsJoinedByString:@"\n"];
            }
        }
        [lines addObject:[NSString stringWithFormat:@"compileMs=%.1f", compileMS]];

        NSError *loadError = nil;
        NSTimeInterval loadStarted = NSDate.timeIntervalSinceReferenceDate;
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
            25u, @{}, &loadError);
        double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStarted) * 1000.0;
        if (!loaded) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=load loadMs=%.1f error=%@",
                loadMS, loadError.localizedDescription ?: @"unknown"]];
            if (tempDirectory) [NSFileManager.defaultManager removeItemAtPath:tempDirectory error:NULL];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:[NSString stringWithFormat:@"loadCount=1 loadMs=%.1f", loadMS]];

        NSString *(^loadedTest)(void) = ^NSString *{
            NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"modelAttributes"));
            NSDictionary *desc = [attrs[@"ANEFModelDescription"] isKindOfClass:NSDictionary.class]
                ? attrs[@"ANEFModelDescription"] : @{};
            NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
                ? desc[@"ANEFModelProcedures"] : @[];
            NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
                ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
            NSUInteger procedureCount = MAX(procedures.count, nameMap.count);
            [lines addObject:[NSString stringWithFormat:@"procedureCount=%lu names=%@",
                (unsigned long)procedureCount, nameMap]];
            if (procedureCount < 2) {
                return @"RESULT=FAIL stage=procedure-discovery expected=2";
            }

            id loweredModel = ((id(*)(id,SEL))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"model"));
            if (!loweredModel) return @"RESULT=FAIL stage=lowered-model-access";

            A12MP_IOSurfaceAPI io = A12MPGetIOSurfaceAPI();
            if (!io.ok) return @"RESULT=FAIL stage=iosurface-api";
            const NSUInteger elementCount = 64u * 128u;
            const NSUInteger surfaceBytes = elementCount * sizeof(uint16_t); // exactly 16 KiB
            A12MP_IOSurfaceRef inputSurface = A12MPMakeSurface(io, surfaceBytes);
            A12MP_IOSurfaceRef outputSurface = A12MPMakeSurface(io, surfaceBytes);
            if (!inputSurface || !outputSurface) {
                if (inputSurface) CFRelease(inputSurface);
                if (outputSurface) CFRelease(outputSurface);
                return @"RESULT=FAIL stage=iosurface-create";
            }

            io.lock(inputSurface, 0, NULL);
            uint16_t *inputBits = (uint16_t *)io.baseAddress(inputSurface);
            for (NSUInteger i = 0; i < elementCount; ++i) inputBits[i] = 0x3C00u; // fp16 1.0
            io.unlock(inputSurface, 0, NULL);

            id inputObject = ((id(*)(Class,SEL,void *))objc_msgSend)(
                surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)inputSurface);
            id outputObject = ((id(*)(Class,SEL,void *))objc_msgSend)(
                surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)outputSurface);
            if (!inputObject || !outputObject) {
                CFRelease(inputSurface); CFRelease(outputSurface);
                return @"RESULT=FAIL stage=iosurface-wrap";
            }

            NSMutableArray<NSString *> *classes = [NSMutableArray arrayWithCapacity:2];
            for (unsigned int procedure = 0; procedure < 2; ++procedure) {
                NSArray *inputIndices = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                    loweredModel, NSSelectorFromString(@"inputSymbolIndicesForProcedureIndex:"), procedure);
                NSArray *outputIndices = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
                    loweredModel, NSSelectorFromString(@"outputSymbolIndicesForProcedureIndex:"), procedure);
                if (![inputIndices isKindOfClass:NSArray.class] || inputIndices.count != 1 ||
                    ![outputIndices isKindOfClass:NSArray.class] || outputIndices.count != 1) {
                    CFRelease(inputSurface); CFRelease(outputSurface);
                    return [NSString stringWithFormat:
                        @"RESULT=FAIL stage=symbol-indices procedure=%u inputs=%@ outputs=%@",
                        procedure, inputIndices, outputIndices];
                }

                io.lock(outputSurface, 0, NULL);
                memset(io.baseAddress(outputSurface), 0, surfaceBytes);
                io.unlock(outputSurface, 0, NULL);

                id request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                    requestClass,
                    NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
                    @[inputObject], inputIndices, @[outputObject], outputIndices, @(procedure));
                if (!request) {
                    CFRelease(inputSurface); CFRelease(outputSurface);
                    return [NSString stringWithFormat:@"RESULT=FAIL stage=request procedure=%u", procedure];
                }

                NSError *evalError = nil;
                NSTimeInterval evalStarted = NSDate.timeIntervalSinceReferenceDate;
                BOOL evalOK = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError **))objc_msgSend)(
                    memoryModel, NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
                    25u, @{}, request, &evalError);
                double evalMS = (NSDate.timeIntervalSinceReferenceDate - evalStarted) * 1000.0;
                if (!evalOK) {
                    CFRelease(inputSurface); CFRelease(outputSurface);
                    return [NSString stringWithFormat:
                        @"RESULT=FAIL stage=evaluate procedure=%u evalMs=%.1f error=%@",
                        procedure, evalMS, evalError.localizedDescription ?: @"unknown"];
                }

                io.lock(outputSurface, 0, NULL);
                const uint16_t *outputBits = (const uint16_t *)io.baseAddress(outputSurface);
                NSUInteger ones = 0, twos = 0;
                for (NSUInteger i = 0; i < elementCount; ++i) {
                    ones += outputBits[i] == 0x3C00u;
                    twos += outputBits[i] == 0x4000u;
                }
                io.unlock(outputSurface, 0, NULL);

                NSString *classification = nil;
                if (ones == elementCount) classification = @"identity";
                else if (twos == elementCount) classification = @"double";
                else classification = @"other";
                [classes addObject:classification];
                [lines addObject:[NSString stringWithFormat:
                    @"procedure%u evalMs=%.1f class=%@ exactOnes=%lu exactTwos=%lu/%lu indices=%@->%@",
                    procedure, evalMS, classification, (unsigned long)ones,
                    (unsigned long)twos, (unsigned long)elementCount,
                    inputIndices, outputIndices]];
            }

            CFRelease(inputSurface);
            CFRelease(outputSurface);
            NSSet *observed = [NSSet setWithArray:classes];
            if (observed.count == 2 && [observed containsObject:@"identity"] &&
                [observed containsObject:@"double"]) {
                return @"RESULT=PASS privateProcedures=2 modelLoads=1 procedureDispatch=identity+double";
            }
            return [NSString stringWithFormat:
                @"RESULT=FAIL stage=output-validation classes=%@", classes];
        };

        NSString *coreResult = loadedTest();
        NSError *unloadError = nil;
        NSTimeInterval unloadStarted = NSDate.timeIntervalSinceReferenceDate;
        BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
        double unloadMS = (NSDate.timeIntervalSinceReferenceDate - unloadStarted) * 1000.0;
        [lines addObject:[NSString stringWithFormat:@"unloadCount=1 unloadMs=%.1f ok=%@%@",
            unloadMS, unloaded ? @"yes" : @"no",
            unloaded ? @"" : [NSString stringWithFormat:@" error=%@", unloadError.localizedDescription ?: @"unknown"]]];
        [lines addObject:coreResult];
        if (tempDirectory) [NSFileManager.defaultManager removeItemAtPath:tempDirectory error:NULL];
        return [lines componentsJoinedByString:@"\n"];
    }
}

__attribute__((constructor))
static void A12MPInstallTwoProcedureLaunchProbe(void) {
    if (NSProcessInfo.processInfo.environment[@"XCTestConfigurationFilePath"] != nil) return;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSString *result = A12MPTwoProcedureProbe();
            NSLog(@"\n========== ANIMAXS_ANE_MULTIPROC_POC ==========\n%@\n=================================================\n", result);
        }
    });
}

#endif // !TARGET_OS_SIMULATOR

NS_ASSUME_NONNULL_END
