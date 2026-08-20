#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdint.h>
#include <string.h>

// Experiment branch only.
//
// Native netplist multi-procedure POC v2. Apple MLCompute's private ANE plist
// builder appends one complete network body and one matching ProcedureList entry
// as a pair. V1 incorrectly pointed two procedures at one shared network body;
// ANEC rejected that shape as InvalidProcedure. V2 mirrors Apple's grammar:
//
//   Networks = [network_identity, network_double]
//   ProcedureList = [procedure_identity -> network_identity,
//                    procedure_double   -> network_double]
//
// Both procedures take one external FP16 `x` input. identity computes ReLU(x),
// while double computes Add(x,x). x is filled with +1, so exact outputs are +1
// and +2 respectively. Tensor shape 64x1x128 FP16 is exactly 16 KiB, avoiding
// the A12 small-IOSurface failure discovered by earlier probes.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE native netplist multi-procedure POC v2\nRESULT=SKIP simulator";
}
#else

typedef CFTypeRef A12MPNetSurfaceRef;
typedef A12MPNetSurfaceRef (*A12MPNetSurfaceCreateFn)(CFDictionaryRef);
typedef int32_t (*A12MPNetSurfaceLockFn)(A12MPNetSurfaceRef, uint32_t, uint32_t *);
typedef int32_t (*A12MPNetSurfaceUnlockFn)(A12MPNetSurfaceRef, uint32_t, uint32_t *);
typedef void *(*A12MPNetSurfaceBaseFn)(A12MPNetSurfaceRef);

typedef struct {
    void *handle;
    A12MPNetSurfaceCreateFn create;
    A12MPNetSurfaceLockFn lock;
    A12MPNetSurfaceUnlockFn unlock;
    A12MPNetSurfaceBaseFn base;
    const CFStringRef *widthKey;
    const CFStringRef *heightKey;
    const CFStringRef *bytesPerElementKey;
    const CFStringRef *bytesPerRowKey;
    const CFStringRef *allocSizeKey;
    const CFStringRef *pixelFormatKey;
    BOOL ok;
} A12MPNetSurfaceAPI;

static inline NSString *A12MPNetDesc(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (NSException *e) {
        return [NSString stringWithFormat:@"<description threw %@: %@>",
            e.name ?: @"?", e.reason ?: @"?"];
    }
}

static inline NSString *A12MPNetError(NSError *error) {
    if (!error) return @"(nil)";
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    NSError *cursor = error;
    for (NSUInteger depth = 0; cursor && depth < 5; ++depth) {
        [parts addObject:[NSString stringWithFormat:@"%@[%ld] %@",
            cursor.domain ?: @"?", (long)cursor.code,
            cursor.localizedDescription ?: A12MPNetDesc(cursor)]];
        id next = cursor.userInfo[NSUnderlyingErrorKey];
        cursor = [next isKindOfClass:NSError.class] ? next : nil;
    }
    return [parts componentsJoinedByString:@" | underlying: "];
}

static inline A12MPNetSurfaceAPI A12MPNetSurfaceRuntime(void) {
    static A12MPNetSurfaceAPI api;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        api.handle = dlopen("/System/Library/Frameworks/IOSurface.framework/IOSurface",
                            RTLD_NOW | RTLD_LOCAL);
        if (!api.handle) return;
        api.create = (A12MPNetSurfaceCreateFn)dlsym(api.handle, "IOSurfaceCreate");
        api.lock = (A12MPNetSurfaceLockFn)dlsym(api.handle, "IOSurfaceLock");
        api.unlock = (A12MPNetSurfaceUnlockFn)dlsym(api.handle, "IOSurfaceUnlock");
        api.base = (A12MPNetSurfaceBaseFn)dlsym(api.handle, "IOSurfaceGetBaseAddress");
        api.widthKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceWidth");
        api.heightKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceHeight");
        api.bytesPerElementKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerElement");
        api.bytesPerRowKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceBytesPerRow");
        api.allocSizeKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfaceAllocSize");
        api.pixelFormatKey = (const CFStringRef *)dlsym(api.handle, "kIOSurfacePixelFormat");
        api.ok = api.create && api.lock && api.unlock && api.base && api.widthKey &&
                 api.heightKey && api.bytesPerElementKey && api.bytesPerRowKey &&
                 api.allocSizeKey && api.pixelFormatKey;
    });
    return api;
}

static inline A12MPNetSurfaceRef A12MPNetMakeSurface(A12MPNetSurfaceAPI api, NSUInteger bytes) {
    if (!api.ok || bytes == 0) return NULL;
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

static inline void A12MPNetFill(A12MPNetSurfaceAPI api,
                                 A12MPNetSurfaceRef surface,
                                 uint16_t halfBits,
                                 NSUInteger elements) {
    api.lock(surface, 0, NULL);
    uint16_t *dst = (uint16_t *)api.base(surface);
    for (NSUInteger i = 0; i < elements; ++i) dst[i] = halfBits;
    api.unlock(surface, 0, NULL);
}

static inline NSDictionary *A12MPNetInput(NSString *symbol,
                                           NSString *unit,
                                           NSUInteger channels,
                                           NSUInteger width) {
    return @{
        @"BatchSize": @1,
        @"InputChannels": @(channels),
        @"InputDepth": @1,
        @"InputHeight": @1,
        @"InputInterleave": @1,
        @"InputName": symbol,
        @"InputType": @"Float16",
        @"InputWidth": @(width),
        @"Name": unit,
        @"OperationName": @"op0"
    };
}

static inline NSDictionary *A12MPNetOutput(NSString *symbol, NSString *unit) {
    return @{
        @"Name": unit,
        @"OperationName": @"op0",
        @"OutputInterleave": @1,
        @"OutputName": symbol,
        @"OutputType": @"Float16"
    };
}

static inline NSDictionary *A12MPNetNetworkBody(NSString *unitName,
                                                  NSDictionary *unit,
                                                  NSString *outputSymbol) {
    return @{
        unitName: unit,
        @"Units": @[unitName],
        @"Weights": @[@"weights.0"],
        outputSymbol: @{
            @"Bottom": unitName,
            @"OutputInterleave": @1,
            @"OutputName": outputSymbol,
            @"OutputType": @"Float16"
        }
    };
}

static inline NSDictionary *A12MPNetProcedure(NSString *procedureName,
                                               NSString *networkName,
                                               NSString *unitName,
                                               NSString *outputSymbol,
                                               NSUInteger channels,
                                               NSUInteger width) {
    return @{
        @"Name": procedureName,
        @"InputList": @[A12MPNetInput(@"x", unitName, channels, width)],
        @"OperationList": @[@{
            @"OperationName": @"op0",
            @"NetworkName": networkName
        }],
        @"OutputList": @[A12MPNetOutput(outputSymbol, unitName)]
    };
}

static inline NSData *A12MPNetBuildDescription(NSError **error) {
    const NSUInteger channels = 64;
    const NSUInteger width = 128;

    NSString *identityNetworkName = @"network_identity";
    NSString *doubleNetworkName = @"network_double";
    NSString *identityUnit = @"identity-1";
    NSString *doubleUnit = @"double-1";
    NSString *identityOutput = @"y_identity";
    NSString *doubleOutput = @"y_double";

    NSDictionary *identity = @{
        @"Bottom": @[@"x"],
        @"InputType": @[@"Float16"],
        @"Name": identityUnit,
        @"OutputChannels": @(channels),
        @"OutputType": @"Float16",
        @"Params": @{@"Type": @"ReLU"},
        @"Type": @"Neuron"
    };
    NSDictionary *doubler = @{
        @"Bottom": @[@"x", @"x"],
        @"InputType": @[@"Float16", @"Float16"],
        @"Name": doubleUnit,
        @"OutputChannels": @(channels),
        @"OutputType": @"Float16",
        @"Params": @{@"Type": @"Add"},
        @"Type": @"ElementWise"
    };

    NSDictionary *identityNetwork = A12MPNetNetworkBody(
        identityUnit, identity, identityOutput);
    NSDictionary *doubleNetwork = A12MPNetNetworkBody(
        doubleUnit, doubler, doubleOutput);
    NSDictionary *identityProcedure = A12MPNetProcedure(
        @"procedure_identity", identityNetworkName, identityUnit,
        identityOutput, channels, width);
    NSDictionary *doubleProcedure = A12MPNetProcedure(
        @"procedure_double", doubleNetworkName, doubleUnit,
        doubleOutput, channels, width);

    NSDictionary *plist = @{
        @"Version": @"1.0.10",
        @"Networks": @[identityNetworkName, doubleNetworkName],
        @"ProcedureList": @[identityProcedure, doubleProcedure],
        identityNetworkName: identityNetwork,
        doubleNetworkName: doubleNetwork
    };
    return [NSPropertyListSerialization dataWithPropertyList:plist
                                                       format:NSPropertyListBinaryFormat_v1_0
                                                      options:0
                                                        error:error];
}

static inline NSArray<NSNumber *> *A12MPNetIndexArray(id raw) {
    if ([raw isKindOfClass:NSArray.class]) return raw;
    if (![raw isKindOfClass:NSIndexSet.class]) return @[];
    NSIndexSet *set = raw;
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:set.count];
    NSUInteger index = set.firstIndex;
    while (index != NSNotFound) {
        [result addObject:@(index)];
        index = [set indexGreaterThanIndex:index];
    }
    return result;
}

static inline NSDictionary *A12MPNetDescriptionDictionary(id model) {
    @try {
        id attrs = ((id(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        id desc = [attrs isKindOfClass:NSDictionary.class] ? attrs[@"ANEFModelDescription"] : nil;
        return [desc isKindOfClass:NSDictionary.class] ? desc : @{};
    } @catch (__unused NSException *e) {
        return @{};
    }
}

static inline NSArray<NSNumber *> *A12MPNetProcedureIDs(NSDictionary *nameMap,
                                                         NSUInteger procedureCount) {
    NSMutableOrderedSet<NSNumber *> *ids = [NSMutableOrderedSet orderedSet];
    NSArray *sortedNames = [[nameMap allKeys] sortedArrayUsingSelector:@selector(compare:)];
    for (id key in sortedNames) {
        id raw = nameMap[key];
        if ([raw respondsToSelector:@selector(unsignedIntegerValue)]) {
            [ids addObject:@([raw unsignedIntegerValue])];
        }
    }
    if (ids.count == 0 && procedureCount == 2) {
        [ids addObject:@0];
        [ids addObject:@1];
    }
    return ids.array;
}

static inline NSString *A12MPNetDispatchProcedure(id memoryModel,
                                                   id loweredModel,
                                                   NSUInteger procedureIndex,
                                                   NSMutableArray<NSString *> *lines) {
    SEL inSel = NSSelectorFromString(@"inputSymbolIndicesForProcedureIndex:");
    SEL outSel = NSSelectorFromString(@"outputSymbolIndicesForProcedureIndex:");
    SEL infoSel = NSSelectorFromString(@"procedureInfoForProcedureIndex:");
    id rawInputs = [loweredModel respondsToSelector:inSel]
        ? ((id(*)(id,SEL,NSUInteger))objc_msgSend)(loweredModel, inSel, procedureIndex) : nil;
    id rawOutputs = [loweredModel respondsToSelector:outSel]
        ? ((id(*)(id,SEL,NSUInteger))objc_msgSend)(loweredModel, outSel, procedureIndex) : nil;
    id info = [loweredModel respondsToSelector:infoSel]
        ? ((id(*)(id,SEL,NSUInteger))objc_msgSend)(loweredModel, infoSel, procedureIndex) : nil;
    NSArray<NSNumber *> *inputIndices = A12MPNetIndexArray(rawInputs);
    NSArray<NSNumber *> *outputIndices = A12MPNetIndexArray(rawOutputs);
    [lines addObject:[NSString stringWithFormat:@"procedureID=%lu info=%@ inputs=%@ outputs=%@",
        (unsigned long)procedureIndex, A12MPNetDesc(info), inputIndices, outputIndices]];

    if (inputIndices.count != 1 || outputIndices.count != 1) {
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=symbol-indices",
            (unsigned long)procedureIndex]];
        return @"fail";
    }

    A12MPNetSurfaceAPI io = A12MPNetSurfaceRuntime();
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    Class requestClass = NSClassFromString(@"_ANERequest");
    if (!io.ok || !surfaceClass || !requestClass) {
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=runtime-surfaces",
            (unsigned long)procedureIndex]];
        return @"fail";
    }

    const NSUInteger elements = 64u * 128u;
    const NSUInteger bytes = elements * sizeof(uint16_t);
    A12MPNetSurfaceRef inputSurface = A12MPNetMakeSurface(io, bytes);
    A12MPNetSurfaceRef outputSurface = A12MPNetMakeSurface(io, bytes);
    if (!inputSurface || !outputSurface) {
        if (inputSurface) CFRelease(inputSurface);
        if (outputSurface) CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=surface-create",
            (unsigned long)procedureIndex]];
        return @"fail";
    }

    A12MPNetFill(io, inputSurface, 0x3C00u, elements);
    A12MPNetFill(io, outputSurface, 0x0000u, elements);
    id wrappedInput = ((id(*)(Class,SEL,void *))objc_msgSend)(
        surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)inputSurface);
    id wrappedOutput = ((id(*)(Class,SEL,void *))objc_msgSend)(
        surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)outputSurface);
    if (!wrappedInput || !wrappedOutput) {
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=surface-wrap",
            (unsigned long)procedureIndex]];
        return @"fail";
    }

    SEL requestSel = NSSelectorFromString(
        @"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:");
    if (![requestClass respondsToSelector:requestSel]) {
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=request-selector",
            (unsigned long)procedureIndex]];
        return @"fail";
    }
    id request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
        requestClass, requestSel,
        @[wrappedInput], inputIndices, @[wrappedOutput], outputIndices, @(procedureIndex));
    if (!request) {
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=request",
            (unsigned long)procedureIndex]];
        return @"fail";
    }

    NSError *mapError = nil;
    BOOL mapped = ((BOOL(*)(id,SEL,id,BOOL,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"mapIOSurfacesWithRequest:cacheInference:error:"),
        request, YES, &mapError);
    if (!mapped) {
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=map error=%@",
            (unsigned long)procedureIndex, A12MPNetError(mapError)]];
        return @"fail";
    }

    NSError *evalError = nil;
    NSTimeInterval evalStart = NSDate.timeIntervalSinceReferenceDate;
    BOOL evaluated = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
        25u, @{}, request, &evalError);
    double evalMS = (NSDate.timeIntervalSinceReferenceDate - evalStart) * 1000.0;
    if ([memoryModel respondsToSelector:NSSelectorFromString(@"unmapIOSurfacesWithRequest:")]) {
        ((void(*)(id,SEL,id))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unmapIOSurfacesWithRequest:"), request);
    }
    if (!evaluated) {
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        [lines addObject:[NSString stringWithFormat:
            @"procedureID=%lu RESULT=FAIL stage=evaluate evalMs=%.2f error=%@",
            (unsigned long)procedureIndex, evalMS, A12MPNetError(evalError)]];
        return @"fail";
    }

    io.lock(outputSurface, 0, NULL);
    const uint16_t *bits = (const uint16_t *)io.base(outputSurface);
    NSUInteger ones = 0, twos = 0, other = 0;
    uint16_t first = bits ? bits[0] : 0xffffu;
    for (NSUInteger i = 0; bits && i < elements; ++i) {
        if (bits[i] == 0x3C00u) ++ones;
        else if (bits[i] == 0x4000u) ++twos;
        else ++other;
    }
    io.unlock(outputSurface, 0, NULL);
    CFRelease(inputSurface);
    CFRelease(outputSurface);

    NSString *kind = @"other";
    if (ones == elements) kind = @"identity";
    else if (twos == elements) kind = @"double";
    [lines addObject:[NSString stringWithFormat:
        @"procedureID=%lu eval=PASS evalMs=%.2f class=%@ ones=%lu twos=%lu other=%lu/%lu first=0x%04x",
        (unsigned long)procedureIndex, evalMS, kind,
        (unsigned long)ones, (unsigned long)twos, (unsigned long)other,
        (unsigned long)elements, first]];
    return kind;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE native netplist multi-procedure POC v2",
            @"grammar=two Networks + two ProcedureList entries; one network per procedure",
            @"procedures=identity(ReLU x) + double(Add x,x), tensor=64x1x128 fp16",
            nil];

        void *aneHandle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        if (!aneHandle) {
            [lines addObject:@"RESULT=FAIL stage=framework-load"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSError *plistError = nil;
        NSData *net = A12MPNetBuildDescription(&plistError);
        if (!net) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=netplist error=%@",
                A12MPNetError(plistError)]];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:[NSString stringWithFormat:
            @"netplist=PASS bytes=%lu networks=2 procedures=2",
            (unsigned long)net.length]];

        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
        if (!descriptorClass || !memoryModelClass) {
            [lines addObject:@"RESULT=FAIL stage=class-discovery"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSData *dummyWeight = [NSMutableData dataWithLength:1024];
        NSDictionary *weights = @{
            @"weights.0": @{@"offset": @0, @"data": dummyWeight}
        };
        id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            descriptorClass,
            NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
            net, weights, nil);
        id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
            memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
        if (!descriptor || !memoryModel) {
            [lines addObject:@"RESULT=FAIL stage=descriptor-model-construction"];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"descriptor=PASS model=PASS"];

        BOOL cacheHit = NO;
        if ([memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]) {
            cacheHit = ((BOOL(*)(id,SEL))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"compiledModelExists"));
        }
        NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
            ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
        if (!cacheHit) {
            if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
                [lines addObject:@"RESULT=FAIL stage=local-model-path"];
                return [lines componentsJoinedByString:@"\n"];
            }
            NSError *ioError = nil;
            NSFileManager *fm = NSFileManager.defaultManager;
            BOOL materialized = [fm createDirectoryAtPath:localPath
                                withIntermediateDirectories:YES attributes:nil error:&ioError] &&
                [net writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                         options:NSDataWritingAtomic error:&ioError] &&
                [dummyWeight writeToFile:[localPath stringByAppendingPathComponent:@"weights.0"]
                                 options:NSDataWritingAtomic error:&ioError];
            if (!materialized) {
                [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=materialize error=%@",
                    A12MPNetError(ioError)]];
                return [lines componentsJoinedByString:@"\n"];
            }
        }
        [lines addObject:[NSString stringWithFormat:@"materialize=PASS cacheHit=%@ localPath=%@",
            cacheHit ? @"yes" : @"no", localPath ?: @"(nil)"]];

        double compileMS = 0.0;
        if (!cacheHit) {
            NSError *compileError = nil;
            NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
            BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
                memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
                25u, @{}, &compileError);
            compileMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
            if (!compiled) {
                [lines addObject:[NSString stringWithFormat:
                    @"compile=FAIL compileMs=%.2f error=%@",
                    compileMS, A12MPNetError(compileError)]];
                [lines addObject:@"RESULT=FAIL stage=compile"];
                return [lines componentsJoinedByString:@"\n"];
            }
        }
        [lines addObject:[NSString stringWithFormat:@"compile=PASS compileMs=%.2f", compileMS]];

        NSError *loadError = nil;
        NSTimeInterval loadStart = NSDate.timeIntervalSinceReferenceDate;
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
            25u, @{}, &loadError);
        double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStart) * 1000.0;
        if (!loaded) {
            [lines addObject:[NSString stringWithFormat:@"load=FAIL loadMs=%.2f error=%@",
                loadMS, A12MPNetError(loadError)]];
            [lines addObject:@"RESULT=FAIL stage=load"];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:[NSString stringWithFormat:@"load=PASS loadMs=%.2f loadCount=1", loadMS]];

        NSDictionary *desc = A12MPNetDescriptionDictionary(memoryModel);
        NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
            ? desc[@"ANEFModelProcedures"] : @[];
        NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
            ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
        NSArray *inputSymbols = [desc[@"kANEFModelInputSymbolsArrayKey"] isKindOfClass:NSArray.class]
            ? desc[@"kANEFModelInputSymbolsArrayKey"] : @[];
        NSArray *outputSymbols = [desc[@"kANEFModelOutputSymbolsArrayKey"] isKindOfClass:NSArray.class]
            ? desc[@"kANEFModelOutputSymbolsArrayKey"] : @[];
        NSUInteger procedureCount = MAX(procedures.count, nameMap.count);
        NSArray<NSNumber *> *procedureIDs = A12MPNetProcedureIDs(nameMap, procedureCount);
        [lines addObject:[NSString stringWithFormat:
            @"procedureCount=%lu names=%@ ids=%@ inputSymbols=%@ outputSymbols=%@ procedures=%@",
            (unsigned long)procedureCount, nameMap, procedureIDs,
            inputSymbols, outputSymbols, procedures]];

        id loweredModel = [memoryModel respondsToSelector:NSSelectorFromString(@"model")]
            ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"model")) : nil;
        NSMutableArray<NSString *> *classes = [NSMutableArray array];
        if (procedureCount == 2 && procedureIDs.count == 2 && loweredModel) {
            for (NSNumber *procedureID in procedureIDs) {
                NSString *kind = A12MPNetDispatchProcedure(
                    memoryModel, loweredModel, procedureID.unsignedIntegerValue, lines);
                [classes addObject:kind ?: @"fail"];
            }
        } else {
            [lines addObject:[NSString stringWithFormat:@"dispatch=SKIP reason=%@",
                !loweredModel ? @"lowered-model-missing" :
                (procedureCount != 2 ? @"procedureCount!=2" : @"procedureIDs!=2")]];
        }

        NSError *unloadError = nil;
        BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
        [lines addObject:[NSString stringWithFormat:@"unload=%@ unloadCount=%d error=%@",
            unloaded ? @"PASS" : @"FAIL", unloaded ? 1 : 0, A12MPNetError(unloadError)]];

        NSSet<NSString *> *observed = [NSSet setWithArray:classes];
        BOOL dispatchPASS = classes.count == 2 && observed.count == 2 &&
            [observed containsObject:@"identity"] && [observed containsObject:@"double"];
        BOOL finalPASS = procedureCount == 2 && procedureIDs.count == 2 &&
                         dispatchPASS && unloaded;
        [lines addObject:[NSString stringWithFormat:@"dispatch=%@ classes=%@",
            dispatchPASS ? @"PASS identity+double" : @"FAIL", classes]];
        [lines addObject:finalPASS
            ? @"RESULT=PASS private-multiprocedure loadCount=1 procedureCount=2 unloadCount=1"
            : @"RESULT=FAIL private-multiprocedure"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
