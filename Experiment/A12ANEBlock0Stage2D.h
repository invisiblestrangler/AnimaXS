#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <dlfcn.h>

// Stage 2D: Stage 2C proved that even the smallest two-real-W8-procedure pair
// makes the ANEC helper disappear, so size is not the cause. Isolate the exact
// boundary with one already-working block-0 self_o donor:
//   A) exact Espresso-dumped native IR (Version 1.0.9, no ProcedureList)
//   B) exact same IR with Version changed to 1.0.10 only
//   C) same IR with one synthesized ProcedureList entry
// If A fails, espresso_dump_ir output is not directly re-compilable through the
// in-memory ANEC path as currently materialized. If A/B pass and C fails, the
// procedure wrapper is the defect. If C passes, the already-observed 2-network
// failure is specifically in multi-network combination/namespacing.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_Stage2A_Base
#import "../AnimaXS/Runtime/ANE/A12ANEMultiProcProbeV2.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2D\nRESULT=SKIP simulator";
}
#else

typedef NS_ENUM(NSInteger, A12S2DCompileResult) {
    A12S2DCompilePass = 0,
    A12S2DCompileFail = 1,
    A12S2DHelperLost = 2,
};

static inline uint64_t A12S2DWeightBytes(NSDictionary *weightMap) {
    uint64_t total = 0;
    for (NSString *name in weightMap) {
        NSDictionary *entry = [weightMap[name] isKindOfClass:NSDictionary.class]
            ? weightMap[name] : nil;
        NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
        if (data) total += data.length;
    }
    return total;
}

static inline A12S2DCompileResult A12S2DCompileOnly(
    NSDictionary *plist,
    NSDictionary *weightMap,
    NSString *label,
    NSMutableArray<NSString *> *lines) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"%@ serialize=FAIL error=%@",
            label, A12S2Error(serializeError)]];
        return A12S2DCompileFail;
    }

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !memoryModelClass) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=class-discovery", label]];
        return A12S2DCompileFail;
    }

    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=descriptor-model", label]];
        return A12S2DCompileFail;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    NSError *ioError = nil;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=localModelPath", label]];
            return A12S2DCompileFail;
        }
        NSFileManager *fm = NSFileManager.defaultManager;
        if (![fm createDirectoryAtPath:localPath withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![netData writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL error=%@",
                label, A12S2Error(ioError)]];
            return A12S2DCompileFail;
        }
        for (NSString *name in weightMap) {
            NSDictionary *entry = [weightMap[name] isKindOfClass:NSDictionary.class]
                ? weightMap[name] : nil;
            NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
            if (!data) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL badWeight=%@",
                    label, name]];
                return A12S2DCompileFail;
            }
            if (![data writeToFile:[localPath stringByAppendingPathComponent:name]
                           options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL weight=%@ error=%@",
                    label, name, A12S2Error(ioError)]];
                return A12S2DCompileFail;
            }
        }
    }

    NSError *compileError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL compiled = cacheHit || ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
        25u, @{}, &compileError);
    double compileMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    uint64_t bytes = A12S2DWeightBytes(weightMap);
    if (!compiled) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ compile=FAIL ms=%.2f netBytes=%lu weightBytes=%llu errorDomain=%@ errorCode=%ld error=%@",
            label, compileMS, (unsigned long)netData.length, (unsigned long long)bytes,
            compileError.domain ?: @"(nil)", (long)compileError.code,
            A12S2Error(compileError)]];
        if ([compileError.domain isEqualToString:NSCocoaErrorDomain] && compileError.code == 4097) {
            [lines addObject:[NSString stringWithFormat:@"%@ classification=HELPER_LOST", label]];
            return A12S2DHelperLost;
        }
        return A12S2DCompileFail;
    }

    [lines addObject:[NSString stringWithFormat:
        @"%@ compile=PASS ms=%.2f cacheHit=%@ netBytes=%lu weightBytes=%llu",
        label, compileMS, cacheHit ? @"yes" : @"no", (unsigned long)netData.length,
        (unsigned long long)bytes]];

    if ([memoryModel respondsToSelector:NSSelectorFromString(@"purgeCompiledModel")]) {
        @try { ((void(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"purgeCompiledModel")); }
        @catch (__unused NSException *e) {}
    }
    return A12S2DCompilePass;
}

static inline NSString *A12S2DResultName(A12S2DCompileResult result) {
    switch (result) {
        case A12S2DCompilePass: return @"PASS";
        case A12S2DHelperLost: return @"HELPER_LOST";
        default: return @"FAIL";
    }
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2D",
            @"goal=isolate whether harvested real W8 IR, version 1.0.10, or ProcedureList first breaks ANEC",
            @"matrix=raw-v109 -> same-v110 -> same-v110+single-procedure; stop on helper loss",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *donor = donors[1]; // self_o: smallest ordinary 1-in/1-out real projection.
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2D-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSURL *dumpDir = [root URLByAppendingPathComponent:@"self_o" isDirectory:YES];

        NSMutableDictionary *raw = nil;
        if (!A12S2Lower(@[donor], dumpDir, @"single-self_o", lines, &raw)) {
            [lines addObject:@"RESULT=FAIL stage=lower-self_o"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSArray *networks = [raw[@"Networks"] isKindOfClass:NSArray.class] ? raw[@"Networks"] : @[];
        NSString *networkName = networks.count == 1 && [networks.firstObject isKindOfClass:NSString.class]
            ? networks.firstObject : nil;
        NSDictionary *network = networkName && [raw[networkName] isKindOfClass:NSDictionary.class]
            ? raw[networkName] : @{};
        [lines addObject:[NSString stringWithFormat:
            @"singleSchema version=%@ network=%@ units=%@ networkKeys=%@ topKeys=%@",
            raw[@"Version"] ?: @"?", networkName ?: @"(nil)", network[@"Units"] ?: @[],
            [[network.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","],
            [[raw.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","]]];

        NSMutableDictionary *weights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(raw, @[donor], dumpDir, weights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=normalize-self_o-weights"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *v109 = [raw mutableCopy];
        [v109 removeObjectForKey:@"ProcedureList"];
        A12S2DCompileResult a = A12S2DCompileOnly(v109, weights, @"trial-A-raw-v109", lines);
        if (a != A12S2DCompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=raw-espresso-dump result=%@", A12S2DResultName(a)]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *v110 = [v109 mutableCopy];
        v110[@"Version"] = @"1.0.10";
        A12S2DCompileResult b = A12S2DCompileOnly(v110, weights, @"trial-B-v110-only", lines);
        if (b != A12S2DCompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=version-1.0.10 result=%@", A12S2DResultName(b)]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *proc1 = [v110 mutableCopy];
        if (!A12S2EnsureProcedureList(proc1, @[donor], lines)) {
            [lines addObject:@"RESULT=FAIL stage=single-procedure-wrapper"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2DCompileResult c = A12S2DCompileOnly(proc1, weights, @"trial-C-single-procedure", lines);
        if (c != A12S2DCompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=single-procedure-wrapper result=%@", A12S2DResultName(c)]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        [lines addObject:@"RESULT=PASS single-real-W8-with-procedure compiles; known failure boundary is multi-network combination"];
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
