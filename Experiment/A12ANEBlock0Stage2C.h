#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>

// Stage 2C: isolate the Stage-2B ANEC helper failure without changing the
// lowered W8 arithmetic. Lower the eight production donors exactly once, then
// compile increasingly large real multi-procedure subsets. This distinguishes
// malformed multi-procedure grammar / QKV multi-output issues from a compiler
// resource threshold. Stop immediately if the XPC compiler helper disappears.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_Stage2B
#import "A12ANEBlock0Stage2B.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2C\nRESULT=SKIP simulator";
}
#else

typedef NS_ENUM(NSInteger, A12S2CCompileResult) {
    A12S2CCompilePass = 0,
    A12S2CCompileFail = 1,
    A12S2CHelperLost = 2,
};

static inline NSMutableDictionary *A12S2CSubsetPlist(
    NSDictionary *full,
    NSArray<NSNumber *> *indices,
    NSArray<NSDictionary *> *allDonors,
    NSMutableArray<NSDictionary *> **donorsOut,
    NSMutableArray<NSString *> *lines,
    NSString *label) {
    NSArray *fullNetworks = [full[@"Networks"] isKindOfClass:NSArray.class]
        ? full[@"Networks"] : @[];
    NSMutableDictionary *subset = [NSMutableDictionary dictionary];
    subset[@"Version"] = @"1.0.10";
    NSMutableArray<NSString *> *networks = [NSMutableArray arrayWithCapacity:indices.count];
    NSMutableArray<NSDictionary *> *donors = [NSMutableArray arrayWithCapacity:indices.count];
    for (NSNumber *rawIndex in indices) {
        NSUInteger i = rawIndex.unsignedIntegerValue;
        if (i >= fullNetworks.count || i >= allDonors.count) {
            [lines addObject:[NSString stringWithFormat:@"%@ subset=FAIL badIndex=%lu",
                label, (unsigned long)i]];
            return nil;
        }
        NSString *networkName = [fullNetworks[i] isKindOfClass:NSString.class]
            ? fullNetworks[i] : nil;
        NSDictionary *body = networkName && [full[networkName] isKindOfClass:NSDictionary.class]
            ? full[networkName] : nil;
        if (!networkName || !body) {
            [lines addObject:[NSString stringWithFormat:@"%@ subset=FAIL missingNetwork index=%lu",
                label, (unsigned long)i]];
            return nil;
        }
        subset[networkName] = [body mutableCopy];
        [networks addObject:networkName];
        [donors addObject:allDonors[i]];
    }
    subset[@"Networks"] = networks;
    if (!A12S2EnsureProcedureList(subset, donors, lines)) {
        [lines addObject:[NSString stringWithFormat:@"%@ subset=FAIL procedureWrapper", label]];
        return nil;
    }
    if (donorsOut) *donorsOut = donors;
    [lines addObject:[NSString stringWithFormat:@"%@ subset=PASS procedures=%lu indices=%@",
        label, (unsigned long)indices.count, indices]];
    return subset;
}

static inline uint64_t A12S2CWeightBytes(NSDictionary *weightMap) {
    uint64_t total = 0;
    for (NSString *name in weightMap) {
        id data = [weightMap[name] isKindOfClass:NSDictionary.class]
            ? weightMap[name][@"data"] : nil;
        if ([data isKindOfClass:NSData.class]) total += [(NSData *)data length];
    }
    return total;
}

static inline A12S2CCompileResult A12S2CCompileOnly(
    NSMutableDictionary *plist,
    NSDictionary *weightMap,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    BOOL keepCompiled) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"%@ serialize=FAIL error=%@",
            label, A12S2Error(serializeError)]];
        return A12S2CCompileFail;
    }

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !memoryModelClass) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=class-discovery", label]];
        return A12S2CCompileFail;
    }

    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=descriptor-model", label]];
        return A12S2CCompileFail;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    NSError *ioError = nil;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=localModelPath", label]];
            return A12S2CCompileFail;
        }
        NSFileManager *fm = NSFileManager.defaultManager;
        if (![fm createDirectoryAtPath:localPath withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![netData writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL error=%@",
                label, A12S2Error(ioError)]];
            return A12S2CCompileFail;
        }
        for (NSString *name in weightMap) {
            NSData *data = [weightMap[name] isKindOfClass:NSDictionary.class]
                ? weightMap[name][@"data"] : nil;
            if (![data isKindOfClass:NSData.class]) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL badWeight=%@",
                    label, name]];
                return A12S2CCompileFail;
            }
            NSString *path = [localPath stringByAppendingPathComponent:name];
            if (![data writeToFile:path options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL weight=%@ error=%@",
                    label, name, A12S2Error(ioError)]];
                return A12S2CCompileFail;
            }
        }
    }

    NSError *compileError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL compiled = cacheHit || ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
        25u, @{}, &compileError);
    double compileMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    uint64_t bytes = A12S2CWeightBytes(weightMap);
    if (!compiled) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ compile=FAIL ms=%.2f weightBytes=%llu errorDomain=%@ errorCode=%ld error=%@",
            label, compileMS, (unsigned long long)bytes,
            compileError.domain ?: @"(nil)", (long)compileError.code,
            A12S2Error(compileError)]];
        if ([compileError.domain isEqualToString:NSCocoaErrorDomain] && compileError.code == 4097) {
            [lines addObject:[NSString stringWithFormat:@"%@ classification=HELPER_LOST", label]];
            return A12S2CHelperLost;
        }
        return A12S2CCompileFail;
    }

    [lines addObject:[NSString stringWithFormat:
        @"%@ compile=PASS ms=%.2f cacheHit=%@ weightBytes=%llu",
        label, compileMS, cacheHit ? @"yes" : @"no", (unsigned long long)bytes]];

    if (!keepCompiled && [memoryModel respondsToSelector:NSSelectorFromString(@"purgeCompiledModel")]) {
        @try { ((void(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"purgeCompiledModel")); }
        @catch (__unused NSException *e) {}
    }
    return A12S2CCompilePass;
}

static inline BOOL A12S2CRunTrial(
    NSDictionary *full,
    NSArray<NSDictionary *> *allDonors,
    NSArray<NSNumber *> *indices,
    NSURL *root,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    A12S2CCompileResult *resultOut,
    NSMutableDictionary **plistOut,
    NSMutableDictionary **weightsOut) {
    NSMutableArray<NSDictionary *> *trialDonors = nil;
    NSMutableDictionary *trial = A12S2CSubsetPlist(
        full, indices, allDonors, &trialDonors, lines, label);
    if (!trial) {
        if (resultOut) *resultOut = A12S2CCompileFail;
        return NO;
    }
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    if (!A12S2NormalizeWeights(trial, trialDonors, root, weights, lines)) {
        [lines addObject:[NSString stringWithFormat:@"%@ RESULT=FAIL stage=weight-normalization", label]];
        if (resultOut) *resultOut = A12S2CCompileFail;
        return NO;
    }
    A12S2CCompileResult result = A12S2CCompileOnly(trial, weights, label, lines, YES);
    if (resultOut) *resultOut = result;
    if (plistOut) *plistOut = trial;
    if (weightsOut) *weightsOut = weights;
    return result == A12S2CCompilePass;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2C",
            @"goal=bisect Stage2B compiler helper failure using exact real W8 subsets",
            @"matrix=simplePair(1,2) -> qkvPair(0,1) -> first6 -> first7 -> all8; stop on helper loss",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2C-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *full = A12S2BBuildCombinedPlist(donors, root, lines);
        if (!full) {
            [lines addObject:@"RESULT=FAIL stage=lower-and-combine"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSArray<NSDictionary *> *trials = @[
            @{@"label": @"trial-simple-pair", @"indices": @[@1, @2]},
            @{@"label": @"trial-qkv-pair", @"indices": @[@0, @1]},
            @{@"label": @"trial-first6", @"indices": @[@0, @1, @2, @3, @4, @5]},
            @{@"label": @"trial-first7", @"indices": @[@0, @1, @2, @3, @4, @5, @6]},
            @{@"label": @"trial-all8", @"indices": @[@0, @1, @2, @3, @4, @5, @6, @7]},
        ];

        BOOL allPriorPassed = YES;
        BOOL all8Compiled = NO;
        NSMutableDictionary *all8Plist = nil;
        NSMutableDictionary *all8Weights = nil;
        for (NSDictionary *trialSpec in trials) {
            NSString *label = trialSpec[@"label"];
            NSArray<NSNumber *> *indices = trialSpec[@"indices"];
            A12S2CCompileResult result = A12S2CCompileFail;
            NSMutableDictionary *trialPlist = nil;
            NSMutableDictionary *trialWeights = nil;
            BOOL passed = A12S2CRunTrial(
                full, donors, indices, root, label, lines,
                &result, &trialPlist, &trialWeights);
            if ([label isEqualToString:@"trial-all8"] && passed) {
                all8Compiled = YES;
                all8Plist = trialPlist;
                all8Weights = trialWeights;
            }
            if (result == A12S2CHelperLost) {
                [lines addObject:[NSString stringWithFormat:
                    @"matrixStop=%@ reason=compiler-helper-lost", label]];
                allPriorPassed = NO;
                break;
            }
            if (!passed) {
                [lines addObject:[NSString stringWithFormat:
                    @"matrixStop=%@ reason=compile-failed-with-helper-alive", label]];
                allPriorPassed = NO;
                break;
            }
        }

        if (all8Compiled && all8Plist && all8Weights) {
            [lines addObject:@"all8Compile=PASS proceeding=load-procedure-count"];
            BOOL loadedPass = A12S2CompileLoad(all8Plist, all8Weights, lines);
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            if (loadedPass) return [lines componentsJoinedByString:@"\n"];
            [lines addObject:@"RESULT=FAIL stage=all8-load-after-compile-pass"];
            return [lines componentsJoinedByString:@"\n"];
        }

        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (allPriorPassed) {
            [lines addObject:@"RESULT=FAIL stage=matrix-ended-without-all8"];
        } else {
            [lines addObject:@"RESULT=DIAGNOSTIC compile-matrix-complete"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
