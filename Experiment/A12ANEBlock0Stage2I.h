#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import "A12ANEBlock0Stage2H.h"

// Stage 2I starts from two settled facts:
//   1) canonical real W8 self_o + cross_q compiles and loads as exactly two
//      procedures in one private ANE model;
//   2) the canonical all-eight block container is rejected by ANEC as
//      InvalidProcedure while the compiler helper remains healthy.
//
// This probe changes no arithmetic. It reuses the exact canonical Stage-2H
// network bodies and isolates the remaining structural dimensions:
//   * procedure count (>2) with identical I/O geometry;
//   * different-but-matching geometry within a pair;
//   * heterogeneous input/output geometry across procedures;
//   * self-QKV's three-output procedure, alone and mixed with self_o.
// If all property trials pass, it then grows a real prefix from 3 -> 8 to find
// the first interaction/count boundary. No diffusion/generation is involved.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2IProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2I\nRESULT=SKIP simulator";
}
#else

typedef NS_ENUM(NSInteger, A12S2IResult) {
    A12S2IPass = 0,
    A12S2IInvalidProcedure = 1,
    A12S2IOtherFailure = 2,
};

static inline NSArray<NSDictionary *> *A12S2IDonorsForIndices(
    NSArray<NSDictionary *> *allDonors,
    NSArray<NSNumber *> *indices) {
    NSMutableArray<NSDictionary *> *out = [NSMutableArray arrayWithCapacity:indices.count];
    for (NSNumber *raw in indices) {
        NSUInteger i = raw.unsignedIntegerValue;
        if (i >= allDonors.count) return nil;
        [out addObject:allDonors[i]];
    }
    return out;
}

static inline A12S2IResult A12S2IRunCompileTrial(
    NSDictionary *canonicalAll,
    NSArray<NSDictionary *> *allDonors,
    NSArray<NSNumber *> *indices,
    NSURL *root,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    NSMutableDictionary **compiledPlistOut,
    NSMutableDictionary **compiledWeightsOut) {

    NSMutableDictionary *trial = A12S2HSubset(canonicalAll, indices, lines, label);
    NSArray<NSDictionary *> *trialDonors = A12S2IDonorsForIndices(allDonors, indices);
    if (!trial || !trialDonors || trialDonors.count != indices.count) {
        [lines addObject:[NSString stringWithFormat:@"%@ RESULT=FAIL stage=subset", label]];
        return A12S2IOtherFailure;
    }

    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    if (!A12S2NormalizeWeights(trial, trialDonors, root, weights, lines)) {
        [lines addObject:[NSString stringWithFormat:@"%@ RESULT=FAIL stage=weights", label]];
        return A12S2IOtherFailure;
    }

    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:trial
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"%@ serialize=FAIL error=%@",
            label, A12S2Error(serializeError)]];
        return A12S2IOtherFailure;
    }

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weights, nil) : nil;
    id memoryModel = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=descriptor-model", label]];
        return A12S2IOtherFailure;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    NSError *ioError = nil;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL stage=localModelPath", label]];
            return A12S2IOtherFailure;
        }
        NSFileManager *fm = NSFileManager.defaultManager;
        if (![fm createDirectoryAtPath:localPath withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![netData writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL error=%@",
                label, A12S2Error(ioError)]];
            return A12S2IOtherFailure;
        }
        for (NSString *name in weights) {
            NSDictionary *entry = [weights[name] isKindOfClass:NSDictionary.class] ? weights[name] : nil;
            NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
            if (!data || ![data writeToFile:[localPath stringByAppendingPathComponent:name]
                                     options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL weight=%@ error=%@",
                    label, name, A12S2Error(ioError)]];
                return A12S2IOtherFailure;
            }
        }
    }

    NSError *compileError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL compiled = cacheHit || ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
        25u, @{}, &compileError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (!compiled) {
        NSString *description = A12S2Error(compileError);
        BOOL invalidProcedure = [description rangeOfString:@"InvalidProcedure"].location != NSNotFound;
        [lines addObject:[NSString stringWithFormat:
            @"%@ compile=FAIL ms=%.2f procedures=%lu netBytes=%lu weightBytes=%llu classification=%@ error=%@",
            label, ms, (unsigned long)indices.count, (unsigned long)netData.length,
            (unsigned long long)A12S2EWeightBytes(weights),
            invalidProcedure ? @"INVALID_PROCEDURE" : @"OTHER", description]];
        if ([localPath isKindOfClass:NSString.class]) {
            [NSFileManager.defaultManager removeItemAtPath:localPath error:nil];
        }
        return invalidProcedure ? A12S2IInvalidProcedure : A12S2IOtherFailure;
    }

    [lines addObject:[NSString stringWithFormat:
        @"%@ compile=PASS ms=%.2f procedures=%lu cacheHit=%@ netBytes=%lu weightBytes=%llu",
        label, ms, (unsigned long)indices.count, cacheHit ? @"yes" : @"no",
        (unsigned long)netData.length, (unsigned long long)A12S2EWeightBytes(weights)]];

    if (compiledPlistOut) *compiledPlistOut = trial;
    if (compiledWeightsOut) *compiledWeightsOut = weights;
    return A12S2IPass;
}

static inline BOOL A12S2ILoadExpectedCount(
    NSDictionary *plist,
    NSDictionary *weights,
    NSUInteger expected,
    NSString *label,
    NSMutableArray<NSString *> *lines) {

    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) return NO;

    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weights, nil) : nil;
    id memoryModel = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) return NO;

    NSError *loadError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"), 25u, @{}, &loadError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:@"%@ load=FAIL ms=%.2f error=%@",
            label, ms, A12S2Error(loadError)]];
        return NO;
    }

    id loweredModel = [memoryModel respondsToSelector:NSSelectorFromString(@"model")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"model")) : nil;
    NSDictionary *desc = A12S2ANEFDescription(loweredModel ?: memoryModel);
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSUInteger count = MAX(procedures.count, nameMap.count);
    [lines addObject:[NSString stringWithFormat:
        @"%@ load=PASS ms=%.2f loadCount=1 procedureCount=%lu expected=%lu names=%@ procedures=%@",
        label, ms, (unsigned long)count, (unsigned long)expected, nameMap, procedures]];

    NSError *unloadError = nil;
    BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
    [lines addObject:[NSString stringWithFormat:@"%@ unload=%@ error=%@",
        label, unloaded ? @"PASS" : @"FAIL", A12S2Error(unloadError)]];
    return unloaded && count == expected;
}

static inline NSString *A12ANEStage2IProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2I",
            @"goal=isolate InvalidProcedure: count vs heterogeneous geometry vs QKV multi-output",
            @"matrix=homogeneous3, context2, mixed-context, mixed-up, mixed-down, qkv-single, qkv-pair; then prefixes if all pass",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2I-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *canonicalAll = A12S2HBuildCanonicalCombined(donors, root, runtimeVersion, lines);
        if (!canonicalAll) {
            [lines addObject:@"RESULT=FAIL stage=canonical-combine"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSArray<NSDictionary *> *propertyTrials = @[
            @{@"label": @"trial-homogeneous3-selfo-crossq-crosso", @"indices": @[@1,@2,@5]},
            @{@"label": @"trial-context2-crossk-crossv", @"indices": @[@3,@4]},
            @{@"label": @"trial-mixed-context-selfo-crossk", @"indices": @[@1,@3]},
            @{@"label": @"trial-mixed-up-selfo-mlpup", @"indices": @[@1,@6]},
            @{@"label": @"trial-mixed-down-selfo-mlpdown", @"indices": @[@1,@7]},
            @{@"label": @"trial-qkv-single", @"indices": @[@0]},
            @{@"label": @"trial-qkv-pair-selfqkv-selfo", @"indices": @[@0,@1]},
        ];

        BOOL allPropertyPass = YES;
        NSMutableDictionary *homogeneous3Plist = nil;
        NSMutableDictionary *homogeneous3Weights = nil;
        for (NSDictionary *spec in propertyTrials) {
            NSString *label = spec[@"label"];
            NSArray<NSNumber *> *indices = spec[@"indices"];
            NSMutableDictionary *compiledPlist = nil;
            NSMutableDictionary *compiledWeights = nil;
            A12S2IResult result = A12S2IRunCompileTrial(
                canonicalAll, donors, indices, root, label, lines,
                &compiledPlist, &compiledWeights);
            [lines addObject:[NSString stringWithFormat:@"%@ RESULT=%@",
                label,
                result == A12S2IPass ? @"PASS" :
                (result == A12S2IInvalidProcedure ? @"INVALID_PROCEDURE" : @"OTHER_FAILURE")]];
            if (result != A12S2IPass) allPropertyPass = NO;
            if ([label isEqualToString:@"trial-homogeneous3-selfo-crossq-crosso"] &&
                result == A12S2IPass) {
                homogeneous3Plist = compiledPlist;
                homogeneous3Weights = compiledWeights;
            }
        }

        if (homogeneous3Plist && homogeneous3Weights) {
            BOOL load3 = A12S2ILoadExpectedCount(
                homogeneous3Plist, homogeneous3Weights, 3,
                @"proof-homogeneous3", lines);
            [lines addObject:load3
                ? @"proof-homogeneous3=PASS oneLoad procedureCount=3"
                : @"proof-homogeneous3=FAIL"];
        }

        if (allPropertyPass) {
            [lines addObject:@"propertyMatrix=ALL_PASS proceeding=prefix-growth"];
            for (NSUInteger count = 3; count <= 8; ++count) {
                NSMutableArray<NSNumber *> *indices = [NSMutableArray arrayWithCapacity:count];
                for (NSUInteger i = 0; i < count; ++i) [indices addObject:@(i)];
                NSString *label = [NSString stringWithFormat:@"trial-prefix%lu", (unsigned long)count];
                NSMutableDictionary *compiledPlist = nil;
                NSMutableDictionary *compiledWeights = nil;
                A12S2IResult result = A12S2IRunCompileTrial(
                    canonicalAll, donors, indices, root, label, lines,
                    &compiledPlist, &compiledWeights);
                [lines addObject:[NSString stringWithFormat:@"%@ RESULT=%@",
                    label,
                    result == A12S2IPass ? @"PASS" :
                    (result == A12S2IInvalidProcedure ? @"INVALID_PROCEDURE" : @"OTHER_FAILURE")]];
                if (result != A12S2IPass) {
                    [lines addObject:[NSString stringWithFormat:
                        @"prefixBoundary=firstFail count=%lu classification=%@",
                        (unsigned long)count,
                        result == A12S2IInvalidProcedure ? @"INVALID_PROCEDURE" : @"OTHER_FAILURE"]];
                    break;
                }
                if (count == 8 && compiledPlist && compiledWeights) {
                    BOOL load8 = A12S2ILoadExpectedCount(
                        compiledPlist, compiledWeights, 8,
                        @"proof-all8", lines);
                    [lines addObject:load8
                        ? @"RESULT=PASS real-block0 loadCount=1 procedureCount=8"
                        : @"RESULT=FAIL stage=all8-load-count8"];
                    [NSFileManager.defaultManager removeItemAtURL:root error:nil];
                    return [lines componentsJoinedByString:@"\n"];
                }
            }
        }

        [lines addObject:@"RESULT=DIAGNOSTIC property-matrix-complete"];
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
