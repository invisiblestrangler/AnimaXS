#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <dlfcn.h>
#import "A12ANEBlock0Stage2E.h"

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2EPreparedProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2E prepared\nRESULT=SKIP simulator";
}
#else

static inline A12S2ECompileResult A12S2EPreparedCompileOnly(
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
        return A12S2ECompileFail;
    }

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !memoryModelClass) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=class-discovery", label]];
        return A12S2ECompileFail;
    }

    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=descriptor-model", label]];
        return A12S2ECompileFail;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    NSError *ioError = nil;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL stage=localModelPath", label]];
            return A12S2ECompileFail;
        }
        NSFileManager *fm = NSFileManager.defaultManager;
        if (![fm createDirectoryAtPath:localPath withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![netData writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL error=%@",
                label, A12S2Error(ioError)]];
            return A12S2ECompileFail;
        }
        for (NSString *name in weightMap) {
            NSDictionary *entry = [weightMap[name] isKindOfClass:NSDictionary.class]
                ? weightMap[name] : nil;
            NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
            if (!data || ![data writeToFile:[localPath stringByAppendingPathComponent:name]
                                     options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"%@ materialize=FAIL weight=%@ error=%@",
                    label, name, A12S2Error(ioError)]];
                return A12S2ECompileFail;
            }
        }
    }
    [lines addObject:[NSString stringWithFormat:@"%@ materialize=PASS cacheHit=%@ localPath=%@",
        label, cacheHit ? @"yes" : @"no", localPath ?: @"(nil)"]];

    NSError *compileError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL compiled = cacheHit || ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
        25u, @{}, &compileError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (!compiled) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ compile=FAIL ms=%.2f netBytes=%lu weightBytes=%llu errorDomain=%@ errorCode=%ld error=%@",
            label, ms, (unsigned long)netData.length,
            (unsigned long long)A12S2EWeightBytes(weightMap),
            compileError.domain ?: @"(nil)", (long)compileError.code,
            A12S2Error(compileError)]];
        if ([compileError.domain isEqualToString:NSCocoaErrorDomain] && compileError.code == 4097) {
            [lines addObject:[NSString stringWithFormat:@"%@ classification=HELPER_LOST", label]];
            return A12S2EHelperLost;
        }
        return A12S2ECompileFail;
    }
    [lines addObject:[NSString stringWithFormat:
        @"%@ compile=PASS ms=%.2f netBytes=%lu weightBytes=%llu",
        label, ms, (unsigned long)netData.length,
        (unsigned long long)A12S2EWeightBytes(weightMap)]];
    return A12S2ECompilePass;
}

static inline NSString *A12ANEStage2EPreparedProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2E prepared",
            @"goal=Apple-parity procedure schema with Stage2D-identical materialization",
            @"matrix=single self_o -> self_o+cross_q -> one load/two procedures",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *selfO = donors[1];
        NSDictionary *crossQ = donors[2];
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2EP-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSURL *selfDir = [root URLByAppendingPathComponent:@"self_o" isDirectory:YES];
        NSURL *crossDir = [root URLByAppendingPathComponent:@"cross_q" isDirectory:YES];

        NSMutableDictionary *rawSelf = nil;
        NSMutableDictionary *rawCross = nil;
        if (!A12S2Lower(@[selfO], selfDir, @"lower-self_o", lines, &rawSelf) ||
            !A12S2Lower(@[crossQ], crossDir, @"lower-cross_q", lines, &rawCross)) {
            [lines addObject:@"RESULT=FAIL stage=lower-donors"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *selfContainer = A12S2EProcedureContainer(
            rawSelf, selfO, selfDir, @"network_self_o", lines);
        NSMutableDictionary *crossContainer = A12S2EProcedureContainer(
            rawCross, crossQ, crossDir, @"network_cross_q", lines);
        if (!selfContainer || !crossContainer) {
            [lines addObject:@"RESULT=FAIL stage=apple-parity-wrapper-build"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *selfWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(selfContainer, @[selfO], selfDir, selfWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=self_o-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult single = A12S2EPreparedCompileOnly(
            selfContainer, selfWeights, @"trial-A-single-apple-parity", lines);
        if (single != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC stage=single-apple-parity result=%ld", (long)single]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *pair = A12S2EMergePair(selfContainer, crossContainer, lines);
        if (!pair) {
            [lines addObject:@"RESULT=FAIL stage=pair-build"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSMutableDictionary *pairWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(pair, @[selfO, crossQ], root, pairWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=pair-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult pairCompile = A12S2EPreparedCompileOnly(
            pair, pairWeights, @"trial-B-pair-apple-parity", lines);
        if (pairCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC stage=pair-apple-parity result=%ld", (long)pairCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        [lines addObject:@"pairCompile=PASS proceeding=compile-load-procedure-count"];
        BOOL loadPass = A12S2CompileLoad(pair, pairWeights, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (loadPass) {
            [lines addObject:@"RESULT=PASS real-W8 Apple-parity pair compiled and loaded as multi-procedure model"];
        } else if (![lines.lastObject hasPrefix:@"RESULT="]) {
            [lines addObject:@"RESULT=FAIL stage=pair-load"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
