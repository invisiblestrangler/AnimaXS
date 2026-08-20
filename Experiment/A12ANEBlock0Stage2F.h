#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <dlfcn.h>

// Stage 2F: Stage 2E proved the first failing delta is attaching ProcedureList
// to an otherwise recompilable real W8 ANECIR network. The remaining concrete
// structural mismatch versus both the proven tiny V2 multi-procedure netplist
// and Apple's _MLCANEPlistBuilder output is that espresso_dump_ir emits legacy
// network-level `Inputs` / `Outputs` arrays. Procedure-authored netplists carry
// the same live I/O through ProcedureList + per-symbol dictionaries instead.
//
// This probe removes ONLY those two legacy arrays after Stage 2E has harvested
// their symbols and constructed the procedure entry. Everything else in the
// lowered real W8 network body, including x/output stride dictionaries and the
// exact Conv/GOC units + weights, remains unchanged.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_Stage2E
#import "A12ANEBlock0Stage2E.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2F\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12S2FStripLegacyIOArrays(NSMutableDictionary *container,
                                             NSMutableArray<NSString *> *lines,
                                             NSString *label) {
    NSArray *networks = [container[@"Networks"] isKindOfClass:NSArray.class]
        ? container[@"Networks"] : @[];
    if (networks.count != 1 || ![networks.firstObject isKindOfClass:NSString.class]) {
        [lines addObject:[NSString stringWithFormat:@"%@ stripLegacyIO=FAIL networks=%@", label, networks]];
        return NO;
    }
    NSString *networkName = networks.firstObject;
    NSMutableDictionary *network = [container[networkName] isKindOfClass:NSDictionary.class]
        ? [container[networkName] mutableCopy] : nil;
    if (!network) {
        [lines addObject:[NSString stringWithFormat:@"%@ stripLegacyIO=FAIL missingNetwork=%@", label, networkName]];
        return NO;
    }

    id oldInputs = network[@"Inputs"];
    id oldOutputs = network[@"Outputs"];
    [network removeObjectForKey:@"Inputs"];
    [network removeObjectForKey:@"Outputs"];
    container[networkName] = network;

    [lines addObject:[NSString stringWithFormat:
        @"%@ stripLegacyIO=PASS network=%@ removedInputs=%@ removedOutputs=%@ remainingKeys=%@",
        label, networkName, A12S2Desc(oldInputs), A12S2Desc(oldOutputs),
        [[network.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","]]];
    return YES;
}

static inline BOOL A12S2FLoadPair(NSDictionary *plist,
                                  NSDictionary *weightMap,
                                  NSMutableArray<NSString *> *lines) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"pairLoad serialize=FAIL error=%@", A12S2Error(serializeError)]];
        return NO;
    }

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !memoryModelClass) {
        [lines addObject:@"pairLoad=FAIL stage=class-discovery"];
        return NO;
    }

    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:@"pairLoad=FAIL stage=descriptor-model"];
        return NO;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    if (!cacheHit) {
        NSError *compileError = nil;
        NSTimeInterval compileStart = NSDate.timeIntervalSinceReferenceDate;
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
            25u, @{}, &compileError);
        double compileMS = (NSDate.timeIntervalSinceReferenceDate - compileStart) * 1000.0;
        [lines addObject:[NSString stringWithFormat:@"pairLoad compile=%@ ms=%.2f error=%@",
            compiled ? @"PASS" : @"FAIL", compileMS, A12S2Error(compileError)]];
        if (!compiled) return NO;
    } else {
        [lines addObject:@"pairLoad compile=CACHE_HIT"];
    }

    NSError *loadError = nil;
    NSTimeInterval loadStart = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStart) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:@"pairLoad load=FAIL ms=%.2f error=%@",
            loadMS, A12S2Error(loadError)]];
        return NO;
    }

    id loweredModel = [memoryModel respondsToSelector:NSSelectorFromString(@"model")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"model")) : nil;
    NSDictionary *desc = A12S2ANEFDescription(loweredModel ?: memoryModel);
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSUInteger procedureCount = MAX(procedures.count, nameMap.count);
    [lines addObject:[NSString stringWithFormat:
        @"pairLoad load=PASS ms=%.2f loadCount=1 procedureCount=%lu names=%@ procedures=%@",
        loadMS, (unsigned long)procedureCount, nameMap, procedures]];

    NSError *unloadError = nil;
    BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
    [lines addObject:[NSString stringWithFormat:@"pairLoad unload=%@ unloadCount=%d error=%@",
        unloaded ? @"PASS" : @"FAIL", unloaded ? 1 : 0, A12S2Error(unloadError)]];
    return procedureCount == 2 && unloaded;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2F",
            @"goal=test legacy Inputs/Outputs arrays as the procedure-schema conflict",
            @"matrix=strip legacy IO arrays -> single self_o compile -> self_o+cross_q compile -> one load/two procedures",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *selfO = donors[1];
        NSDictionary *crossQ = donors[2];
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2F-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
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
            [lines addObject:@"RESULT=FAIL stage=procedure-container-build"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        if (!A12S2FStripLegacyIOArrays(selfContainer, lines, @"self_o") ||
            !A12S2FStripLegacyIOArrays(crossContainer, lines, @"cross_q")) {
            [lines addObject:@"RESULT=FAIL stage=strip-legacy-io"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *selfWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(selfContainer, @[selfO], selfDir, selfWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=self-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult single = A12S2ECompileOnly(
            selfContainer, selfWeights, @"trial-A-single-stripped", lines);
        if (single != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=stripped-single result=%ld", (long)single]];
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
        A12S2ECompileResult pairCompile = A12S2ECompileOnly(
            pair, pairWeights, @"trial-B-pair-stripped", lines);
        if (pairCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=stripped-pair result=%ld", (long)pairCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        [lines addObject:@"pairCompile=PASS proceeding=one-load-two-procedures"];
        BOOL loadPass = A12S2FLoadPair(pair, pairWeights, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        [lines addObject:loadPass
            ? @"RESULT=PASS real-W8 stripped pair loadCount=1 procedureCount=2 unloadCount=1"
            : @"RESULT=FAIL stage=stripped-pair-load"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
