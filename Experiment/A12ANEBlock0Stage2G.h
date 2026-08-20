#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <dlfcn.h>
#import "A12ANEBlock0Stage2EPrepared.h"

// Stage 2G: Stage 2F proved that removing only the legacy network-level
// `Inputs`/`Outputs` arrays is insufficient. Apple's own _MLCANEPlistBuilder
// shows the remaining concrete schema mismatch:
//   * live inputs exist only in ProcedureList.InputList; it does NOT install a
//     network dictionary for the input symbol;
//   * live outputs are installed into the network as a fresh dictionary with
//     exactly OutputName, Bottom, OutputType, OutputInterleave.
//
// The Espresso dump has the opposite extras: an `x` input dictionary with
// stride fields, and a `y@output` dictionary with stride fields but no
// OutputName. This probe canonicalizes those two pieces exactly to Apple's
// procedure-builder shape while leaving the real W8 unit graph + weights
// untouched. It also reports/uses kMLCCurrentANEIRVersion when exported by the
// device MLCompute framework instead of assuming a procedure IR version.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2GProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2G\nRESULT=SKIP simulator";
}
#else

static inline NSString *A12S2GCurrentANEIRVersion(NSMutableArray<NSString *> *lines) {
    void *handle = dlopen("/System/Library/Frameworks/MLCompute.framework/MLCompute",
                          RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        [lines addObject:@"runtimeIRVersion=ABSENT framework"];
        return nil;
    }
    void *ptr = dlsym(handle, "kMLCCurrentANEIRVersion");
    if (!ptr) {
        [lines addObject:@"runtimeIRVersion=ABSENT symbol"];
        return nil;
    }
    id value = nil;
    @try { value = *(__unsafe_unretained id *)ptr; }
    @catch (__unused NSException *e) { value = nil; }
    NSString *version = [value isKindOfClass:NSString.class] ? value : nil;
    [lines addObject:[NSString stringWithFormat:@"runtimeIRVersion=%@ class=%@",
        version ?: A12S2Desc(value), value ? NSStringFromClass([value class]) : @"(nil)"]];
    return version;
}

static inline BOOL A12S2GCanonicalizeProcedureNetwork(NSMutableDictionary *container,
                                                       NSMutableArray<NSString *> *lines,
                                                       NSString *label) {
    NSArray *networks = [container[@"Networks"] isKindOfClass:NSArray.class]
        ? container[@"Networks"] : @[];
    NSArray *procedures = [container[@"ProcedureList"] isKindOfClass:NSArray.class]
        ? container[@"ProcedureList"] : @[];
    if (networks.count != 1 || procedures.count != 1 ||
        ![networks.firstObject isKindOfClass:NSString.class] ||
        ![procedures.firstObject isKindOfClass:NSDictionary.class]) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ canonical=FAIL networks=%@ procedures=%@", label, networks, procedures]];
        return NO;
    }

    NSString *networkName = networks.firstObject;
    NSDictionary *procedure = procedures.firstObject;
    NSMutableDictionary *network = [container[networkName] isKindOfClass:NSDictionary.class]
        ? [container[networkName] mutableCopy] : nil;
    if (!network) {
        [lines addObject:[NSString stringWithFormat:@"%@ canonical=FAIL missingNetwork=%@", label, networkName]];
        return NO;
    }

    NSArray *inputList = [procedure[@"InputList"] isKindOfClass:NSArray.class]
        ? procedure[@"InputList"] : @[];
    NSArray *outputList = [procedure[@"OutputList"] isKindOfClass:NSArray.class]
        ? procedure[@"OutputList"] : @[];

    id oldInputs = network[@"Inputs"];
    id oldOutputs = network[@"Outputs"];
    [network removeObjectForKey:@"Inputs"];
    [network removeObjectForKey:@"Outputs"];

    NSMutableArray<NSString *> *removedInputKeys = [NSMutableArray array];
    for (id rawEntry in inputList) {
        if (![rawEntry isKindOfClass:NSDictionary.class]) return NO;
        NSString *symbol = [rawEntry[@"InputName"] isKindOfClass:NSString.class]
            ? rawEntry[@"InputName"] : nil;
        if (!symbol) return NO;
        id old = network[symbol];
        if (old) {
            [removedInputKeys addObject:symbol];
            [network removeObjectForKey:symbol];
        }
    }

    NSMutableArray<NSString *> *rewrittenOutputs = [NSMutableArray array];
    for (id rawEntry in outputList) {
        if (![rawEntry isKindOfClass:NSDictionary.class]) return NO;
        NSDictionary *entry = rawEntry;
        NSString *symbol = [entry[@"OutputName"] isKindOfClass:NSString.class]
            ? entry[@"OutputName"] : nil;
        NSString *bottom = [entry[@"Name"] isKindOfClass:NSString.class]
            ? entry[@"Name"] : nil;
        if (!symbol || !bottom) return NO;
        NSDictionary *old = [network[symbol] isKindOfClass:NSDictionary.class]
            ? network[symbol] : @{};
        id outputType = entry[@"OutputType"] ?: old[@"OutputType"] ?: @"Float16";
        id interleave = entry[@"OutputInterleave"] ?: old[@"OutputInterleave"] ?: @1;
        network[symbol] = @{
            @"OutputName": symbol,
            @"Bottom": bottom,
            @"OutputType": outputType,
            @"OutputInterleave": interleave,
        };
        [rewrittenOutputs addObject:symbol];
    }

    container[networkName] = network;
    [lines addObject:[NSString stringWithFormat:
        @"%@ canonical=PASS network=%@ removedLegacyInputs=%@ removedLegacyOutputs=%@ removedInputDicts=%@ rewrittenOutputs=%@ remainingKeys=%@",
        label, networkName, A12S2Desc(oldInputs), A12S2Desc(oldOutputs),
        removedInputKeys, rewrittenOutputs,
        [[network.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","]]];
    for (NSString *symbol in rewrittenOutputs) {
        [lines addObject:[NSString stringWithFormat:@"%@ outputDict[%@]=%@",
            label, symbol, A12S2Desc(network[symbol])]];
    }
    return YES;
}

static inline BOOL A12S2GLoadPair(NSDictionary *plist,
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
    NSArray *anefProcedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSUInteger procedureCount = MAX(anefProcedures.count, nameMap.count);
    [lines addObject:[NSString stringWithFormat:
        @"pairLoad load=PASS ms=%.2f loadCount=1 procedureCount=%lu names=%@ procedures=%@",
        loadMS, (unsigned long)procedureCount, nameMap, anefProcedures]];

    NSError *unloadError = nil;
    BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
    [lines addObject:[NSString stringWithFormat:@"pairLoad unload=%@ unloadCount=%d error=%@",
        unloaded ? @"PASS" : @"FAIL", unloaded ? 1 : 0, A12S2Error(unloadError)]];
    return procedureCount == 2 && unloaded;
}

static inline NSString *A12ANEStage2GProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2G",
            @"goal=match Apple _MLCANEPlistBuilder live-I/O network dictionaries exactly",
            @"matrix=canonical single self_o -> canonical self_o+cross_q -> one load/two procedures",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *selfO = donors[1];
        NSDictionary *crossQ = donors[2];
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2G-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
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

        if (runtimeVersion.length) {
            selfContainer[@"Version"] = runtimeVersion;
            crossContainer[@"Version"] = runtimeVersion;
        }
        [lines addObject:[NSString stringWithFormat:@"selectedProcedureIRVersion=%@",
            selfContainer[@"Version"] ?: @"(nil)"]];

        if (!A12S2GCanonicalizeProcedureNetwork(selfContainer, lines, @"self_o") ||
            !A12S2GCanonicalizeProcedureNetwork(crossContainer, lines, @"cross_q")) {
            [lines addObject:@"RESULT=FAIL stage=canonicalize-procedure-network"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *selfWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(selfContainer, @[selfO], selfDir, selfWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=self-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult single = A12S2EPreparedCompileOnly(
            selfContainer, selfWeights, @"trial-A-single-canonical", lines);
        if (single != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=canonical-single result=%ld", (long)single]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *pair = A12S2EMergePair(selfContainer, crossContainer, lines);
        if (!pair) {
            [lines addObject:@"RESULT=FAIL stage=pair-build"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        if (runtimeVersion.length) pair[@"Version"] = runtimeVersion;
        else if (selfContainer[@"Version"]) pair[@"Version"] = selfContainer[@"Version"];
        [lines addObject:[NSString stringWithFormat:@"pairProcedureIRVersion=%@", pair[@"Version"] ?: @"(nil)"]];

        NSMutableDictionary *pairWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(pair, @[selfO, crossQ], root, pairWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=pair-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult pairCompile = A12S2EPreparedCompileOnly(
            pair, pairWeights, @"trial-B-pair-canonical", lines);
        if (pairCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=canonical-pair result=%ld", (long)pairCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        [lines addObject:@"pairCompile=PASS proceeding=one-load-two-procedures"];
        BOOL loadPass = A12S2GLoadPair(pair, pairWeights, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        [lines addObject:loadPass
            ? @"RESULT=PASS real-W8 canonical pair loadCount=1 procedureCount=2 unloadCount=1"
            : @"RESULT=FAIL stage=canonical-pair-load"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
