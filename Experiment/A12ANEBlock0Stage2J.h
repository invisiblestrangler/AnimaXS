#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import "A12ANEBlock0Stage2I.h"

// Stage 2J follows the decisive Stage-2I result: procedure count >2 and mixed
// I/O geometry both compile, while self-QKV is InvalidProcedure even by itself.
// The remaining illegal property is therefore the three-output procedure.
//
// Keep one private ANE model per DiT block, but split the already-lowered QKV
// network into three single-output network/procedure pairs (self_q/self_k/self_v)
// that share the exact same lowered W8 weights. The complete block container is
// therefore 10 procedures in ONE loaded model, preserving the 28-load target.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2JProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2J\nRESULT=SKIP simulator";
}
#else

static inline NSDictionary *A12S2JOutputEntryForUnit(
    NSDictionary *procedure, NSString *unitName, NSString *outputSymbol) {
    NSArray *outputs = [procedure[@"OutputList"] isKindOfClass:NSArray.class]
        ? procedure[@"OutputList"] : @[];
    for (id raw in outputs) {
        if (![raw isKindOfClass:NSDictionary.class]) continue;
        NSDictionary *entry = raw;
        NSString *name = [entry[@"Name"] isKindOfClass:NSString.class] ? entry[@"Name"] : nil;
        NSString *symbol = [entry[@"OutputName"] isKindOfClass:NSString.class] ? entry[@"OutputName"] : nil;
        if ([name isEqualToString:unitName] || [symbol isEqualToString:outputSymbol]) return entry;
    }
    return nil;
}

static inline NSMutableDictionary *A12S2JSplitQKV(
    NSDictionary *canonicalAll,
    NSMutableArray<NSString *> *lines) {
    NSArray *networks = [canonicalAll[@"Networks"] isKindOfClass:NSArray.class]
        ? canonicalAll[@"Networks"] : @[];
    NSArray *procedures = [canonicalAll[@"ProcedureList"] isKindOfClass:NSArray.class]
        ? canonicalAll[@"ProcedureList"] : @[];
    if (networks.count != 8 || procedures.count != 8 ||
        ![networks.firstObject isKindOfClass:NSString.class] ||
        ![procedures.firstObject isKindOfClass:NSDictionary.class]) {
        [lines addObject:[NSString stringWithFormat:
            @"splitQKV=FAIL stage=canonical-shape networks=%lu procedures=%lu",
            (unsigned long)networks.count, (unsigned long)procedures.count]];
        return nil;
    }

    NSString *qkvNetworkName = networks.firstObject;
    NSDictionary *qkvNetwork = [canonicalAll[qkvNetworkName] isKindOfClass:NSDictionary.class]
        ? canonicalAll[qkvNetworkName] : nil;
    NSDictionary *qkvProcedure = procedures.firstObject;
    NSArray *qkvWeights = [qkvNetwork[@"Weights"] isKindOfClass:NSArray.class]
        ? qkvNetwork[@"Weights"] : @[];
    NSArray *inputList = [qkvProcedure[@"InputList"] isKindOfClass:NSArray.class]
        ? qkvProcedure[@"InputList"] : @[];
    if (!qkvNetwork || qkvWeights.count == 0 || inputList.count != 1 ||
        ![inputList.firstObject isKindOfClass:NSDictionary.class]) {
        [lines addObject:@"splitQKV=FAIL stage=qkv-body"];
        return nil;
    }

    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"Version"] = canonicalAll[@"Version"] ?: @"1.0.10";
    NSMutableArray<NSString *> *outNetworks = [NSMutableArray arrayWithCapacity:10];
    NSMutableArray<NSDictionary *> *outProcedures = [NSMutableArray arrayWithCapacity:10];

    NSArray<NSDictionary *> *parts = @[
        @{@"unit": @"q", @"output": @"q@output", @"network": @"network_00_self_q", @"procedure": @"procedure_self_q"},
        @{@"unit": @"k", @"output": @"k@output", @"network": @"network_00_self_k", @"procedure": @"procedure_self_k"},
        @{@"unit": @"v", @"output": @"v@output", @"network": @"network_00_self_v", @"procedure": @"procedure_self_v"},
    ];

    for (NSDictionary *part in parts) {
        NSString *unitName = part[@"unit"];
        NSString *outputSymbol = part[@"output"];
        NSString *networkName = part[@"network"];
        NSString *procedureName = part[@"procedure"];
        NSDictionary *unit = [qkvNetwork[unitName] isKindOfClass:NSDictionary.class]
            ? qkvNetwork[unitName] : nil;
        NSDictionary *outputDict = [qkvNetwork[outputSymbol] isKindOfClass:NSDictionary.class]
            ? qkvNetwork[outputSymbol] : nil;
        NSDictionary *oldOutputEntry = A12S2JOutputEntryForUnit(qkvProcedure, unitName, outputSymbol);
        if (!unit || !outputDict || !oldOutputEntry) {
            [lines addObject:[NSString stringWithFormat:
                @"splitQKV=FAIL unit=%@ unitPresent=%@ outputPresent=%@ outputEntry=%@",
                unitName, unit ? @"yes" : @"no", outputDict ? @"yes" : @"no",
                oldOutputEntry ? @"yes" : @"no"]];
            return nil;
        }

        NSMutableDictionary *network = [NSMutableDictionary dictionary];
        network[@"Units"] = @[unitName];
        network[@"Weights"] = [qkvWeights copy];
        network[unitName] = [unit mutableCopy];
        network[outputSymbol] = [outputDict mutableCopy];

        NSMutableDictionary *inputEntry = [inputList.firstObject mutableCopy];
        inputEntry[@"Name"] = unitName;
        NSMutableDictionary *outputEntry = [oldOutputEntry mutableCopy];
        outputEntry[@"Name"] = unitName;
        outputEntry[@"OutputName"] = outputSymbol;
        NSDictionary *procedure = @{
            @"Name": procedureName,
            @"InputList": @[inputEntry],
            @"OperationList": @[@{@"OperationName": @"op0", @"NetworkName": networkName}],
            @"OutputList": @[outputEntry],
        };

        out[networkName] = network;
        [outNetworks addObject:networkName];
        [outProcedures addObject:procedure];
        [lines addObject:[NSString stringWithFormat:
            @"splitQKV[%@]=PASS network=%@ procedure=%@ unitKeys=%@ weights=%@ input=%@ output=%@",
            unitName, networkName, procedureName,
            [[network.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","],
            qkvWeights, inputEntry, outputEntry]];
    }

    for (NSUInteger i = 1; i < networks.count; ++i) {
        if (![networks[i] isKindOfClass:NSString.class] ||
            ![procedures[i] isKindOfClass:NSDictionary.class]) return nil;
        NSString *networkName = networks[i];
        NSDictionary *body = [canonicalAll[networkName] isKindOfClass:NSDictionary.class]
            ? canonicalAll[networkName] : nil;
        if (!body) return nil;
        out[networkName] = [body mutableCopy];
        [outNetworks addObject:networkName];
        [outProcedures addObject:procedures[i]];
    }

    out[@"Networks"] = outNetworks;
    out[@"ProcedureList"] = outProcedures;
    [lines addObject:[NSString stringWithFormat:
        @"splitQKV=PASS version=%@ networks=%lu procedures=%lu names=%@",
        out[@"Version"], (unsigned long)outNetworks.count,
        (unsigned long)outProcedures.count, [outProcedures valueForKey:@"Name"]]];
    return out;
}

static inline NSArray<NSDictionary *> *A12S2JDonorMap(NSArray<NSDictionary *> *donors) {
    if (donors.count != 8) return nil;
    return @[donors[0], donors[0], donors[0],
             donors[1], donors[2], donors[3], donors[4],
             donors[5], donors[6], donors[7]];
}

// Same semantic normalization as A12S2NormalizeWeights, but deduplicates the
// three split Q/K/V references so the proof does not triple the QKV payload.
static inline BOOL A12S2JNormalizeWeightsDedup(
    NSMutableDictionary *plist,
    NSArray<NSDictionary *> *donorMap,
    NSURL *root,
    NSMutableDictionary *weightMap,
    NSMutableArray<NSString *> *lines) {
    NSArray *networks = [plist[@"Networks"] isKindOfClass:NSArray.class]
        ? plist[@"Networks"] : @[];
    if (networks.count != donorMap.count) return NO;

    NSMutableDictionary<NSString *, NSString *> *nameBySource = [NSMutableDictionary dictionary];
    uint64_t totalBytes = 0;
    for (NSUInteger n = 0; n < networks.count; ++n) {
        NSString *networkName = [networks[n] isKindOfClass:NSString.class] ? networks[n] : nil;
        NSMutableDictionary *network = networkName && [plist[networkName] isKindOfClass:NSDictionary.class]
            ? [plist[networkName] mutableCopy] : nil;
        if (!network) return NO;
        plist[networkName] = network;
        NSArray *refs = [network[@"Weights"] isKindOfClass:NSArray.class] ? network[@"Weights"] : @[];
        NSMutableArray<NSString *> *normalized = [NSMutableArray arrayWithCapacity:refs.count];
        NSURL *donorURL = donorMap[n][@"url"];
        for (NSUInteger w = 0; w < refs.count; ++w) {
            if (![refs[w] isKindOfClass:NSString.class]) return NO;
            NSString *reference = refs[w];
            NSURL *source = A12S2ResolveWeight(reference, root, donorURL);
            if (!source) {
                [lines addObject:[NSString stringWithFormat:
                    @"weightDedup=FAIL network=%@ ref=%@", networkName, reference]];
                return NO;
            }
            NSString *sourceKey = source.path.stringByStandardizingPath;
            NSString *newName = nameBySource[sourceKey];
            if (!newName) {
                NSError *readError = nil;
                NSData *data = [NSData dataWithContentsOfURL:source
                    options:NSDataReadingMappedIfSafe error:&readError];
                if (!data) {
                    [lines addObject:[NSString stringWithFormat:
                        @"weightDedup=FAIL source=%@ error=%@", source.path, A12S2Error(readError)]];
                    return NO;
                }
                newName = [NSString stringWithFormat:@"s2j_w%02lu_%@",
                    (unsigned long)nameBySource.count,
                    source.lastPathComponent.length ? source.lastPathComponent : @"weights.bin"];
                nameBySource[sourceKey] = newName;
                weightMap[newName] = @{@"offset": @0, @"data": data};
                totalBytes += data.length;
                [lines addObject:[NSString stringWithFormat:
                    @"weightDedup NEW source=%@ -> %@ bytes=%lu",
                    source.path, newName, (unsigned long)data.length]];
            } else {
                [lines addObject:[NSString stringWithFormat:
                    @"weightDedup REUSE source=%@ -> %@", source.path, newName]];
            }
            [normalized addObject:newName];
        }
        network[@"Weights"] = normalized;
    }
    [lines addObject:[NSString stringWithFormat:
        @"weightsDedup=PASS uniqueFiles=%lu totalBytes=%llu",
        (unsigned long)weightMap.count, (unsigned long long)totalBytes]];
    return YES;
}

static inline NSString *A12ANEStage2JProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2J",
            @"goal=replace illegal 3-output QKV procedure with q/k/v single-output procedures",
            @"architecture=10 procedures in ONE private ANE model per block; target remains 28 loaded models",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2J-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *canonicalAll = A12S2HBuildCanonicalCombined(
            donors, root, runtimeVersion, lines);
        if (!canonicalAll) {
            [lines addObject:@"RESULT=FAIL stage=canonical-combine"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *full10 = A12S2JSplitQKV(canonicalAll, lines);
        NSArray<NSDictionary *> *donorMap = A12S2JDonorMap(donors);
        if (!full10 || donorMap.count != 10) {
            [lines addObject:@"RESULT=FAIL stage=split-qkv"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        // Snapshot before any weight normalization mutates network Weights.
        NSMutableDictionary *split3 = A12S2HSubset(
            full10, @[@0,@1,@2], lines, @"trial-A-split-qkv3");
        NSMutableDictionary *all10 = A12S2HSubset(
            full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9], lines, @"trial-B-full10");
        if (!split3 || !all10) {
            [lines addObject:@"RESULT=FAIL stage=snapshot"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *splitWeights = [NSMutableDictionary dictionary];
        if (!A12S2JNormalizeWeightsDedup(
            split3, @[donors[0], donors[0], donors[0]], root, splitWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=split3-weights"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult splitCompile = A12S2EPreparedCompileOnly(
            split3, splitWeights, @"trial-A-split-qkv3", lines);
        if (splitCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=split-qkv3 result=%ld", (long)splitCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        BOOL splitLoad = A12S2ILoadExpectedCount(
            split3, splitWeights, 3, @"proof-split-qkv3", lines);
        if (!splitLoad) {
            [lines addObject:@"RESULT=FAIL stage=split-qkv3-load"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"splitQKVProof=PASS oneLoad procedureCount=3"];

        NSMutableDictionary *allWeights = [NSMutableDictionary dictionary];
        if (!A12S2JNormalizeWeightsDedup(all10, donorMap, root, allWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=all10-weights"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult allCompile = A12S2EPreparedCompileOnly(
            all10, allWeights, @"trial-B-full10", lines);
        if (allCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=full10 result=%ld", (long)allCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        BOOL allLoad = A12S2ILoadExpectedCount(
            all10, allWeights, 10, @"proof-full10", lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (!allLoad) {
            [lines addObject:@"RESULT=FAIL stage=full10-load"];
            return [lines componentsJoinedByString:@"\n"];
        }

        [lines addObject:@"RESULT=PASS real-block0-one-model tenProcedures qkvSplit=3 loadedModelsPerBlock=1 targetLoadedModels=28"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
