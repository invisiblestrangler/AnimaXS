#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import "A12ANEBlock0Stage2G.h"

// Stage 2H follows the Stage-2G breakthrough: a canonical real W8 self_o
// procedure compiles successfully. Stage 2G then hit only a probe bookkeeping
// bug because single-procedure weight normalization mutated the container before
// that same object was reused for the pair.
//
// This probe avoids that mutation entirely. It lowers/canonicalizes all eight
// block-0 donors once, snapshots canonical containers before any normalization,
// then tests the real self_o+cross_q pair and, if that succeeds, all eight
// procedures in one loaded private ANE model.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2HProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2H\nRESULT=SKIP simulator";
}
#else

static inline NSMutableDictionary *A12S2HBuildCanonicalCombined(
    NSArray<NSDictionary *> *donors,
    NSURL *root,
    NSString *version,
    NSMutableArray<NSString *> *lines) {
    NSMutableDictionary *combined = [NSMutableDictionary dictionary];
    combined[@"Version"] = version.length ? version : @"1.0.10";
    NSMutableArray<NSString *> *networkNames = [NSMutableArray arrayWithCapacity:donors.count];
    NSMutableArray<NSDictionary *> *procedures = [NSMutableArray arrayWithCapacity:donors.count];

    for (NSUInteger i = 0; i < donors.count; ++i) {
        NSDictionary *donor = donors[i];
        NSString *donorName = donor[@"name"];
        NSURL *dumpDir = [root URLByAppendingPathComponent:
            [NSString stringWithFormat:@"donor-%02lu-%@", (unsigned long)i, donorName]
            isDirectory:YES];
        NSString *label = [NSString stringWithFormat:@"lower[%lu]-%@",
            (unsigned long)i, donorName];
        NSMutableDictionary *raw = nil;
        if (!A12S2Lower(@[donor], dumpDir, label, lines, &raw)) {
            [lines addObject:[NSString stringWithFormat:@"canonicalCombine=FAIL lower=%@", donorName]];
            return nil;
        }

        NSString *networkName = [NSString stringWithFormat:@"network_%02lu_%@",
            (unsigned long)i, donorName];
        NSMutableDictionary *container = A12S2EProcedureContainer(
            raw, donor, dumpDir, networkName, lines);
        if (!container) {
            [lines addObject:[NSString stringWithFormat:@"canonicalCombine=FAIL wrapper=%@", donorName]];
            return nil;
        }
        container[@"Version"] = combined[@"Version"];
        if (!A12S2GCanonicalizeProcedureNetwork(container, lines, donorName)) {
            [lines addObject:[NSString stringWithFormat:@"canonicalCombine=FAIL canonicalize=%@", donorName]];
            return nil;
        }

        NSArray *containerNetworks = [container[@"Networks"] isKindOfClass:NSArray.class]
            ? container[@"Networks"] : @[];
        NSArray *containerProcedures = [container[@"ProcedureList"] isKindOfClass:NSArray.class]
            ? container[@"ProcedureList"] : @[];
        if (containerNetworks.count != 1 || containerProcedures.count != 1) {
            [lines addObject:[NSString stringWithFormat:
                @"canonicalCombine=FAIL donor=%@ networks=%lu procedures=%lu",
                donorName, (unsigned long)containerNetworks.count,
                (unsigned long)containerProcedures.count]];
            return nil;
        }
        NSString *emittedNetworkName = containerNetworks.firstObject;
        NSDictionary *network = [container[emittedNetworkName] isKindOfClass:NSDictionary.class]
            ? container[emittedNetworkName] : nil;
        NSDictionary *procedure = [containerProcedures.firstObject isKindOfClass:NSDictionary.class]
            ? containerProcedures.firstObject : nil;
        if (!network || !procedure) return nil;

        combined[emittedNetworkName] = [network mutableCopy];
        [networkNames addObject:emittedNetworkName];
        [procedures addObject:procedure];
        [lines addObject:[NSString stringWithFormat:
            @"canonicalCombine[%lu]=PASS donor=%@ network=%@ procedure=%@ keys=%@",
            (unsigned long)i, donorName, emittedNetworkName,
            procedure[@"Name"] ?: @"?",
            [[network.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","]]];
    }

    combined[@"Networks"] = networkNames;
    combined[@"ProcedureList"] = procedures;
    [lines addObject:[NSString stringWithFormat:
        @"canonicalCombine=PASS version=%@ networks=%lu procedures=%lu",
        combined[@"Version"], (unsigned long)networkNames.count,
        (unsigned long)procedures.count]];
    return combined;
}

static inline NSMutableDictionary *A12S2HSubset(
    NSDictionary *full,
    NSArray<NSNumber *> *indices,
    NSMutableArray<NSString *> *lines,
    NSString *label) {
    NSArray *networks = [full[@"Networks"] isKindOfClass:NSArray.class]
        ? full[@"Networks"] : @[];
    NSArray *procedures = [full[@"ProcedureList"] isKindOfClass:NSArray.class]
        ? full[@"ProcedureList"] : @[];
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"Version"] = full[@"Version"] ?: @"1.0.10";
    NSMutableArray<NSString *> *outNetworks = [NSMutableArray arrayWithCapacity:indices.count];
    NSMutableArray<NSDictionary *> *outProcedures = [NSMutableArray arrayWithCapacity:indices.count];

    for (NSNumber *rawIndex in indices) {
        NSUInteger i = rawIndex.unsignedIntegerValue;
        if (i >= networks.count || i >= procedures.count ||
            ![networks[i] isKindOfClass:NSString.class] ||
            ![procedures[i] isKindOfClass:NSDictionary.class]) {
            [lines addObject:[NSString stringWithFormat:@"%@ subset=FAIL index=%lu",
                label, (unsigned long)i]];
            return nil;
        }
        NSString *networkName = networks[i];
        NSDictionary *body = [full[networkName] isKindOfClass:NSDictionary.class]
            ? full[networkName] : nil;
        if (!body) return nil;
        out[networkName] = [body mutableCopy];
        [outNetworks addObject:networkName];
        [outProcedures addObject:procedures[i]];
    }
    out[@"Networks"] = outNetworks;
    out[@"ProcedureList"] = outProcedures;
    [lines addObject:[NSString stringWithFormat:
        @"%@ subset=PASS version=%@ networks=%@ procedures=%@",
        label, out[@"Version"], outNetworks,
        [outProcedures valueForKey:@"Name"]]];
    return out;
}

static inline NSString *A12ANEStage2HProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2H",
            @"goal=canonical real W8 pair -> all eight procedures in one private ANE model",
            @"matrix=pair compile+load/count2 -> all8 compile+load/count8; no redundant single compile",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2H-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *canonicalAll = A12S2HBuildCanonicalCombined(
            donors, root, runtimeVersion, lines);
        if (!canonicalAll) {
            [lines addObject:@"RESULT=FAIL stage=canonical-combine"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        // Snapshot BOTH test containers before any call to NormalizeWeights,
        // because normalization intentionally rewrites each network's Weights.
        NSMutableDictionary *pair = A12S2HSubset(
            canonicalAll, @[@1, @2], lines, @"pair-self_o-cross_q");
        NSMutableDictionary *all8 = A12S2HSubset(
            canonicalAll, @[@0, @1, @2, @3, @4, @5, @6, @7], lines, @"all8");
        if (!pair || !all8) {
            [lines addObject:@"RESULT=FAIL stage=subset-build"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *pairWeights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(pair, @[donors[1], donors[2]], root, pairWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=pair-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult pairCompile = A12S2EPreparedCompileOnly(
            pair, pairWeights, @"trial-A-pair-canonical", lines);
        if (pairCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=canonical-pair result=%ld", (long)pairCompile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"pairCompile=PASS proceeding=one-load-two-procedures"];
        if (!A12S2GLoadPair(pair, pairWeights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=pair-load-count2"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"pairProof=PASS loadCount=1 procedureCount=2 unloadCount=1"];

        NSMutableDictionary *all8Weights = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(all8, donors, root, all8Weights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=all8-weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult all8Compile = A12S2EPreparedCompileOnly(
            all8, all8Weights, @"trial-B-all8-canonical", lines);
        if (all8Compile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC boundary=canonical-all8 result=%ld", (long)all8Compile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"all8Compile=PASS proceeding=one-load-eight-procedures"];
        BOOL all8Load = A12S2CompileLoad(all8, all8Weights, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (!all8Load) {
            if (![lines.lastObject hasPrefix:@"RESULT="]) {
                [lines addObject:@"RESULT=FAIL stage=all8-load-count8"];
            }
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"RESULT=PASS real-block0-private-multiprocedure pairCount=2 all8Count=8 oneLoadEach"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
