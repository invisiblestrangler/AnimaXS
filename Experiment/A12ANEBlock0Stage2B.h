#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

// Stage 2B keeps the proven Stage-2A helpers but replaces the one incorrect
// assumption from Stage 2A: espresso_dump_ir(plan) serializes only one active
// network even when the plan contains multiple added networks. Therefore each
// production donor is lowered independently, preserving Apple's exact native
// W8 Conv/GOC representation, and the eight resulting native network bodies are
// combined under unique names in one procedure-capable netplist.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_Stage2A
#import "../AnimaXS/Runtime/ANE/A12ANEMultiProcProbeV2.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2B\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12S2BCanonicalizeWeights(
    NSMutableDictionary *network,
    NSURL *dumpDirectory,
    NSURL *donorDirectory,
    NSMutableArray<NSString *> *lines,
    NSString *label) {
    NSArray *refs = [network[@"Weights"] isKindOfClass:NSArray.class]
        ? network[@"Weights"] : @[];
    NSMutableArray<NSString *> *absoluteRefs = [NSMutableArray arrayWithCapacity:refs.count];
    for (NSUInteger w = 0; w < refs.count; ++w) {
        if (![refs[w] isKindOfClass:NSString.class]) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ canonicalWeights=FAIL index=%lu nonString=%@",
                label, (unsigned long)w, A12S2Desc(refs[w])]];
            return NO;
        }
        NSString *reference = refs[w];
        NSURL *source = A12S2ResolveWeight(reference, dumpDirectory, donorDirectory);
        if (!source) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ canonicalWeights=FAIL index=%lu ref=%@",
                label, (unsigned long)w, reference]];
            return NO;
        }
        [absoluteRefs addObject:source.path];
        [lines addObject:[NSString stringWithFormat:
            @"%@ canonicalWeight[%lu] %@ -> %@",
            label, (unsigned long)w, reference, source.path]];
    }
    network[@"Weights"] = absoluteRefs;
    return YES;
}

static inline NSMutableDictionary *A12S2BBuildCombinedPlist(
    NSArray<NSDictionary *> *donors,
    NSURL *root,
    NSMutableArray<NSString *> *lines) {
    NSMutableDictionary *combined = [NSMutableDictionary dictionary];
    combined[@"Version"] = @"1.0.10";
    NSMutableArray<NSString *> *networkNames = [NSMutableArray arrayWithCapacity:donors.count];

    for (NSUInteger i = 0; i < donors.count; ++i) {
        NSDictionary *donor = donors[i];
        NSString *donorName = donor[@"name"];
        NSURL *dumpDirectory = [root URLByAppendingPathComponent:
            [NSString stringWithFormat:@"donor-%02lu-%@", (unsigned long)i, donorName]
            isDirectory:YES];
        NSString *label = [NSString stringWithFormat:@"lower[%lu]-%@",
            (unsigned long)i, donorName];

        NSMutableDictionary *single = nil;
        if (!A12S2Lower(@[donor], dumpDirectory, label, lines, &single)) {
            [lines addObject:[NSString stringWithFormat:
                @"combine=FAIL stage=lower donor=%@", donorName]];
            return nil;
        }

        NSArray *singleNetworks = [single[@"Networks"] isKindOfClass:NSArray.class]
            ? single[@"Networks"] : @[];
        if (singleNetworks.count != 1 || ![singleNetworks.firstObject isKindOfClass:NSString.class]) {
            [lines addObject:[NSString stringWithFormat:
                @"combine=FAIL stage=single-network donor=%@ count=%lu value=%@",
                donorName, (unsigned long)singleNetworks.count, A12S2Desc(singleNetworks)]];
            return nil;
        }

        NSString *sourceName = singleNetworks.firstObject;
        NSMutableDictionary *network = [single[sourceName] isKindOfClass:NSDictionary.class]
            ? [single[sourceName] mutableCopy] : nil;
        if (!network) {
            [lines addObject:[NSString stringWithFormat:
                @"combine=FAIL stage=network-body donor=%@ source=%@",
                donorName, sourceName]];
            return nil;
        }

        if (!A12S2BCanonicalizeWeights(network, dumpDirectory, donor[@"url"], lines, label)) {
            [lines addObject:[NSString stringWithFormat:
                @"combine=FAIL stage=canonical-weights donor=%@", donorName]];
            return nil;
        }

        NSString *networkName = [NSString stringWithFormat:@"net_%02lu_%@",
            (unsigned long)i, donorName];
        combined[networkName] = network;
        [networkNames addObject:networkName];

        NSArray *inputs = [network[@"Inputs"] isKindOfClass:NSArray.class]
            ? network[@"Inputs"] : @[];
        NSArray *outputs = [network[@"Outputs"] isKindOfClass:NSArray.class]
            ? network[@"Outputs"] : @[];
        NSArray *units = [network[@"Units"] isKindOfClass:NSArray.class]
            ? network[@"Units"] : @[];
        NSArray *weights = [network[@"Weights"] isKindOfClass:NSArray.class]
            ? network[@"Weights"] : @[];
        [lines addObject:[NSString stringWithFormat:
            @"combine[%lu]=PASS donor=%@ sourceNetwork=%@ targetNetwork=%@ inputs=%@ outputs=%@ units=%lu weights=%lu",
            (unsigned long)i, donorName, sourceName, networkName,
            inputs, outputs, (unsigned long)units.count, (unsigned long)weights.count]];
    }

    combined[@"Networks"] = networkNames;
    [lines addObject:[NSString stringWithFormat:
        @"combine=PASS version=%@ networks=%lu names=%@",
        combined[@"Version"], (unsigned long)networkNames.count, networkNames]];
    return combined;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2B",
            @"goal=8 individually lowered production W8 networks -> one native ANE model -> 8 procedures",
            @"mode=lower each donor alone; combine exact native bodies; add proven ProcedureList; compile/load once",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2B-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];

        NSMutableDictionary *combined = A12S2BBuildCombinedPlist(donors, root, lines);
        if (!combined) {
            [lines addObject:@"RESULT=FAIL stage=combine-individual-lowered-networks"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        if (!A12S2EnsureProcedureList(combined, donors, lines)) {
            [lines addObject:@"RESULT=FAIL stage=procedure-wrapper"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSArray *procedures = [combined[@"ProcedureList"] isKindOfClass:NSArray.class]
            ? combined[@"ProcedureList"] : @[];
        [lines addObject:[NSString stringWithFormat:
            @"combinedProcedureList=%lu version=%@",
            (unsigned long)procedures.count, combined[@"Version"]]];

        NSMutableDictionary *weightMap = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeights(combined, donors, root, weightMap, lines)) {
            [lines addObject:@"RESULT=FAIL stage=weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        BOOL pass = A12S2CompileLoad(combined, weightMap, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (!pass && ![lines.lastObject hasPrefix:@"RESULT="]) {
            [lines addObject:@"RESULT=FAIL stage=real-block0-combined-compile-load"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
