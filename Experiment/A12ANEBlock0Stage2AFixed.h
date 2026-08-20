#import <Foundation/Foundation.h>
#import <TargetConditionals.h>

// Compile-only declaration for the experiment helper. The object guarded by
// isKindOfClass:NSArray is an NSArray at runtime and therefore implements this
// selector; declaring it on NSObject lets ARC type-check dot syntax on `id`.
@interface NSObject (A12Stage2AFirstObjectCompileOnly)
@property(nonatomic, readonly, nullable) id firstObject;
@end

// Keep the original Stage-2A implementation intact, but hide its launch symbol
// so this wrapper can provide the corrected launch path below.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_UnfixedStage2A
#import "../AnimaXS/Runtime/ANE/A12ANEMultiProcProbeV2.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return A12ANETargetedRuntimeProbe_UnfixedStage2A();
}
#else

// Same logic as the Stage-2A helper, except the normalized weight filenames are
// appended to an initially-empty NSMutableArray rather than assigned at an
// out-of-range index. No model/weight semantics change.
static inline BOOL A12S2NormalizeWeightsFixed(
    NSMutableDictionary *plist,
    NSArray<NSDictionary *> *donors,
    NSURL *dumpDirectory,
    NSMutableDictionary *weightMap,
    NSMutableArray<NSString *> *lines) {
    NSArray *networks = [plist[@"Networks"] isKindOfClass:NSArray.class] ? plist[@"Networks"] : @[];
    if (networks.count != donors.count) return NO;
    uint64_t totalBytes = 0;
    for (NSUInteger n = 0; n < networks.count; ++n) {
        NSString *networkName = networks[n];
        NSMutableDictionary *network = [plist[networkName] isKindOfClass:NSDictionary.class]
            ? [plist[networkName] mutableCopy] : nil;
        if (!network) return NO;
        plist[networkName] = network;
        NSArray *refs = [network[@"Weights"] isKindOfClass:NSArray.class] ? network[@"Weights"] : @[];
        NSMutableArray *normalized = [NSMutableArray arrayWithCapacity:refs.count];
        NSURL *donorURL = donors[n][@"url"];
        for (NSUInteger w = 0; w < refs.count; ++w) {
            if (![refs[w] isKindOfClass:NSString.class]) return NO;
            NSString *reference = refs[w];
            NSURL *source = A12S2ResolveWeight(reference, dumpDirectory, donorURL);
            NSError *readError = nil;
            NSData *data = source
                ? [NSData dataWithContentsOfURL:source options:NSDataReadingMappedIfSafe error:&readError]
                : nil;
            if (!data) {
                [lines addObject:[NSString stringWithFormat:
                    @"weightNormalize=FAIL network=%@ ref=%@ source=%@ error=%@",
                    networkName, reference, source.path ?: @"(nil)", A12S2Error(readError)]];
                return NO;
            }
            NSString *newName = [NSString stringWithFormat:@"s2_n%02lu_w%02lu_%@",
                (unsigned long)n, (unsigned long)w,
                reference.lastPathComponent.length ? reference.lastPathComponent : @"weights.bin"];
            [normalized addObject:newName];
            weightMap[newName] = @{@"offset": @0, @"data": data};
            totalBytes += data.length;
            [lines addObject:[NSString stringWithFormat:
                @"weight[%lu,%lu] %@ -> %@ bytes=%lu",
                (unsigned long)n, (unsigned long)w, reference, newName,
                (unsigned long)data.length]];
        }
        network[@"Weights"] = normalized;
    }
    [lines addObject:[NSString stringWithFormat:@"weights=PASS files=%lu totalBytes=%llu",
        (unsigned long)weightMap.count, (unsigned long long)totalBytes]];
    return YES;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2A",
            @"goal=8 production W8 Espresso programs -> one native ANE model -> 8 procedures",
            @"mode=lower exact donors with Espresso; synthesize only ProcedureList when needed; compile/load once",
            nil];

        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2A-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSURL *controlDir = [root URLByAppendingPathComponent:@"control" isDirectory:YES];
        NSURL *multiDir = [root URLByAppendingPathComponent:@"multi" isDirectory:YES];

        NSMutableDictionary *controlPlist = nil;
        if (!A12S2Lower(@[donors[1]], controlDir, @"control-self_o", lines, &controlPlist)) {
            [lines addObject:@"RESULT=FAIL stage=single-donor-lowering-control"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSArray *controlNetworks = [controlPlist[@"Networks"] isKindOfClass:NSArray.class]
            ? controlPlist[@"Networks"] : @[];
        if (controlNetworks.count != 1) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL stage=single-control-networks count=%lu",
                (unsigned long)controlNetworks.count]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *multiPlist = nil;
        if (!A12S2Lower(donors, multiDir, @"multi8", lines, &multiPlist)) {
            [lines addObject:@"RESULT=FAIL stage=multi8-espresso-lowering"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        if (!A12S2EnsureProcedureList(multiPlist, donors, lines)) {
            [lines addObject:@"RESULT=FAIL stage=procedure-wrapper"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *weightMap = [NSMutableDictionary dictionary];
        if (!A12S2NormalizeWeightsFixed(multiPlist, donors, multiDir, weightMap, lines)) {
            [lines addObject:@"RESULT=FAIL stage=weight-normalization"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        BOOL pass = A12S2CompileLoad(multiPlist, weightMap, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (!pass && ![lines.lastObject hasPrefix:@"RESULT="]) {
            [lines addObject:@"RESULT=FAIL stage=real-block0-compile-load"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
