#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

// Experiment branch only.
//
// Stage 2A: prove that the EIGHT already-working W8 Espresso programs for one
// real DiT block can be lowered into one native ANEC container and loaded as
// eight private ANE procedures. This does not run diffusion and does not alter
// production preparation/runtime code.
//
// Important design rule: do NOT hand-author the quantized Conv/GOC IR. The
// device's Espresso framework lowers the exact existing model.espresso.net /
// model.espresso.shape / model.espresso.weights donors through the same private
// C ABI used by Apple's _ANEEspressoIRTranslator. We only add ProcedureList if
// the dumped IR contains multiple Networks without procedure wrappers.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2A\nRESULT=SKIP simulator";
}
#else

typedef void *(*A12S2CreateContextFn)(uint64_t, uint64_t);
typedef void *(*A12S2CreatePlanFn)(void *, uint64_t);
typedef int (*A12S2AddNetworkFn)(void *, const char *, uint64_t, uint64_t [2]);
typedef int (*A12S2SetMetadataKeyFn)(uint64_t, uint64_t, const char *);
typedef int (*A12S2BuildPlanFn)(void *);
typedef int (*A12S2DumpIRFn)(void *, char **);
typedef int (*A12S2DestroyPlanFn)(void *);
typedef int (*A12S2DestroyContextFn)(void *);

typedef struct {
    void *handle;
    A12S2CreateContextFn createContext;
    A12S2CreatePlanFn createPlan;
    A12S2AddNetworkFn addNetwork;
    A12S2SetMetadataKeyFn setMetadataKey;
    A12S2BuildPlanFn buildPlan;
    A12S2DumpIRFn dumpIR;
    A12S2DestroyPlanFn destroyPlan;
    A12S2DestroyContextFn destroyContext;
    BOOL ok;
} A12S2EspressoAPI;

static inline NSString *A12S2Desc(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (NSException *e) {
        return [NSString stringWithFormat:@"<description threw %@: %@>",
            e.name ?: @"?", e.reason ?: @"?"];
    }
}

static inline NSString *A12S2Error(NSError *error) {
    if (!error) return @"(nil)";
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    NSError *cursor = error;
    for (NSUInteger depth = 0; cursor && depth < 5; ++depth) {
        [parts addObject:[NSString stringWithFormat:@"%@[%ld] %@",
            cursor.domain ?: @"?", (long)cursor.code,
            cursor.localizedDescription ?: A12S2Desc(cursor)]];
        id next = cursor.userInfo[NSUnderlyingErrorKey];
        cursor = [next isKindOfClass:NSError.class] ? next : nil;
    }
    return [parts componentsJoinedByString:@" | underlying: "];
}

static inline A12S2EspressoAPI A12S2Espresso(void) {
    static A12S2EspressoAPI api;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        api.handle = dlopen(
            "/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
            RTLD_NOW | RTLD_LOCAL);
        if (!api.handle) return;
        api.createContext = (A12S2CreateContextFn)dlsym(api.handle, "espresso_create_context");
        api.createPlan = (A12S2CreatePlanFn)dlsym(api.handle, "espresso_create_plan");
        api.addNetwork = (A12S2AddNetworkFn)dlsym(api.handle, "espresso_plan_add_network");
        api.setMetadataKey = (A12S2SetMetadataKeyFn)dlsym(
            api.handle, "espresso_network_compiler_set_metadata_key");
        api.buildPlan = (A12S2BuildPlanFn)dlsym(api.handle, "espresso_plan_build");
        api.dumpIR = (A12S2DumpIRFn)dlsym(api.handle, "espresso_dump_ir");
        api.destroyPlan = (A12S2DestroyPlanFn)dlsym(api.handle, "espresso_plan_destroy");
        api.destroyContext = (A12S2DestroyContextFn)dlsym(api.handle, "espresso_context_destroy");
        if (!api.destroyContext) {
            api.destroyContext = (A12S2DestroyContextFn)dlsym(api.handle, "_espresso_context_destroy");
        }
        api.ok = api.createContext && api.createPlan && api.addNetwork &&
                 api.setMetadataKey && api.buildPlan && api.dumpIR &&
                 api.destroyPlan && api.destroyContext;
    });
    return api;
}

static inline NSArray<NSDictionary *> *A12S2Specs(void) {
    return @[
        @{@"name": @"self_qkv", @"needle": @"selfqkv", @"qkv": @(YES),
          @"inputC": @2048, @"outputC": @2048, @"spatial": @1024},
        @{@"name": @"self_o", @"needle": @"self_o", @"qkv": @(NO),
          @"inputC": @2048, @"outputC": @2048, @"spatial": @1024},
        @{@"name": @"cross_q", @"needle": @"cross_q", @"qkv": @(NO),
          @"inputC": @2048, @"outputC": @2048, @"spatial": @1024},
        @{@"name": @"cross_k", @"needle": @"cross_k", @"qkv": @(NO),
          @"inputC": @1024, @"outputC": @2048, @"spatial": @512},
        @{@"name": @"cross_v", @"needle": @"cross_v", @"qkv": @(NO),
          @"inputC": @1024, @"outputC": @2048, @"spatial": @512},
        @{@"name": @"cross_o", @"needle": @"cross_o", @"qkv": @(NO),
          @"inputC": @2048, @"outputC": @2048, @"spatial": @1024},
        @{@"name": @"mlp_up", @"needle": @"mlp1", @"qkv": @(NO),
          @"inputC": @2048, @"outputC": @8192, @"spatial": @1024},
        @{@"name": @"mlp_down", @"needle": @"mlp2", @"qkv": @(NO),
          @"inputC": @8192, @"outputC": @2048, @"spatial": @1024},
    ];
}

static inline NSString *A12S2MetadataKey(NSDictionary *spec) {
    NSUInteger spatial = [spec[@"spatial"] unsignedIntegerValue];
    NSUInteger inputC = [spec[@"inputC"] unsignedIntegerValue];
    NSUInteger outputC = [spec[@"outputC"] unsignedIntegerValue];
    if ([spec[@"qkv"] boolValue]) {
        return [NSString stringWithFormat:
            @"{\"isegment\":0,\"inputs\":{\"x\":{\"shape\":[%lu,1,1,%lu,1]}},\"outputs\":{\"q\":{\"shape\":[%lu,1,1,%lu,1]},\"k\":{\"shape\":[%lu,1,1,%lu,1]},\"v\":{\"shape\":[%lu,1,1,%lu,1]}}}",
            (unsigned long)spatial, (unsigned long)inputC,
            (unsigned long)spatial, (unsigned long)outputC,
            (unsigned long)spatial, (unsigned long)outputC,
            (unsigned long)spatial, (unsigned long)outputC];
    }
    return [NSString stringWithFormat:
        @"{\"isegment\":0,\"inputs\":{\"x\":{\"shape\":[%lu,1,1,%lu,1]}},\"outputs\":{\"y\":{\"shape\":[%lu,1,1,%lu,1]}}}",
        (unsigned long)spatial, (unsigned long)inputC,
        (unsigned long)spatial, (unsigned long)outputC];
}

static inline NSDate *A12S2ModificationDate(NSURL *url) {
    NSNumber *isDirectory = nil;
    NSDate *date = nil;
    [url getResourceValue:&isDirectory forKey:NSURLIsDirectoryKey error:nil];
    [url getResourceValue:&date forKey:NSURLContentModificationDateKey error:nil];
    return date ?: [NSDate distantPast];
}

static inline NSArray<NSMutableDictionary *> *A12S2FindBlock0Donors(
    NSMutableArray<NSString *> *lines) {
    NSString *cache = [NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    if (!cache) return nil;
    NSURL *root = [[NSURL fileURLWithPath:cache isDirectory:YES]
        URLByAppendingPathComponent:@"AnimaXS-ANE" isDirectory:YES];
    NSError *listError = nil;
    NSArray<NSURL *> *entries = [NSFileManager.defaultManager
        contentsOfDirectoryAtURL:root
        includingPropertiesForKeys:@[NSURLContentModificationDateKey, NSURLIsDirectoryKey]
        options:NSDirectoryEnumerationSkipsHiddenFiles
        error:&listError];
    if (!entries) {
        [lines addObject:[NSString stringWithFormat:@"donorCache=FAIL root=%@ error=%@",
            root.path, A12S2Error(listError)]];
        return nil;
    }

    NSMutableArray<NSMutableDictionary *> *found = [NSMutableArray array];
    for (NSDictionary *baseSpec in A12S2Specs()) {
        NSString *needle = baseSpec[@"needle"];
        NSString *prefix = [NSString stringWithFormat:@"ane-u8-row-v1-b0-%@-", needle];
        NSMutableArray<NSURL *> *matches = [NSMutableArray array];
        for (NSURL *candidate in entries) {
            NSString *name = candidate.lastPathComponent ?: @"";
            if (![name hasPrefix:prefix] || ![name hasSuffix:@".mlmodelc"]) continue;
            NSURL *net = [candidate URLByAppendingPathComponent:@"model.espresso.net"];
            if ([NSFileManager.defaultManager fileExistsAtPath:net.path]) [matches addObject:candidate];
        }
        [matches sortUsingComparator:^NSComparisonResult(NSURL *a, NSURL *b) {
            return [A12S2ModificationDate(b) compare:A12S2ModificationDate(a)];
        }];
        if (matches.count == 0) {
            [lines addObject:[NSString stringWithFormat:@"donor=%@ MISSING prefix=%@",
                baseSpec[@"name"], prefix]];
            return nil;
        }
        NSURL *chosen = matches.firstObject;
        NSMutableDictionary *spec = [baseSpec mutableCopy];
        spec[@"url"] = chosen;
        spec[@"netURL"] = [chosen URLByAppendingPathComponent:@"model.espresso.net"];
        spec[@"metadataKey"] = A12S2MetadataKey(spec);
        [found addObject:spec];
        [lines addObject:[NSString stringWithFormat:@"donor=%@ candidates=%lu chosen=%@",
            spec[@"name"], (unsigned long)matches.count, chosen.lastPathComponent]];
    }
    return found;
}

static inline NSDictionary *A12S2ReadPlist(NSURL *url, NSError **error) {
    NSData *data = [NSData dataWithContentsOfURL:url options:0 error:error];
    if (!data) return nil;
    NSPropertyListFormat format = NSPropertyListBinaryFormat_v1_0;
    id plist = [NSPropertyListSerialization propertyListWithData:data
        options:NSPropertyListMutableContainersAndLeaves format:&format error:error];
    return [plist isKindOfClass:NSDictionary.class] ? plist : nil;
}

static inline BOOL A12S2Lower(
    NSArray<NSDictionary *> *donors,
    NSURL *outputDirectory,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    NSMutableDictionary **plistOut) {
    A12S2EspressoAPI api = A12S2Espresso();
    if (!api.ok) {
        [lines addObject:[NSString stringWithFormat:@"%@ lower=FAIL stage=espresso-symbols", label]];
        return NO;
    }
    NSError *dirError = nil;
    if (![NSFileManager.defaultManager createDirectoryAtURL:outputDirectory
        withIntermediateDirectories:YES attributes:nil error:&dirError]) {
        [lines addObject:[NSString stringWithFormat:@"%@ lower=FAIL stage=mkdir error=%@",
            label, A12S2Error(dirError)]];
        return NO;
    }

    void *ctx = api.createContext(10008ULL, 0xFFFFFFFFULL);
    void *plan = ctx ? api.createPlan(ctx, 0ULL) : NULL;
    if (!ctx || !plan) {
        [lines addObject:[NSString stringWithFormat:@"%@ lower=FAIL stage=context-plan", label]];
        if (plan) api.destroyPlan(plan);
        if (ctx) api.destroyContext(ctx);
        return NO;
    }

    BOOL success = YES;
    for (NSUInteger i = 0; i < donors.count; ++i) {
        NSDictionary *spec = donors[i];
        NSURL *netURL = spec[@"netURL"];
        NSString *key = spec[@"metadataKey"];
        uint64_t networkHandle[2] = {0, 0};
        int addStatus = api.addNetwork(
            plan, netURL.fileSystemRepresentation, 0x10020ULL, networkHandle);
        [lines addObject:[NSString stringWithFormat:
            @"%@ add[%lu]=%@ status=%d handle=(0x%llx,0x%llx)",
            label, (unsigned long)i, spec[@"name"], addStatus,
            (unsigned long long)networkHandle[0], (unsigned long long)networkHandle[1]]];
        if (addStatus != 0) { success = NO; break; }
        int keyStatus = api.setMetadataKey(
            networkHandle[0], networkHandle[1], key.UTF8String);
        [lines addObject:[NSString stringWithFormat:@"%@ metadata[%lu] status=%d",
            label, (unsigned long)i, keyStatus]];
        if (keyStatus != 0) { success = NO; break; }
    }

    if (success) {
        NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
        int buildStatus = api.buildPlan(plan);
        double buildMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
        [lines addObject:[NSString stringWithFormat:@"%@ planBuild status=%d ms=%.2f",
            label, buildStatus, buildMS]];
        success = buildStatus == 0;
    }

    if (success) {
        char pathBuffer[PATH_MAX];
        memset(pathBuffer, 0, sizeof(pathBuffer));
        strlcpy(pathBuffer, outputDirectory.path.fileSystemRepresentation, sizeof(pathBuffer));
        char *pathPointer = pathBuffer;
        NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
        int dumpStatus = api.dumpIR(plan, &pathPointer);
        double dumpMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
        [lines addObject:[NSString stringWithFormat:@"%@ dumpIR status=%d ms=%.2f out=%@",
            label, dumpStatus, dumpMS, outputDirectory.path]];
        success = dumpStatus == 0;
    }

    api.destroyPlan(plan);
    api.destroyContext(ctx);
    if (!success) {
        [lines addObject:[NSString stringWithFormat:@"%@ lower=FAIL", label]];
        return NO;
    }

    NSURL *plistURL = [outputDirectory URLByAppendingPathComponent:@"net.plist"];
    NSError *plistError = nil;
    NSDictionary *raw = A12S2ReadPlist(plistURL, &plistError);
    if (!raw) {
        [lines addObject:[NSString stringWithFormat:@"%@ lower=FAIL stage=read-net-plist error=%@",
            label, A12S2Error(plistError)]];
        return NO;
    }
    NSMutableDictionary *plist = [raw mutableCopy];
    NSArray *networks = [plist[@"Networks"] isKindOfClass:NSArray.class] ? plist[@"Networks"] : @[];
    NSArray *procedures = [plist[@"ProcedureList"] isKindOfClass:NSArray.class] ? plist[@"ProcedureList"] : @[];
    [lines addObject:[NSString stringWithFormat:
        @"%@ lower=PASS version=%@ networks=%lu procedures=%lu topKeys=%@",
        label, plist[@"Version"] ?: @"?", (unsigned long)networks.count,
        (unsigned long)procedures.count, [[plist.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","]]];
    for (NSUInteger i = 0; i < networks.count; ++i) {
        NSString *networkName = [networks[i] isKindOfClass:NSString.class] ? networks[i] : @"?";
        NSDictionary *body = [plist[networkName] isKindOfClass:NSDictionary.class] ? plist[networkName] : @{};
        NSArray *inputs = [body[@"Inputs"] isKindOfClass:NSArray.class] ? body[@"Inputs"] : @[];
        NSArray *outputs = [body[@"Outputs"] isKindOfClass:NSArray.class] ? body[@"Outputs"] : @[];
        NSArray *units = [body[@"Units"] isKindOfClass:NSArray.class] ? body[@"Units"] : @[];
        NSArray *weights = [body[@"Weights"] isKindOfClass:NSArray.class] ? body[@"Weights"] : @[];
        [lines addObject:[NSString stringWithFormat:
            @"%@ nativeNetwork[%lu]=%@ inputs=%@ outputs=%@ units=%lu weights=%@",
            label, (unsigned long)i, networkName, inputs, outputs,
            (unsigned long)units.count, weights]];
    }
    if (plistOut) *plistOut = plist;
    return YES;
}

static inline NSString *A12S2ConsumerUnit(NSDictionary *network, NSString *symbol) {
    NSArray *units = [network[@"Units"] isKindOfClass:NSArray.class] ? network[@"Units"] : @[];
    for (id rawName in units) {
        if (![rawName isKindOfClass:NSString.class]) continue;
        NSString *unitName = rawName;
        NSDictionary *unit = [network[unitName] isKindOfClass:NSDictionary.class] ? network[unitName] : nil;
        id bottom = unit[@"Bottom"];
        if ([bottom isKindOfClass:NSString.class] && [bottom isEqualToString:symbol]) return unitName;
        if ([bottom isKindOfClass:NSArray.class] && [bottom containsObject:symbol]) return unitName;
    }
    return [units.firstObject isKindOfClass:NSString.class] ? units.firstObject : symbol;
}

static inline NSMutableDictionary *A12S2ProcedureInput(
    NSDictionary *network, NSString *symbol) {
    NSDictionary *source = [network[symbol] isKindOfClass:NSDictionary.class] ? network[symbol] : @{};
    NSMutableDictionary *entry = [NSMutableDictionary dictionary];
    entry[@"OperationName"] = @"op0";
    entry[@"Name"] = A12S2ConsumerUnit(network, symbol) ?: symbol;
    entry[@"InputName"] = symbol;
    NSArray *keys = @[@"InputType", @"InputChannels", @"InputInterleave",
                      @"BatchSize", @"InputWidth", @"InputHeight", @"InputDepth"];
    for (NSString *key in keys) if (source[key]) entry[key] = source[key];
    if (!entry[@"InputType"]) entry[@"InputType"] = @"Float16";
    if (!entry[@"InputInterleave"]) entry[@"InputInterleave"] = @1;
    if (!entry[@"BatchSize"]) entry[@"BatchSize"] = @1;
    if (!entry[@"InputDepth"]) entry[@"InputDepth"] = @1;
    return entry;
}

static inline NSMutableDictionary *A12S2ProcedureOutput(
    NSDictionary *network, NSString *symbol) {
    NSDictionary *source = [network[symbol] isKindOfClass:NSDictionary.class] ? network[symbol] : @{};
    id bottom = source[@"Bottom"];
    NSString *unitName = nil;
    if ([bottom isKindOfClass:NSString.class]) unitName = bottom;
    else if ([bottom isKindOfClass:NSArray.class] && [bottom.firstObject isKindOfClass:NSString.class]) unitName = bottom.firstObject;
    if (!unitName) {
        NSArray *units = [network[@"Units"] isKindOfClass:NSArray.class] ? network[@"Units"] : @[];
        unitName = [units.lastObject isKindOfClass:NSString.class] ? units.lastObject : symbol;
    }
    return [@{
        @"OperationName": @"op0",
        @"Name": unitName,
        @"OutputName": symbol,
        @"OutputType": source[@"OutputType"] ?: @"Float16",
        @"OutputInterleave": source[@"OutputInterleave"] ?: @1
    } mutableCopy];
}

static inline BOOL A12S2EnsureProcedureList(
    NSMutableDictionary *plist,
    NSArray<NSDictionary *> *donors,
    NSMutableArray<NSString *> *lines) {
    NSArray *networks = [plist[@"Networks"] isKindOfClass:NSArray.class] ? plist[@"Networks"] : @[];
    NSArray *existing = [plist[@"ProcedureList"] isKindOfClass:NSArray.class] ? plist[@"ProcedureList"] : @[];
    if (existing.count == donors.count) {
        [lines addObject:[NSString stringWithFormat:@"procedureWrapper=EXISTING count=%lu",
            (unsigned long)existing.count]];
        return YES;
    }
    if (networks.count != donors.count) {
        [lines addObject:[NSString stringWithFormat:
            @"procedureWrapper=FAIL networks=%lu donors=%lu existingProcedures=%lu",
            (unsigned long)networks.count, (unsigned long)donors.count,
            (unsigned long)existing.count]];
        return NO;
    }

    NSMutableArray *procedures = [NSMutableArray arrayWithCapacity:donors.count];
    for (NSUInteger i = 0; i < donors.count; ++i) {
        NSString *networkName = [networks[i] isKindOfClass:NSString.class] ? networks[i] : nil;
        NSMutableDictionary *network = [plist[networkName] isKindOfClass:NSDictionary.class]
            ? [plist[networkName] mutableCopy] : nil;
        if (!networkName || !network) return NO;
        plist[networkName] = network;
        NSArray *inputs = [network[@"Inputs"] isKindOfClass:NSArray.class] ? network[@"Inputs"] : @[];
        NSArray *outputs = [network[@"Outputs"] isKindOfClass:NSArray.class] ? network[@"Outputs"] : @[];
        if (inputs.count == 0 || outputs.count == 0) {
            [lines addObject:[NSString stringWithFormat:
                @"procedureWrapper=FAIL network=%@ inputs=%@ outputs=%@",
                networkName, inputs, outputs]];
            return NO;
        }
        NSMutableArray *inputList = [NSMutableArray arrayWithCapacity:inputs.count];
        for (id value in inputs) {
            if (![value isKindOfClass:NSString.class]) return NO;
            [inputList addObject:A12S2ProcedureInput(network, value)];
        }
        NSMutableArray *outputList = [NSMutableArray arrayWithCapacity:outputs.count];
        for (id value in outputs) {
            if (![value isKindOfClass:NSString.class]) return NO;
            [outputList addObject:A12S2ProcedureOutput(network, value)];
        }
        NSString *procedureName = [@"procedure_" stringByAppendingString:donors[i][@"name"]];
        NSDictionary *procedure = @{
            @"Name": procedureName,
            @"InputList": inputList,
            @"OperationList": @[@{@"OperationName": @"op0", @"NetworkName": networkName}],
            @"OutputList": outputList
        };
        [procedures addObject:procedure];
        [lines addObject:[NSString stringWithFormat:
            @"procedureWrapper[%lu]=%@ network=%@ inputs=%lu outputs=%lu",
            (unsigned long)i, procedureName, networkName,
            (unsigned long)inputList.count, (unsigned long)outputList.count]];
    }
    plist[@"ProcedureList"] = procedures;
    [lines addObject:[NSString stringWithFormat:@"procedureWrapper=SYNTHESIZED count=%lu",
        (unsigned long)procedures.count]];
    return YES;
}

static inline NSURL *A12S2ResolveWeight(
    NSString *reference, NSURL *dumpDirectory, NSURL *donorDirectory) {
    NSFileManager *fm = NSFileManager.defaultManager;
    if (reference.length == 0) return nil;
    if (reference.isAbsolutePath && [fm fileExistsAtPath:reference]) {
        return [NSURL fileURLWithPath:reference];
    }
    NSURL *candidate = [dumpDirectory URLByAppendingPathComponent:reference];
    if ([fm fileExistsAtPath:candidate.path]) return candidate;
    candidate = [donorDirectory URLByAppendingPathComponent:reference];
    if ([fm fileExistsAtPath:candidate.path]) return candidate;

    NSString *basename = reference.lastPathComponent;
    NSDirectoryEnumerator<NSURL *> *enumerator = [fm enumeratorAtURL:dumpDirectory
        includingPropertiesForKeys:nil options:NSDirectoryEnumerationSkipsHiddenFiles
        errorHandler:nil];
    for (NSURL *url in enumerator) {
        if ([url.lastPathComponent isEqualToString:basename]) return url;
    }
    return nil;
}

static inline BOOL A12S2NormalizeWeights(
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
            NSData *data = source ? [NSData dataWithContentsOfURL:source options:NSDataReadingMappedIfSafe error:&readError] : nil;
            if (!data) {
                [lines addObject:[NSString stringWithFormat:
                    @"weightNormalize=FAIL network=%@ ref=%@ source=%@ error=%@",
                    networkName, reference, source.path ?: @"(nil)", A12S2Error(readError)]];
                return NO;
            }
            NSString *newName = [NSString stringWithFormat:@"s2_n%02lu_w%02lu_%@",
                (unsigned long)n, (unsigned long)w,
                reference.lastPathComponent.length ? reference.lastPathComponent : @"weights.bin"];
            normalized[w] = newName;
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

static inline NSDictionary *A12S2ANEFDescription(id object) {
    @try {
        id attrs = [object respondsToSelector:NSSelectorFromString(@"modelAttributes")]
            ? ((id(*)(id,SEL))objc_msgSend)(object, NSSelectorFromString(@"modelAttributes")) : nil;
        id desc = [attrs isKindOfClass:NSDictionary.class] ? attrs[@"ANEFModelDescription"] : nil;
        return [desc isKindOfClass:NSDictionary.class] ? desc : @{};
    } @catch (__unused NSException *e) { return @{}; }
}

static inline BOOL A12S2CompileLoad(
    NSMutableDictionary *plist,
    NSDictionary *weightMap,
    NSMutableArray<NSString *> *lines) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"nativeSerialize=FAIL error=%@",
            A12S2Error(serializeError)]];
        return NO;
    }
    [lines addObject:[NSString stringWithFormat:@"nativeSerialize=PASS bytes=%lu",
        (unsigned long)netData.length]];

    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
           RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !memoryModelClass) {
        [lines addObject:@"compile=FAIL stage=ANE-class-discovery"];
        return NO;
    }
    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:@"compile=FAIL stage=descriptor-model"];
        return NO;
    }
    [lines addObject:@"descriptor=PASS model=PASS"];

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:@"compile=FAIL stage=localModelPath"];
            return NO;
        }
        NSFileManager *fm = NSFileManager.defaultManager;
        NSError *ioError = nil;
        if (![fm createDirectoryAtPath:localPath withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![netData writeToFile:[localPath stringByAppendingPathComponent:@"net.plist"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:@"materialize=FAIL error=%@",
                A12S2Error(ioError)]];
            return NO;
        }
        for (NSString *name in weightMap) {
            NSData *data = weightMap[name][@"data"];
            NSString *path = [localPath stringByAppendingPathComponent:name];
            if (![fm createDirectoryAtPath:path.stringByDeletingLastPathComponent
                withIntermediateDirectories:YES attributes:nil error:&ioError] ||
                ![data writeToFile:path options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:@"materialize=FAIL weight=%@ error=%@",
                    name, A12S2Error(ioError)]];
                return NO;
            }
        }
    }
    [lines addObject:[NSString stringWithFormat:@"materialize=PASS cacheHit=%@ localPath=%@",
        cacheHit ? @"yes" : @"no", localPath ?: @"(nil)"]];

    double compileMS = 0.0;
    if (!cacheHit) {
        NSError *compileError = nil;
        NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
            25u, @{}, &compileError);
        compileMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
        if (!compiled) {
            [lines addObject:[NSString stringWithFormat:@"compile=FAIL ms=%.2f error=%@",
                compileMS, A12S2Error(compileError)]];
            return NO;
        }
    }
    [lines addObject:[NSString stringWithFormat:@"compile=PASS ms=%.2f", compileMS]];

    NSError *loadError = nil;
    NSTimeInterval loadStart = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStart) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:@"load=FAIL ms=%.2f error=%@",
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
        @"load=PASS ms=%.2f loadCount=1 procedureCount=%lu names=%@ procedures=%@",
        loadMS, (unsigned long)procedureCount, nameMap, procedures]];

    NSError *unloadError = nil;
    BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
    [lines addObject:[NSString stringWithFormat:@"unload=%@ unloadCount=%d error=%@",
        unloaded ? @"PASS" : @"FAIL", unloaded ? 1 : 0, A12S2Error(unloadError)]];
    BOOL pass = procedureCount == 8 && unloaded;
    [lines addObject:pass
        ? @"RESULT=PASS real-block0-private-multiprocedure loadCount=1 procedureCount=8 unloadCount=1"
        : [NSString stringWithFormat:@"RESULT=FAIL real-block0-private-multiprocedure procedureCount=%lu",
            (unsigned long)procedureCount]];
    return pass;
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
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=single-control-networks count=%lu",
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
        if (!A12S2NormalizeWeights(multiPlist, donors, multiDir, weightMap, lines)) {
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
