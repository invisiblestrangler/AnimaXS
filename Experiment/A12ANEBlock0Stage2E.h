#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <dlfcn.h>

// Stage 2E: Stage 2D proved the harvested real W8 ANECIR itself recompiles and
// Version=1.0.10 is accepted; adding our synthesized ProcedureList is the first
// failing transformation. Match Apple's _MLCANEPlistBuilder grammar more
// literally: network_* names and shape-complete live-input entries sourced from
// model.espresso.shape. First prove one self_o procedure, then a self_o+cross_q
// pair and finally load once to verify two emitted procedures.
#define A12ANETargetedRuntimeProbe A12ANETargetedRuntimeProbe_Stage2A_Base
#import "../AnimaXS/Runtime/ANE/A12ANEMultiProcProbeV2.h"
#undef A12ANETargetedRuntimeProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2E\nRESULT=SKIP simulator";
}
#else

typedef NS_ENUM(NSInteger, A12S2ECompileResult) {
    A12S2ECompilePass = 0,
    A12S2ECompileFail = 1,
    A12S2EHelperLost = 2,
};

static inline NSDictionary *A12S2EShapeMap(NSURL *donorURL, NSMutableArray<NSString *> *lines, NSString *label) {
    NSURL *shapeURL = [donorURL URLByAppendingPathComponent:@"model.espresso.shape"];
    NSData *data = [NSData dataWithContentsOfURL:shapeURL];
    if (!data) {
        [lines addObject:[NSString stringWithFormat:@"%@ shape=FAIL path=%@", label, shapeURL.path]];
        return nil;
    }
    NSError *error = nil;
    id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    NSDictionary *root = [json isKindOfClass:NSDictionary.class] ? json : nil;
    NSDictionary *map = [root[@"layer_shapes"] isKindOfClass:NSDictionary.class] ? root[@"layer_shapes"] : nil;
    if (!map) {
        [lines addObject:[NSString stringWithFormat:@"%@ shape=FAIL error=%@ json=%@", label, A12S2Error(error), A12S2Desc(root)]];
        return nil;
    }
    [lines addObject:[NSString stringWithFormat:@"%@ shape=PASS keys=%@ x=%@ y=%@",
        label, [[map.allKeys sortedArrayUsingSelector:@selector(compare:)] componentsJoinedByString:@","],
        A12S2Desc(map[@"x"]), A12S2Desc(map[@"y"])]];
    return map;
}

static inline NSMutableDictionary *A12S2ELiveInput(NSDictionary *network, NSString *symbol, NSDictionary *shape) {
    NSString *unitName = A12S2ConsumerUnit(network, symbol) ?: symbol;
    NSNumber *w = [shape[@"w"] respondsToSelector:@selector(unsignedIntegerValue)] ? shape[@"w"] : @1;
    NSNumber *h = [shape[@"h"] respondsToSelector:@selector(unsignedIntegerValue)] ? shape[@"h"] : @1;
    NSNumber *k = [shape[@"k"] respondsToSelector:@selector(unsignedIntegerValue)] ? shape[@"k"] : @1;
    NSNumber *n = [shape[@"n"] respondsToSelector:@selector(unsignedIntegerValue)] ? shape[@"n"] : @1;
    return [@{
        @"OperationName": @"op0",
        @"Name": unitName,
        @"InputName": symbol,
        @"InputType": @"Float16",
        @"InputWidth": w,
        @"InputHeight": h,
        @"InputDepth": @1,
        @"InputChannels": k,
        @"InputInterleave": @1,
        @"BatchSize": n,
    } mutableCopy];
}

static inline NSMutableDictionary *A12S2ELiveOutput(NSDictionary *network, NSString *symbol) {
    NSDictionary *source = [network[symbol] isKindOfClass:NSDictionary.class] ? network[symbol] : @{};
    id bottom = source[@"Bottom"];
    NSString *unitName = nil;
    if ([bottom isKindOfClass:NSString.class]) unitName = bottom;
    else if ([bottom isKindOfClass:NSArray.class] && [((NSArray *)bottom).firstObject isKindOfClass:NSString.class]) {
        unitName = ((NSArray *)bottom).firstObject;
    }
    if (!unitName) {
        NSArray *units = [network[@"Units"] isKindOfClass:NSArray.class] ? network[@"Units"] : @[];
        unitName = [units.lastObject isKindOfClass:NSString.class] ? units.lastObject : symbol;
    }
    return [@{
        @"OperationName": @"op0",
        @"Name": unitName,
        @"OutputName": symbol,
        @"OutputType": source[@"OutputType"] ?: @"Float16",
        @"OutputInterleave": source[@"OutputInterleave"] ?: @1,
    } mutableCopy];
}

static inline BOOL A12S2ECanonicalizeWeights(NSMutableDictionary *network,
                                              NSURL *dumpDir,
                                              NSURL *donorURL,
                                              NSMutableArray<NSString *> *lines,
                                              NSString *label) {
    NSArray *refs = [network[@"Weights"] isKindOfClass:NSArray.class] ? network[@"Weights"] : @[];
    NSMutableArray *absolute = [NSMutableArray arrayWithCapacity:refs.count];
    for (NSUInteger i = 0; i < refs.count; ++i) {
        if (![refs[i] isKindOfClass:NSString.class]) return NO;
        NSURL *source = A12S2ResolveWeight(refs[i], dumpDir, donorURL);
        if (!source) {
            [lines addObject:[NSString stringWithFormat:@"%@ canonicalWeight=FAIL index=%lu ref=%@",
                label, (unsigned long)i, refs[i]]];
            return NO;
        }
        [absolute addObject:source.path];
    }
    network[@"Weights"] = absolute;
    return YES;
}

static inline NSMutableDictionary *A12S2EProcedureContainer(NSDictionary *raw,
                                                             NSDictionary *donor,
                                                             NSURL *dumpDir,
                                                             NSString *networkName,
                                                             NSMutableArray<NSString *> *lines) {
    NSArray *rawNetworks = [raw[@"Networks"] isKindOfClass:NSArray.class] ? raw[@"Networks"] : @[];
    NSString *sourceName = rawNetworks.count == 1 && [rawNetworks.firstObject isKindOfClass:NSString.class]
        ? rawNetworks.firstObject : nil;
    NSMutableDictionary *network = sourceName && [raw[sourceName] isKindOfClass:NSDictionary.class]
        ? [raw[sourceName] mutableCopy] : nil;
    if (!network) return nil;
    if (!A12S2ECanonicalizeWeights(network, dumpDir, donor[@"url"], lines, donor[@"name"])) return nil;

    NSDictionary *shapes = A12S2EShapeMap(donor[@"url"], lines, donor[@"name"]);
    if (!shapes) return nil;
    NSArray *inputs = [network[@"Inputs"] isKindOfClass:NSArray.class] ? network[@"Inputs"] : @[];
    NSArray *outputs = [network[@"Outputs"] isKindOfClass:NSArray.class] ? network[@"Outputs"] : @[];
    NSMutableArray *inputList = [NSMutableArray arrayWithCapacity:inputs.count];
    for (id rawSymbol in inputs) {
        if (![rawSymbol isKindOfClass:NSString.class]) return nil;
        NSDictionary *shape = [shapes[rawSymbol] isKindOfClass:NSDictionary.class] ? shapes[rawSymbol] : nil;
        if (!shape) {
            [lines addObject:[NSString stringWithFormat:@"%@ wrapper=FAIL missingShape input=%@", donor[@"name"], rawSymbol]];
            return nil;
        }
        [inputList addObject:A12S2ELiveInput(network, rawSymbol, shape)];
    }
    NSMutableArray *outputList = [NSMutableArray arrayWithCapacity:outputs.count];
    for (id rawSymbol in outputs) {
        if (![rawSymbol isKindOfClass:NSString.class]) return nil;
        [outputList addObject:A12S2ELiveOutput(network, rawSymbol)];
    }

    NSString *procedureName = [@"procedure_" stringByAppendingString:donor[@"name"]];
    NSDictionary *procedure = @{
        @"Name": procedureName,
        @"InputList": inputList,
        @"OperationList": @[@{@"OperationName": @"op0", @"NetworkName": networkName}],
        @"OutputList": outputList,
    };
    NSMutableDictionary *container = [NSMutableDictionary dictionary];
    container[@"Version"] = @"1.0.10";
    container[@"Networks"] = @[networkName];
    container[networkName] = network;
    container[@"ProcedureList"] = @[procedure];
    [lines addObject:[NSString stringWithFormat:
        @"%@ appleParityWrapper network=%@ procedure=%@ inputs=%@ outputs=%@ input0=%@ output0=%@ xDict=%@ outDict=%@",
        donor[@"name"], networkName, procedureName, inputs, outputs,
        A12S2Desc(inputList.firstObject), A12S2Desc(outputList.firstObject),
        A12S2Desc(network[@"x"]), outputs.count ? A12S2Desc(network[outputs.firstObject]) : @"(none)"]];
    return container;
}

static inline uint64_t A12S2EWeightBytes(NSDictionary *weightMap) {
    uint64_t total = 0;
    for (NSString *name in weightMap) {
        NSDictionary *entry = [weightMap[name] isKindOfClass:NSDictionary.class] ? weightMap[name] : nil;
        NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
        if (data) total += data.length;
    }
    return total;
}

static inline A12S2ECompileResult A12S2ECompileOnly(NSDictionary *plist,
                                                     NSDictionary *weightMap,
                                                     NSString *label,
                                                     NSMutableArray<NSString *> *lines) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"%@ serialize=FAIL error=%@", label, A12S2Error(serializeError)]];
        return A12S2ECompileFail;
    }
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW | RTLD_LOCAL);
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass, NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"), netData, weightMap, nil) : nil;
    id memoryModel = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL stage=descriptor-model", label]];
        return A12S2ECompileFail;
    }
    NSError *compileError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"), 25u, @{}, &compileError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (!compiled) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ compile=FAIL ms=%.2f netBytes=%lu weightBytes=%llu errorDomain=%@ errorCode=%ld error=%@",
            label, ms, (unsigned long)netData.length, (unsigned long long)A12S2EWeightBytes(weightMap),
            compileError.domain ?: @"(nil)", (long)compileError.code, A12S2Error(compileError)]];
        if ([compileError.domain isEqualToString:NSCocoaErrorDomain] && compileError.code == 4097) return A12S2EHelperLost;
        return A12S2ECompileFail;
    }
    [lines addObject:[NSString stringWithFormat:@"%@ compile=PASS ms=%.2f netBytes=%lu weightBytes=%llu",
        label, ms, (unsigned long)netData.length, (unsigned long long)A12S2EWeightBytes(weightMap)]];
    return A12S2ECompilePass;
}

static inline NSMutableDictionary *A12S2EMergePair(NSDictionary *a,
                                                     NSDictionary *b,
                                                     NSMutableArray<NSString *> *lines) {
    NSString *aName = [a[@"Networks"] isKindOfClass:NSArray.class] ? [a[@"Networks"] firstObject] : nil;
    NSString *bName = [b[@"Networks"] isKindOfClass:NSArray.class] ? [b[@"Networks"] firstObject] : nil;
    if (![aName isKindOfClass:NSString.class] || ![bName isKindOfClass:NSString.class]) return nil;
    NSMutableDictionary *out = [NSMutableDictionary dictionary];
    out[@"Version"] = @"1.0.10";
    out[@"Networks"] = @[aName, bName];
    out[aName] = [a[aName] mutableCopy];
    out[bName] = [b[bName] mutableCopy];
    NSArray *aProcedures = [a[@"ProcedureList"] isKindOfClass:NSArray.class] ? a[@"ProcedureList"] : @[];
    NSArray *bProcedures = [b[@"ProcedureList"] isKindOfClass:NSArray.class] ? b[@"ProcedureList"] : @[];
    if (aProcedures.count != 1 || bProcedures.count != 1) return nil;
    out[@"ProcedureList"] = @[aProcedures.firstObject, bProcedures.firstObject];
    [lines addObject:[NSString stringWithFormat:@"pair=BUILT networks=%@ procedures=%@",
        out[@"Networks"], [out[@"ProcedureList"] valueForKey:@"Name"]]];
    return out;
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2E",
            @"goal=match Apple _MLCANEPlistBuilder procedure schema for real W8 lowered networks",
            @"matrix=single self_o Apple-parity wrapper -> self_o+cross_q Apple-parity pair -> one load/two procedures",
            nil];
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *selfO = donors[1];
        NSDictionary *crossQ = donors[2];
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2E-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
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
        A12S2ECompileResult single = A12S2ECompileOnly(selfContainer, selfWeights, @"trial-A-single-apple-parity", lines);
        if (single != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=DIAGNOSTIC stage=single-apple-parity result=%ld", (long)single]];
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
        A12S2ECompileResult pairCompile = A12S2ECompileOnly(pair, pairWeights, @"trial-B-pair-apple-parity", lines);
        if (pairCompile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=DIAGNOSTIC stage=pair-apple-parity result=%ld", (long)pairCompile]];
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
