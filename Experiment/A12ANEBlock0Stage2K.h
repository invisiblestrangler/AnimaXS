#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <math.h>
#import <string.h>
#import "../AnimaXS/Runtime/ANE/A12ANEBridge.h"
#import "A12ANEBlock0Stage2J.h"

// Stage 2K correctness gate. Stage 2J proved that one real block-0 W8 model can
// compile/load as 10 legal single-output procedures while preserving the target
// of one loaded private ANE model per block. Before production integration, this
// probe compares every multiprocedure output against the CURRENT production
// prepared donor path on exactly the same deterministic FP16 IOSurface input.
//
// Q/K/V compare against the production fused A12ANEQKVModel. The remaining
// seven projections compare against A12ANEProjectionModel. Only one donor model
// is resident beside the 10-procedure model at any time, and every donor is
// invalidated immediately after its comparison.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2KProbe(void) {
    return @"ANE real block0 multi-procedure Stage 2K\nRESULT=SKIP simulator";
}
#else

@interface A12ANESurface (A12Stage2KPrivateSurface)
- (CFTypeRef)a12SurfaceRef;
@end

static inline NSArray<NSNumber *> *A12S2KIndexArray(id raw) {
    if ([raw isKindOfClass:NSArray.class]) return raw;
    if (![raw isKindOfClass:NSIndexSet.class]) return @[];
    NSIndexSet *set = raw;
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:set.count];
    NSUInteger index = set.firstIndex;
    while (index != NSNotFound) {
        [result addObject:@(index)];
        index = [set indexGreaterThanIndex:index];
    }
    return result;
}

static inline NSString *A12S2KCacheKey(NSDictionary *donor) {
    NSURL *url = [donor[@"url"] isKindOfClass:NSURL.class] ? donor[@"url"] : nil;
    NSString *last = url.lastPathComponent;
    return last.length ? last.stringByDeletingPathExtension : nil;
}

static inline void A12S2KFillDeterministic(A12ANESurface *surface) {
    // Small, exactly representable FP16 values avoid introducing overflow while
    // still exercising positive/negative/zero input paths.
    static const uint16_t pattern[] = {
        0x0000u, 0x2800u, 0xA800u, 0x2C00u,
        0xAC00u, 0x3000u, 0xB000u, 0x3400u,
        0xB400u, 0x2000u, 0xA000u, 0x2400u,
        0xA400u, 0x2A00u, 0xAA00u, 0x0000u,
    };
    uint16_t *dst = (uint16_t *)surface.metalBuffer.contents;
    NSUInteger count = surface.byteCount / sizeof(uint16_t);
    for (NSUInteger i = 0; dst && i < count; ++i) dst[i] = pattern[i & 15u];
}

static inline void A12S2KZero(A12ANESurface *surface) {
    void *dst = surface.metalBuffer.contents;
    if (dst) memset(dst, 0, surface.byteCount);
}

static inline float A12S2KHalfFloat(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(bits));
    return (float)value;
}

static inline BOOL A12S2KCompare(
    A12ANESurface *reference,
    A12ANESurface *candidate,
    NSUInteger channels,
    NSUInteger spatial,
    NSString *label,
    NSMutableArray<NSString *> *lines) {
    if (!reference || !candidate || reference.channels != channels || candidate.channels != channels ||
        reference.spatial != spatial || candidate.spatial != spatial) {
        [lines addObject:[NSString stringWithFormat:@"compare[%@]=FAIL stage=shape", label]];
        return NO;
    }
    const uint16_t *a = (const uint16_t *)reference.metalBuffer.contents;
    const uint16_t *b = (const uint16_t *)candidate.metalBuffer.contents;
    if (!a || !b) {
        [lines addObject:[NSString stringWithFormat:@"compare[%@]=FAIL stage=contents", label]];
        return NO;
    }

    NSUInteger mismatches = 0;
    NSUInteger firstIndex = NSNotFound;
    uint16_t firstA = 0, firstB = 0;
    float maxAbs = 0.0f;
    double sumAbs = 0.0;
    NSUInteger finiteDiffs = 0;
    NSUInteger refStride = reference.planeStrideElements;
    NSUInteger candStride = candidate.planeStrideElements;
    for (NSUInteger c = 0; c < channels; ++c) {
        for (NSUInteger s = 0; s < spatial; ++s) {
            NSUInteger logical = c * spatial + s;
            uint16_t av = a[c * refStride + s];
            uint16_t bv = b[c * candStride + s];
            if (av == bv) continue;
            ++mismatches;
            if (firstIndex == NSNotFound) {
                firstIndex = logical;
                firstA = av;
                firstB = bv;
            }
            float af = A12S2KHalfFloat(av);
            float bf = A12S2KHalfFloat(bv);
            if (isfinite(af) && isfinite(bf)) {
                float d = fabsf(af - bf);
                if (d > maxAbs) maxAbs = d;
                sumAbs += d;
                ++finiteDiffs;
            }
        }
    }
    NSUInteger total = channels * spatial;
    double meanAbs = finiteDiffs ? sumAbs / (double)finiteDiffs : 0.0;
    BOOL exact = mismatches == 0;
    [lines addObject:[NSString stringWithFormat:
        @"compare[%@]=%@ exact=%@ mismatches=%lu/%lu maxAbs=%.9g meanAbs=%.9g firstIndex=%@ ref=0x%04x candidate=0x%04x",
        label, exact ? @"PASS" : @"DIFF", exact ? @"yes" : @"no",
        (unsigned long)mismatches, (unsigned long)total,
        maxAbs, meanAbs,
        firstIndex == NSNotFound ? @"none" : [NSString stringWithFormat:@"%lu", (unsigned long)firstIndex],
        firstA, firstB]];
    return exact;
}

static inline id A12S2KSurfaceObject(A12ANESurface *surface) {
    if (!surface) return nil;
    Class cls = NSClassFromString(@"_ANEIOSurfaceObject");
    SEL surfaceSel = NSSelectorFromString(@"a12SurfaceRef");
    if (!cls || ![surface respondsToSelector:surfaceSel]) return nil;
    CFTypeRef raw = ((CFTypeRef(*)(id,SEL))objc_msgSend)(surface, surfaceSel);
    return raw ? ((id(*)(Class,SEL,void *))objc_msgSend)(
        cls, NSSelectorFromString(@"objectWithIOSurface:"), (void *)raw) : nil;
}

static inline BOOL A12S2KEvaluateProcedure(
    id memoryModel,
    id loweredModel,
    NSUInteger procedureIndex,
    A12ANESurface *input,
    A12ANESurface *output,
    NSString *label,
    NSMutableArray<NSString *> *lines) {
    SEL inSel = NSSelectorFromString(@"inputSymbolIndicesForProcedureIndex:");
    SEL outSel = NSSelectorFromString(@"outputSymbolIndicesForProcedureIndex:");
    NSArray<NSNumber *> *inputIndices = A12S2KIndexArray(
        [loweredModel respondsToSelector:inSel]
            ? ((id(*)(id,SEL,NSUInteger))objc_msgSend)(loweredModel, inSel, procedureIndex) : nil);
    NSArray<NSNumber *> *outputIndices = A12S2KIndexArray(
        [loweredModel respondsToSelector:outSel]
            ? ((id(*)(id,SEL,NSUInteger))objc_msgSend)(loweredModel, outSel, procedureIndex) : nil);
    if (inputIndices.count != 1 || outputIndices.count != 1) {
        [lines addObject:[NSString stringWithFormat:
            @"multiproc[%@]=FAIL stage=symbol-indices procedure=%lu inputs=%@ outputs=%@",
            label, (unsigned long)procedureIndex, inputIndices, outputIndices]];
        return NO;
    }

    id inputObject = A12S2KSurfaceObject(input);
    id outputObject = A12S2KSurfaceObject(output);
    Class requestClass = NSClassFromString(@"_ANERequest");
    if (!inputObject || !outputObject || !requestClass) {
        [lines addObject:[NSString stringWithFormat:@"multiproc[%@]=FAIL stage=surface-wrap", label]];
        return NO;
    }
    id request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
        requestClass,
        NSSelectorFromString(@"requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:"),
        @[inputObject], inputIndices, @[outputObject], outputIndices, @(procedureIndex));
    if (!request) {
        [lines addObject:[NSString stringWithFormat:@"multiproc[%@]=FAIL stage=request", label]];
        return NO;
    }

    NSError *mapError = nil;
    BOOL mapped = ((BOOL(*)(id,SEL,id,BOOL,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"mapIOSurfacesWithRequest:cacheInference:error:"),
        request, YES, &mapError);
    if (!mapped) {
        [lines addObject:[NSString stringWithFormat:@"multiproc[%@]=FAIL stage=map error=%@",
            label, A12S2Error(mapError)]];
        return NO;
    }

    NSError *evalError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL evaluated = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
        25u, @{}, request, &evalError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if ([memoryModel respondsToSelector:NSSelectorFromString(@"unmapIOSurfacesWithRequest:")]) {
        ((void(*)(id,SEL,id))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unmapIOSurfacesWithRequest:"), request);
    }
    [lines addObject:[NSString stringWithFormat:@"multiproc[%@] eval=%@ ms=%.3f procedure=%lu error=%@",
        label, evaluated ? @"PASS" : @"FAIL", ms,
        (unsigned long)procedureIndex, A12S2Error(evalError)]];
    return evaluated;
}

static inline id A12S2KLoadTenProcedureModel(
    NSDictionary *plist,
    NSDictionary *weights,
    NSMutableArray<NSString *> *lines,
    id *loweredModelOut,
    NSDictionary **nameMapOut) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"full10Load=FAIL stage=serialize error=%@",
            A12S2Error(serializeError)]];
        return nil;
    }
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weights, nil) : nil;
    id memoryModel = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!memoryModel) {
        [lines addObject:@"full10Load=FAIL stage=descriptor-model"];
        return nil;
    }
    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    if (!cacheHit) {
        NSError *compileError = nil;
        NSTimeInterval compileStart = NSDate.timeIntervalSinceReferenceDate;
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"), 25u, @{}, &compileError);
        double compileMS = (NSDate.timeIntervalSinceReferenceDate - compileStart) * 1000.0;
        [lines addObject:[NSString stringWithFormat:@"full10Load compile=%@ ms=%.2f error=%@",
            compiled ? @"PASS" : @"FAIL", compileMS, A12S2Error(compileError)]];
        if (!compiled) return nil;
    } else {
        [lines addObject:@"full10Load compile=CACHE_HIT"];
    }

    NSError *loadError = nil;
    NSTimeInterval loadStart = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"), 25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStart) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:@"full10Load=FAIL stage=load ms=%.2f error=%@",
            loadMS, A12S2Error(loadError)]];
        return nil;
    }
    id loweredModel = [memoryModel respondsToSelector:NSSelectorFromString(@"model")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"model")) : nil;
    NSDictionary *desc = A12S2ANEFDescription(loweredModel ?: memoryModel);
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSUInteger count = MAX(nameMap.count, procedures.count);
    [lines addObject:[NSString stringWithFormat:
        @"full10Load=PASS ms=%.2f loadCount=1 procedureCount=%lu names=%@",
        loadMS, (unsigned long)count, nameMap]];
    if (count != 10 || !loweredModel) {
        NSError *unloadError = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
        [lines addObject:@"full10Load=FAIL stage=procedure-count"];
        return nil;
    }
    if (loweredModelOut) *loweredModelOut = loweredModel;
    if (nameMapOut) *nameMapOut = nameMap;
    return memoryModel;
}

static inline BOOL A12S2KUnload(id memoryModel, NSMutableArray<NSString *> *lines) {
    if (!memoryModel) return YES;
    NSError *error = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &error);
    [lines addObject:[NSString stringWithFormat:@"full10Unload=%@ unloadCount=%d error=%@",
        ok ? @"PASS" : @"FAIL", ok ? 1 : 0, A12S2Error(error)]];
    return ok;
}

static inline NSNumber *A12S2KProcedureID(NSDictionary *nameMap, NSString *name) {
    id raw = nameMap[name];
    return [raw respondsToSelector:@selector(unsignedIntegerValue)] ? @([raw unsignedIntegerValue]) : nil;
}

static inline BOOL A12S2KRunProjectionComparison(
    id memoryModel,
    id loweredModel,
    NSDictionary *nameMap,
    NSDictionary *donor,
    NSString *procedureName,
    NSUInteger inputChannels,
    NSUInteger outputChannels,
    NSUInteger spatial,
    NSMutableArray<NSString *> *lines) {
    NSNumber *procedureID = A12S2KProcedureID(nameMap, procedureName);
    NSString *cacheKey = A12S2KCacheKey(donor);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *surfaceError = nil;
    A12ANESurface *input = [[A12ANESurface alloc] initWithDevice:device
        channels:inputChannels spatial:spatial error:&surfaceError];
    A12ANESurface *reference = [[A12ANESurface alloc] initWithDevice:device
        channels:outputChannels spatial:spatial error:&surfaceError];
    A12ANESurface *candidate = [[A12ANESurface alloc] initWithDevice:device
        channels:outputChannels spatial:spatial error:&surfaceError];
    if (!procedureID || !cacheKey || !input || !reference || !candidate) {
        [lines addObject:[NSString stringWithFormat:
            @"trial[%@]=FAIL stage=setup procedureID=%@ cacheKey=%@ surfaceError=%@",
            procedureName, procedureID, cacheKey, A12S2Error(surfaceError)]];
        return NO;
    }
    A12S2KFillDeterministic(input);
    A12S2KZero(reference);
    A12S2KZero(candidate);

    NSError *baselineError = nil;
    A12ANEProjectionModel *baseline = [[A12ANEProjectionModel alloc]
        initPreparedWithInputChannels:inputChannels
        outputChannels:outputChannels spatial:spatial
        label:[@"stage2k_" stringByAppendingString:procedureName]
        cacheKey:cacheKey error:&baselineError];
    if (!baseline) {
        [lines addObject:[NSString stringWithFormat:@"trial[%@]=FAIL stage=baseline-load error=%@",
            procedureName, A12S2Error(baselineError)]];
        return NO;
    }
    double baselineMS = 0.0;
    BOOL baselineOK = [baseline evaluateInput:input output:reference
        milliseconds:&baselineMS error:&baselineError];
    [baseline invalidate];
    [lines addObject:[NSString stringWithFormat:@"baseline[%@] eval=%@ ms=%.3f error=%@",
        procedureName, baselineOK ? @"PASS" : @"FAIL", baselineMS, A12S2Error(baselineError)]];
    if (!baselineOK) return NO;

    BOOL multiOK = A12S2KEvaluateProcedure(
        memoryModel, loweredModel, procedureID.unsignedIntegerValue,
        input, candidate, procedureName, lines);
    if (!multiOK) return NO;
    return A12S2KCompare(reference, candidate, outputChannels, spatial, procedureName, lines);
}

static inline BOOL A12S2KRunQKVComparisons(
    id memoryModel,
    id loweredModel,
    NSDictionary *nameMap,
    NSDictionary *donor,
    NSMutableArray<NSString *> *lines) {
    NSString *cacheKey = A12S2KCacheKey(donor);
    NSNumber *qID = A12S2KProcedureID(nameMap, @"procedure_self_q");
    NSNumber *kID = A12S2KProcedureID(nameMap, @"procedure_self_k");
    NSNumber *vID = A12S2KProcedureID(nameMap, @"procedure_self_v");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    A12ANESurface *input = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&error];
    A12ANESurface *qRef = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&error];
    A12ANESurface *kRef = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&error];
    A12ANESurface *vRef = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&error];
    A12ANESurface *candidate = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&error];
    if (!cacheKey || !qID || !kID || !vID || !input || !qRef || !kRef || !vRef || !candidate) {
        [lines addObject:[NSString stringWithFormat:@"qkvCorrectness=FAIL stage=setup error=%@", A12S2Error(error)]];
        return NO;
    }
    A12S2KFillDeterministic(input);
    A12S2KZero(qRef); A12S2KZero(kRef); A12S2KZero(vRef);

    A12ANEQKVModel *baseline = [[A12ANEQKVModel alloc]
        initPreparedWithChannels:2048 spatial:1024
        label:@"stage2k_self_qkv" cacheKey:cacheKey error:&error];
    if (!baseline) {
        [lines addObject:[NSString stringWithFormat:@"qkvCorrectness=FAIL stage=baseline-load error=%@", A12S2Error(error)]];
        return NO;
    }
    double baselineMS = 0.0;
    BOOL baselineOK = [baseline evaluateInput:input qOutput:qRef kOutput:kRef vOutput:vRef
        milliseconds:&baselineMS error:&error];
    [baseline invalidate];
    [lines addObject:[NSString stringWithFormat:@"baseline[self_qkv] eval=%@ ms=%.3f error=%@",
        baselineOK ? @"PASS" : @"FAIL", baselineMS, A12S2Error(error)]];
    if (!baselineOK) return NO;

    NSArray<NSDictionary *> *parts = @[
        @{@"name": @"procedure_self_q", @"id": qID, @"ref": qRef},
        @{@"name": @"procedure_self_k", @"id": kID, @"ref": kRef},
        @{@"name": @"procedure_self_v", @"id": vID, @"ref": vRef},
    ];
    for (NSDictionary *part in parts) {
        NSString *name = part[@"name"];
        NSNumber *pid = part[@"id"];
        A12ANESurface *ref = part[@"ref"];
        A12S2KZero(candidate);
        if (!A12S2KEvaluateProcedure(memoryModel, loweredModel, pid.unsignedIntegerValue,
                                     input, candidate, name, lines) ||
            !A12S2KCompare(ref, candidate, 2048, 1024, name, lines)) {
            return NO;
        }
    }
    return YES;
}

static inline NSString *A12ANEStage2KProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE real block0 multi-procedure Stage 2K",
            @"goal=dispatch all 10 procedures and compare against current production prepared donors",
            @"criterion=exact logical FP16 equality on identical deterministic IOSurface inputs",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSArray<NSMutableDictionary *> *donors = A12S2FindBlock0Donors(lines);
        if (donors.count != 8) {
            [lines addObject:@"RESULT=SKIP missing-real-block0-donors"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"AnimaXS-ANE-S2K-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *canonicalAll = A12S2HBuildCanonicalCombined(donors, root, runtimeVersion, lines);
        NSMutableDictionary *full10 = canonicalAll ? A12S2JSplitQKV(canonicalAll, lines) : nil;
        NSArray<NSDictionary *> *donorMap = A12S2JDonorMap(donors);
        if (!full10 || donorMap.count != 10) {
            [lines addObject:@"RESULT=FAIL stage=build-full10"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSMutableDictionary *all10 = A12S2HSubset(
            full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9], lines, @"correctness-full10");
        NSMutableDictionary *weights = [NSMutableDictionary dictionary];
        if (!all10 || !A12S2JNormalizeWeightsDedup(all10, donorMap, root, weights, lines)) {
            [lines addObject:@"RESULT=FAIL stage=normalize-full10"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2ECompileResult compile = A12S2EPreparedCompileOnly(
            all10, weights, @"correctness-full10", lines);
        if (compile != A12S2ECompilePass) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=compile-full10 result=%ld", (long)compile]];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        id loweredModel = nil;
        NSDictionary *nameMap = nil;
        id memoryModel = A12S2KLoadTenProcedureModel(all10, weights, lines, &loweredModel, &nameMap);
        if (!memoryModel) {
            [lines addObject:@"RESULT=FAIL stage=load-full10"];
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return [lines componentsJoinedByString:@"\n"];
        }

        BOOL pass = A12S2KRunQKVComparisons(memoryModel, loweredModel, nameMap, donors[0], lines);
        NSArray<NSDictionary *> *projectionTrials = @[
            @{@"name": @"procedure_self_o", @"donor": @1, @"in": @2048, @"out": @2048, @"spatial": @1024},
            @{@"name": @"procedure_cross_q", @"donor": @2, @"in": @2048, @"out": @2048, @"spatial": @1024},
            @{@"name": @"procedure_cross_k", @"donor": @3, @"in": @1024, @"out": @2048, @"spatial": @512},
            @{@"name": @"procedure_cross_v", @"donor": @4, @"in": @1024, @"out": @2048, @"spatial": @512},
            @{@"name": @"procedure_cross_o", @"donor": @5, @"in": @2048, @"out": @2048, @"spatial": @1024},
            @{@"name": @"procedure_mlp_up", @"donor": @6, @"in": @2048, @"out": @8192, @"spatial": @1024},
            @{@"name": @"procedure_mlp_down", @"donor": @7, @"in": @8192, @"out": @2048, @"spatial": @1024},
        ];
        if (pass) {
            for (NSDictionary *trial in projectionTrials) {
                NSUInteger donorIndex = [trial[@"donor"] unsignedIntegerValue];
                pass = A12S2KRunProjectionComparison(
                    memoryModel, loweredModel, nameMap, donors[donorIndex],
                    trial[@"name"], [trial[@"in"] unsignedIntegerValue],
                    [trial[@"out"] unsignedIntegerValue],
                    [trial[@"spatial"] unsignedIntegerValue], lines);
                if (!pass) break;
            }
        }

        BOOL unloaded = A12S2KUnload(memoryModel, lines);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (pass && unloaded) {
            [lines addObject:@"RESULT=PASS all10-dispatch bitExactAgainstProduction loadedModelsPerBlock=1 targetLoadedModels=28"];
        } else {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL correctness=%@ unload=%@",
                pass ? @"PASS" : @"FAIL", unloaded ? @"PASS" : @"FAIL"]];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
