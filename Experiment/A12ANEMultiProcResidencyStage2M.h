#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>
#import <unistd.h>
#import "A12ANEMultiProcResidencyStage2L.h"

// Stage 2M isolates the two quantities Stage 2L intentionally mixed:
//   1. first-time construction/compile admission cost, and
//   2. steady-state ANE residency of already-compiled multi-procedure models.
//
// Phase A compiles every real 10-procedure block one at a time, dispatches all
// ten procedures, unloads it, verifies a newly-constructed descriptor sees the
// compiled cache, and retains only the descriptor-cleared unloaded model handle.
//
// Phase B progressively reloads those SAME unloaded handles. No donor lowering,
// plist construction, 189 MB weight map, materialization, or compilation occurs
// in this phase. It therefore measures the clean loaded-program residency ceiling.
//
// Phase C, after Phase B cleanup, cycles representative unloaded handles through
// reload -> self_o dispatch -> unload so scheduler load/unload costs are visible.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2MProbe(void) {
    return @"ANE multiprocedure cache-hit residency Stage 2M\nRESULT=SKIP simulator";
}
#else

static inline NSArray<NSString *> *A12S2MExpectedProcedureNames(void) {
    return @[
        @"procedure_self_q", @"procedure_self_k", @"procedure_self_v",
        @"procedure_self_o", @"procedure_cross_q", @"procedure_cross_k",
        @"procedure_cross_v", @"procedure_cross_o",
        @"procedure_mlp_up", @"procedure_mlp_down",
    ];
}

static inline NSArray<NSDictionary *> *A12S2MProcedureTrials(void) {
    return @[
        @{@"name": @"procedure_self_q",   @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_self_k",   @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_self_v",   @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_self_o",   @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_cross_q",  @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_cross_k",  @"in": @1024, @"out": @2048, @"spatial": @512},
        @{@"name": @"procedure_cross_v",  @"in": @1024, @"out": @2048, @"spatial": @512},
        @{@"name": @"procedure_cross_o",  @"in": @2048, @"out": @2048, @"spatial": @1024},
        @{@"name": @"procedure_mlp_up",   @"in": @2048, @"out": @8192, @"spatial": @1024},
        @{@"name": @"procedure_mlp_down", @"in": @8192, @"out": @2048, @"spatial": @1024},
    ];
}

static inline BOOL A12S2MNameMapHasAllTen(
    NSDictionary *nameMap,
    NSMutableArray<NSString *> *lines,
    NSString *label) {
    if (nameMap.count != 10) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ procedureMap=FAIL count=%lu names=%@",
            label, (unsigned long)nameMap.count, nameMap]];
        return NO;
    }
    for (NSString *name in A12S2MExpectedProcedureNames()) {
        if (!A12S2KProcedureID(nameMap, name)) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ procedureMap=FAIL missing=%@ names=%@", label, name, nameMap]];
            return NO;
        }
    }
    return YES;
}

static inline BOOL A12S2MDispatchAllTen(
    id memoryModel,
    id loweredModel,
    NSDictionary *nameMap,
    NSMutableArray<NSString *> *lines,
    NSString *label,
    double *wallMSOut) {
    if (!A12S2MNameMapHasAllTen(nameMap, lines, label)) return NO;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        [lines addObject:[NSString stringWithFormat:@"%@ dispatchAll10=FAIL no-metal-device", label]];
        return NO;
    }

    NSTimeInterval allStart = NSDate.timeIntervalSinceReferenceDate;
    for (NSDictionary *trial in A12S2MProcedureTrials()) {
        @autoreleasepool {
            NSString *name = trial[@"name"];
            NSNumber *pid = A12S2KProcedureID(nameMap, name);
            NSUInteger inputChannels = [trial[@"in"] unsignedIntegerValue];
            NSUInteger outputChannels = [trial[@"out"] unsignedIntegerValue];
            NSUInteger spatial = [trial[@"spatial"] unsignedIntegerValue];
            NSError *surfaceError = nil;
            A12ANESurface *input = [[A12ANESurface alloc]
                initWithDevice:device channels:inputChannels spatial:spatial error:&surfaceError];
            A12ANESurface *output = [[A12ANESurface alloc]
                initWithDevice:device channels:outputChannels spatial:spatial error:&surfaceError];
            if (!pid || !input || !output) {
                [lines addObject:[NSString stringWithFormat:
                    @"%@ dispatchAll10=FAIL procedure=%@ pid=%@ surfaceError=%@",
                    label, name, pid, A12S2Error(surfaceError)]];
                return NO;
            }
            A12S2KFillDeterministic(input);
            A12S2KZero(output);
            if (!A12S2KEvaluateProcedure(
                    memoryModel, loweredModel, pid.unsignedIntegerValue,
                    input, output, name, lines)) {
                [lines addObject:[NSString stringWithFormat:
                    @"%@ dispatchAll10=FAIL procedure=%@", label, name]];
                return NO;
            }
        }
    }
    if (wallMSOut) {
        *wallMSOut = (NSDate.timeIntervalSinceReferenceDate - allStart) * 1000.0;
    }
    return YES;
}

static inline BOOL A12S2MReconstructedCacheHit(
    NSDictionary *plist,
    NSDictionary *weights,
    NSString *label,
    NSMutableArray<NSString *> *lines) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ cacheVerify=FAIL serialize=%@", label, A12S2Error(serializeError)]];
        return NO;
    }

    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weights, nil) : nil;
    id model = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!model) {
        [lines addObject:[NSString stringWithFormat:@"%@ cacheVerify=FAIL model=nil", label]];
        return NO;
    }

    BOOL hit = [model respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"compiledModelExists")) : NO;
    SEL setDescriptor = NSSelectorFromString(@"setDescriptor:");
    if ([model respondsToSelector:setDescriptor]) {
        ((void(*)(id,SEL,id))objc_msgSend)(model, setDescriptor, nil);
    }
    if (!hit) {
        [lines addObject:[NSString stringWithFormat:@"%@ cacheVerify=FAIL cacheHit=no", label]];
    }
    return hit;
}

static inline id A12S2MReloadHandle(
    id memoryModel,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    NSDictionary **nameMapOut,
    double *loadMSOut) {
    NSError *loadError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:
            @"%@ handleReload=FAIL load=%.2fms error=%@",
            label, loadMS, A12S2Error(loadError)]];
        return nil;
    }

    id lowered = [memoryModel respondsToSelector:NSSelectorFromString(@"model")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"model")) : nil;
    NSDictionary *desc = A12S2ANEFDescription(lowered ?: memoryModel);
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSUInteger count = MAX(nameMap.count, procedures.count);
    if (!lowered || count != 10 || !A12S2MNameMapHasAllTen(nameMap, lines, label)) {
        double unloadMS = 0.0;
        A12S2LUnloadModel(memoryModel, &unloadMS);
        [lines addObject:[NSString stringWithFormat:
            @"%@ handleReload=FAIL procedureCount=%lu lowered=%@ unloadAfterFailure=%.2fms",
            label, (unsigned long)count, lowered ? @"yes" : @"no", unloadMS]];
        return nil;
    }

    if (nameMapOut) *nameMapOut = nameMap;
    if (loadMSOut) *loadMSOut = loadMS;
    return lowered;
}

static inline BOOL A12S2MRunSelfO(
    id memoryModel,
    id loweredModel,
    NSDictionary *nameMap,
    A12ANESurface *input,
    A12ANESurface *output,
    NSString *label,
    NSMutableArray<NSString *> *detail,
    double *wallMSOut) {
    NSNumber *pid = A12S2KProcedureID(nameMap, @"procedure_self_o");
    if (!pid || !input || !output) return NO;
    A12S2KZero(output);
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = A12S2KEvaluateProcedure(
        memoryModel, loweredModel, pid.unsignedIntegerValue,
        input, output, label, detail);
    if (wallMSOut) {
        *wallMSOut = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    }
    return ok;
}

static inline NSString *A12ANEStage2MProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE multiprocedure cache-hit residency Stage 2M",
            @"goal=isolate pure resident-program ceiling from Stage2L compile/materialization pressure",
            @"phaseA=compile/load/dispatch10/unload each block; verify reconstructed cache hit; retain descriptor-cleared unloaded handle",
            @"phaseB=reload retained handles only; no donor lowering, weight map, materialization, or compile; sentinel newest+self_o(block0)",
            @"phaseC=post-cleanup warm reload/self_o/unload cycles for representative handles",
            nil];

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        NSURL *probeRoot = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2M-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];

        NSMutableArray<NSDictionary *> *handles = [NSMutableArray arrayWithCapacity:28];
        BOOL phaseAPass = YES;
        NSString *phaseAFailure = nil;

        // PHASE A: compile every block in isolation and retain only an unloaded,
        // descriptor-cleared model handle. No ANE model remains loaded between blocks.
        for (NSUInteger block = 0; block < 28; ++block) {
            @autoreleasepool {
                NSMutableArray<NSString *> *detail = [NSMutableArray array];
                NSArray<NSMutableDictionary *> *donors = A12S2LFindDonors(block, detail);
                if (donors.count != 8) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-missing-donors", (unsigned long)block];
                    A12S2LAppendTail(lines, detail, 12, @"phaseA detail: ");
                    break;
                }

                uint64_t donorBytes = A12S2LDonorWeightBytes(donors);
                NSURL *blockRoot = [probeRoot URLByAppendingPathComponent:
                    [NSString stringWithFormat:@"phaseA-b%02lu", (unsigned long)block]
                    isDirectory:YES];
                NSMutableDictionary *canonical = A12S2HBuildCanonicalCombined(
                    donors, blockRoot, runtimeVersion, detail);
                NSMutableDictionary *full10 = canonical ? A12S2JSplitQKV(canonical, detail) : nil;
                NSMutableDictionary *plist = full10 ? A12S2HSubset(
                    full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9],
                    detail, [NSString stringWithFormat:@"phaseA-b%02lu-full10", (unsigned long)block]) : nil;
                NSArray<NSDictionary *> *donorMap = donors ? A12S2JDonorMap(donors) : nil;
                NSMutableDictionary *weights = [NSMutableDictionary dictionary];
                BOOL normalized = plist && donorMap.count == 10 && A12S2JNormalizeWeightsDedup(
                    plist, donorMap, blockRoot, weights, detail);
                if (!normalized) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-container-build", (unsigned long)block];
                    A12S2LAppendTail(lines, detail, 16, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                uint64_t weightMapBytes = A12S2LWeightMapBytes(weights);
                id lowered = nil;
                NSDictionary *nameMap = nil;
                double compileMS = 0.0, loadMS = 0.0;
                BOOL initialCacheHit = NO;
                NSString *label = [NSString stringWithFormat:@"phaseA-b%02lu", (unsigned long)block];
                id memoryModel = A12S2LLoadTenProcedureModel(
                    plist, weights, label, detail,
                    &lowered, &nameMap, &compileMS, &loadMS, &initialCacheHit);
                if (!memoryModel) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-load", (unsigned long)block];
                    A12S2LAppendTail(lines, detail, 18, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                double dispatch10MS = 0.0;
                BOOL dispatch10 = A12S2MDispatchAllTen(
                    memoryModel, lowered, nameMap, detail, label, &dispatch10MS);
                double unloadMS = 0.0;
                BOOL unloaded = A12S2LUnloadModel(memoryModel, &unloadMS);
                BOOL reconstructedHit = unloaded && A12S2MReconstructedCacheHit(
                    plist, weights, label, detail);

                [lines addObject:[NSString stringWithFormat:
                    @"phaseA b%02lu donor=%.1fMB weightMap=%.1fMB initialCacheHit=%@ compile=%.1fms load=%.2fms dispatch10=%@ dispatch10Wall=%.2fms unload=%@ unloadMS=%.2f reconstructedCacheHit=%@",
                    (unsigned long)block,
                    (double)donorBytes / (1024.0 * 1024.0),
                    (double)weightMapBytes / (1024.0 * 1024.0),
                    initialCacheHit ? @"yes" : @"no", compileMS, loadMS,
                    dispatch10 ? @"PASS" : @"FAIL", dispatch10MS,
                    unloaded ? @"PASS" : @"FAIL", unloadMS,
                    reconstructedHit ? @"yes" : @"no"]];

                if (!dispatch10 || !unloaded || !reconstructedHit) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-verify", (unsigned long)block];
                    A12S2LAppendTail(lines, detail, 14, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                // A12S2LLoadTenProcedureModel already cleared the descriptor after
                // load. This object is now unloaded and should retain only enough
                // compiled-model identity to be reloadable without the weight map.
                [handles addObject:@{
                    @"block": @(block),
                    @"model": memoryModel,
                    @"donorBytes": @(donorBytes),
                }];
                [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
            }
        }

        if (!phaseAPass || handles.count != 28) {
            [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC phaseA failure=%@ handles=%lu expected=28",
                phaseAFailure ?: @"unknown", (unsigned long)handles.count]];
            return [lines componentsJoinedByString:@"\n"];
        }
        [lines addObject:@"phaseA=PASS precompiled=28 reconstructedCacheHits=28 unloadedHandles=28"];

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *surfaceError = nil;
        A12ANESurface *input = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        A12ANESurface *output = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        if (!input || !output) {
            [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL stage=phaseB-surface error=%@", A12S2Error(surfaceError)]];
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2KFillDeterministic(input);
        A12S2KZero(output);

        // Prove the key production mechanism before starting the residency sweep:
        // the descriptor-cleared unloaded object itself must be directly reloadable.
        {
            NSMutableArray<NSString *> *detail = [NSMutableArray array];
            NSDictionary *firstHandle = handles.firstObject;
            id model = firstHandle[@"model"];
            NSDictionary *nameMap = nil;
            double loadMS = 0.0;
            id lowered = A12S2MReloadHandle(
                model, @"handle-preflight-b00", detail, &nameMap, &loadMS);
            double evalMS = 0.0;
            BOOL evalOK = lowered && A12S2MRunSelfO(
                model, lowered, nameMap, input, output,
                @"handle-preflight-b00-self_o", detail, &evalMS);
            double unloadMS = 0.0;
            BOOL unloadOK = lowered && A12S2LUnloadModel(model, &unloadMS);
            if (!lowered || !evalOK || !unloadOK) {
                A12S2LAppendTail(lines, detail, 14, @"preflight detail: ");
                [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
                [lines addObject:[NSString stringWithFormat:
                    @"RESULT=DIAGNOSTIC stage=unloaded-handle-reload load=%.2f eval=%.2f unload=%.2f",
                    loadMS, evalMS, unloadMS]];
                return [lines componentsJoinedByString:@"\n"];
            }
            [lines addObject:[NSString stringWithFormat:
                @"handleReloadPreflight=PASS b00 load=%.2fms eval=%.2fms unload=%.2fms descriptorRebuild=no weightMap=no",
                loadMS, evalMS, unloadMS]];
        }

        // Give any asynchronous work from Phase A/preflight a chance to drain,
        // then install fresh observers. Phase B counters therefore cannot inherit
        // first-compile warnings.
        usleep(250000);

        __block volatile unsigned int uiWarnings = 0;
        __block volatile unsigned int dispatchWarnings = 0;
        __block volatile unsigned long dispatchFlags = 0;
        id warningToken = [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
            object:nil queue:nil usingBlock:^(__unused NSNotification *note) {
                uiWarnings += 1;
            }];
        dispatch_queue_t pressureQueue = dispatch_queue_create(
            "com.invisiblestrangler.AnimaXS.s2m-pressure", DISPATCH_QUEUE_SERIAL);
        dispatch_source_t pressureSource = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            pressureQueue);
        if (pressureSource) {
            dispatch_source_set_event_handler(pressureSource, ^{
                dispatchFlags = dispatch_source_get_data(pressureSource);
                dispatchWarnings += 1;
            });
            dispatch_resume(pressureSource);
        }

        // PHASE B: pure reload-only residency sweep.
        NSMutableArray<NSDictionary *> *resident = [NSMutableArray arrayWithCapacity:28];
        NSMutableArray<NSNumber *> *healthyLoads = [NSMutableArray array];
        NSMutableArray<NSNumber *> *healthyEvals = [NSMutableArray array];
        NSUInteger onsetBlock = NSNotFound;
        NSString *onsetReason = nil;
        BOOL onsetModelAdmitted = NO;
        uint64_t cumulativeDonorBytes = 0;

        for (NSUInteger block = 0; block < handles.count; ++block) {
            NSMutableArray<NSString *> *detail = [NSMutableArray array];
            NSDictionary *handle = handles[block];
            id memoryModel = handle[@"model"];
            unsigned int uiBefore = uiWarnings;
            unsigned int dispatchBefore = dispatchWarnings;

            NSDictionary *nameMap = nil;
            double loadMS = 0.0;
            id lowered = A12S2MReloadHandle(
                memoryModel,
                [NSString stringWithFormat:@"phaseB-b%02lu", (unsigned long)block],
                detail, &nameMap, &loadMS);
            if (!lowered) {
                onsetBlock = block;
                onsetReason = @"handle-reload-failure";
                onsetModelAdmitted = NO;
                A12S2LAppendTail(lines, detail, 14, @"phaseB detail: ");
                break;
            }

            [resident addObject:@{
                @"block": @(block),
                @"model": memoryModel,
                @"lowered": lowered,
                @"names": nameMap,
            }];
            cumulativeDonorBytes += [handle[@"donorBytes"] unsignedLongLongValue];

            double evalMS = 0.0;
            BOOL evalOK = A12S2MRunSelfO(
                memoryModel, lowered, nameMap, input, output,
                [NSString stringWithFormat:@"phaseB-b%02lu-self_o", (unsigned long)block],
                detail, &evalMS);

            BOOL sentinelOK = YES;
            double sentinelMS = 0.0;
            if (block > 0) {
                NSDictionary *first = resident.firstObject;
                sentinelOK = A12S2MRunSelfO(
                    first[@"model"], first[@"lowered"], first[@"names"],
                    input, output, @"phaseB-b00-sentinel", detail, &sentinelMS);
            }

            if (pressureSource) dispatch_sync(pressureQueue, ^{});
            unsigned int uiNow = uiWarnings;
            unsigned int dispatchNow = dispatchWarnings;
            unsigned long flagsNow = dispatchFlags;
            BOOL freshPressure = uiNow > uiBefore || dispatchNow > dispatchBefore;

            double baselineLoad = A12S2LMedian(healthyLoads);
            double baselineEval = A12S2LMedian(healthyEvals);
            double loadThreshold = MAX(500.0, baselineLoad > 0.0 ? baselineLoad * 8.0 : 500.0);
            double evalThreshold = MAX(100.0, baselineEval > 0.0 ? baselineEval * 10.0 : 100.0);
            BOOL loadPathology = healthyLoads.count >= 3 && loadMS > loadThreshold;
            BOOL evalPathology = healthyEvals.count >= 3 &&
                (evalMS > evalThreshold || (block > 0 && sentinelMS > evalThreshold));

            [lines addObject:[NSString stringWithFormat:
                @"phaseB b%02lu resident=%lu cumulativeDonor=%.1fMB reloadOnly=yes tempWeightMap=0 compile=0 load=%.2fms eval=%.2fms sentinel=%.2fms freshPressure=%@ pressure(ui=%u dispatch=%u flags=0x%lx) thresholds(load=%.1f eval=%.1f)",
                (unsigned long)block, (unsigned long)resident.count,
                (double)cumulativeDonorBytes / (1024.0 * 1024.0),
                loadMS, evalMS, sentinelMS,
                freshPressure ? @"yes" : @"no",
                uiNow, dispatchNow, flagsNow, loadThreshold, evalThreshold]];

            if (!evalOK || !sentinelOK || freshPressure || loadPathology || evalPathology) {
                onsetBlock = block;
                onsetModelAdmitted = YES;
                if (!evalOK) onsetReason = @"newest-eval-failure";
                else if (!sentinelOK) onsetReason = @"oldest-sentinel-failure";
                else if (freshPressure) onsetReason = @"fresh-memory-pressure";
                else if (loadPathology) onsetReason = @"pathological-load";
                else onsetReason = @"pathological-eval";
                A12S2LAppendTail(lines, detail, 10, @"phaseB detail: ");
                [lines addObject:[NSString stringWithFormat:
                    @"phaseB STOP-ONSET b%02lu reason=%@ residentNow=%lu safeBeforeOnset=%lu",
                    (unsigned long)block, onsetReason,
                    (unsigned long)resident.count,
                    (unsigned long)(resident.count > 0 ? resident.count - 1 : 0)]];
                break;
            }

            if (healthyLoads.count < 8) [healthyLoads addObject:@(loadMS)];
            if (healthyEvals.count < 8) [healthyEvals addObject:@(MAX(evalMS, sentinelMS))];
        }

        NSUInteger admitted = resident.count;
        NSUInteger lastSafe = onsetBlock == NSNotFound
            ? admitted
            : (onsetModelAdmitted ? (admitted > 0 ? admitted - 1 : 0) : admitted);
        [lines addObject:[NSString stringWithFormat:
            @"phaseB residencyResult admitted=%lu lastSafe=%lu onsetBlock=%@ reason=%@ cumulativeDonor=%.1fMB uiWarnings=%u dispatchWarnings=%u flags=0x%lx",
            (unsigned long)admitted, (unsigned long)lastSafe,
            onsetBlock == NSNotFound ? @"none" :
                [NSString stringWithFormat:@"%lu", (unsigned long)onsetBlock],
            onsetReason ?: @"none",
            (double)cumulativeDonorBytes / (1024.0 * 1024.0),
            (unsigned int)uiWarnings, (unsigned int)dispatchWarnings,
            (unsigned long)dispatchFlags]];

        double unloadTotal = 0.0;
        NSUInteger unloadFailures = 0;
        for (NSDictionary *record in resident.reverseObjectEnumerator) {
            double ms = 0.0;
            BOOL ok = A12S2LUnloadModel(record[@"model"], &ms);
            unloadTotal += ms;
            if (!ok) ++unloadFailures;
            [lines addObject:[NSString stringWithFormat:
                @"phaseB cleanup b%02lu unload=%@ ms=%.2f",
                (unsigned long)[record[@"block"] unsignedIntegerValue],
                ok ? @"PASS" : @"FAIL", ms]];
        }
        [lines addObject:[NSString stringWithFormat:
            @"phaseB cleanupSummary unloads=%lu failures=%lu total=%.2fms avg=%.2fms",
            (unsigned long)resident.count, (unsigned long)unloadFailures,
            unloadTotal, resident.count ? unloadTotal / (double)resident.count : 0.0]];

        if (warningToken) [NSNotificationCenter.defaultCenter removeObserver:warningToken];
        if (pressureSource) dispatch_source_cancel(pressureSource);

        // PHASE C: after loaded programs are gone, measure repeated warm scheduler
        // cost for early/middle/late blocks using the same descriptor-free handles.
        usleep(500000);
        BOOL phaseCPass = YES;
        NSArray<NSNumber *> *sampleBlocks = @[@0, @9, @27];
        for (NSNumber *rawBlock in sampleBlocks) {
            NSUInteger block = rawBlock.unsignedIntegerValue;
            if (block >= handles.count) continue;
            id memoryModel = handles[block][@"model"];
            for (NSUInteger cycle = 0; cycle < 3; ++cycle) {
                NSMutableArray<NSString *> *detail = [NSMutableArray array];
                NSDictionary *nameMap = nil;
                double loadMS = 0.0;
                id lowered = A12S2MReloadHandle(
                    memoryModel,
                    [NSString stringWithFormat:@"phaseC-b%02lu-c%lu",
                        (unsigned long)block, (unsigned long)cycle],
                    detail, &nameMap, &loadMS);
                double evalMS = 0.0;
                BOOL evalOK = lowered && A12S2MRunSelfO(
                    memoryModel, lowered, nameMap, input, output,
                    [NSString stringWithFormat:@"phaseC-b%02lu-c%lu-self_o",
                        (unsigned long)block, (unsigned long)cycle],
                    detail, &evalMS);
                double unloadMS = 0.0;
                BOOL unloadOK = lowered && A12S2LUnloadModel(memoryModel, &unloadMS);
                [lines addObject:[NSString stringWithFormat:
                    @"phaseC b%02lu cycle=%lu load=%.2fms eval=%.2fms unload=%.2fms result=%@",
                    (unsigned long)block, (unsigned long)cycle,
                    loadMS, evalMS, unloadMS,
                    (lowered && evalOK && unloadOK) ? @"PASS" : @"FAIL"]];
                if (!lowered || !evalOK || !unloadOK) {
                    phaseCPass = NO;
                    A12S2LAppendTail(lines, detail, 10, @"phaseC detail: ");
                    break;
                }
            }
            if (!phaseCPass) break;
        }

        [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];

        if (!phaseCPass) {
            [lines addObject:@"RESULT=FAIL stage=phaseC-warm-cycle"];
        } else if (unloadFailures > 0) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL stage=phaseB-cleanup unloadFailures=%lu",
                (unsigned long)unloadFailures]];
        } else if (onsetBlock == NSNotFound && admitted == 28) {
            [lines addObject:
                @"RESULT=PASS full28-reload-only-multiprocedure-blocks-resident phaseA=PASS phaseC=PASS"];
        } else if (onsetBlock != NSNotFound) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC clean-residency-boundary lastSafe=%lu onsetAt=%lu reason=%@ phaseA=PASS phaseC=PASS",
                (unsigned long)lastSafe, (unsigned long)onsetBlock, onsetReason ?: @"unknown"]];
        } else {
            [lines addObject:@"RESULT=FAIL stage=phaseB-incomplete"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
