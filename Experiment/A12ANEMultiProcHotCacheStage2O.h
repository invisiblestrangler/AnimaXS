#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>
#import <mach/mach.h>
#import <malloc/malloc.h>
#import <stdio.h>
#import <unistd.h>
#import "A12ANEMultiProcLoaderParityStage2N.h"

// Stage 2O measures how long the ~14 ms ANE runtime-hot reload state discovered
// by Stage 2N survives accesses to other real multiprocedure block identities.
//
// Stage 2N proved:
//   * compiledModelExists=yes alone still gives a ~150-160 ms first runtime load;
//   * after load -> real eval -> unload, the same compiled identity reloads in
//     ~13-14 ms, with either a new _ANEInMemoryModel object or the same object.
//
// The production-relevant 6-pinned + 2-streaming scheduler has 22 streamed
// blocks (6...27), so the reuse distance of any streamed block is 21 distinct
// streamed identities. Stage 2O therefore makes distance 20/21/22 explicit.
//
// Setup creates one lightweight, unloaded, reloadable handle for each real block.
// The 180.2 MB construction map exists only inside each block's autorelease pool.
// Measurement then keeps at most one ANE model loaded at a time:
//   warm b00 -> access N distinct b01...bNN -> reload b00.
// Each access includes a real procedure_self_o evaluation before unload.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2OProbe(void) {
    return @"ANE multiprocedure hot-cache reuse-distance Stage 2O\nRESULT=SKIP simulator";
}
#else

static inline uint64_t A12S2OFootprintBytes(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self_, TASK_VM_INFO, (task_info_t)&info, &count);
    return kr == KERN_SUCCESS ? (uint64_t)info.phys_footprint : 0;
}
static inline double A12S2OFootprintMB(void) {
    return (double)A12S2OFootprintBytes() / (1024.0 * 1024.0);
}
static inline void A12S2ORelief(void) {
    malloc_zone_pressure_relief(NULL, 0);
}

static inline NSString *A12S2ORootPath(void) {
    NSString *cache = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    return [(cache ?: NSTemporaryDirectory()) stringByAppendingPathComponent:@"AnimaXS-ANE"];
}
static inline NSString *A12S2OLivePath(void) {
    return [A12S2ORootPath() stringByAppendingPathComponent:@"stage2o-live-latest.log"];
}
static inline NSString *A12S2OFinalPath(void) {
    return [A12S2ORootPath() stringByAppendingPathComponent:@"stage2o-final-latest.log"];
}
static inline FILE **A12S2OFileSlot(void) { static FILE *f = NULL; return &f; }
static inline void A12S2OLive(NSString *line, BOOL durable) {
    if (!line) return;
    fprintf(stderr, "[ANE-S2O] %s\n", line.UTF8String ?: "");
    fflush(stderr);
    FILE *f = *A12S2OFileSlot();
    if (f) {
        fprintf(f, "%s\n", line.UTF8String ?: "");
        fflush(f);
        if (durable) fsync(fileno(f));
    }
}
static inline void A12S2OReport(NSMutableArray<NSString *> *report, NSString *line) {
    if (line) [report addObject:line];
    A12S2OLive(line, YES);
}
static inline void A12S2OBegin(void) {
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:A12S2ORootPath() withIntermediateDirectories:YES attributes:nil error:nil];
    FILE **slot = A12S2OFileSlot();
    if (*slot) { fclose(*slot); *slot = NULL; }
    *slot = fopen(A12S2OLivePath().fileSystemRepresentation, "w");
    A12S2OLive([NSString stringWithFormat:@"LIVE BEGIN footprint=%.1fMB live=%@ final=%@",
        A12S2OFootprintMB(), A12S2OLivePath(), A12S2OFinalPath()], YES);
}
static inline NSString *A12S2OFinish(NSMutableArray<NSString *> *report) {
    NSString *text = [report componentsJoinedByString:@"\n"];
    [text writeToFile:A12S2OFinalPath() atomically:YES encoding:NSUTF8StringEncoding error:nil];
    A12S2OLive([NSString stringWithFormat:@"LIVE COMPLETE footprint=%.1fMB final=%@",
        A12S2OFootprintMB(), A12S2OFinalPath()], YES);
    FILE **slot = A12S2OFileSlot();
    if (*slot) { fclose(*slot); *slot = NULL; }
    return text;
}

static inline NSString *A12S2OTrimName(NSUInteger mode) {
    switch (mode) {
        case 3: return @"model+attributes+program";
        case 2: return @"model+attributes";
        case 1: return @"model";
        default: return @"descriptor-only";
    }
}
static inline BOOL A12S2ONilSetter(id obj, NSString *selector, BOOL required) {
    SEL sel = NSSelectorFromString(selector);
    if (![obj respondsToSelector:sel]) return !required;
    ((void(*)(id,SEL,id))objc_msgSend)(obj, sel, nil);
    return YES;
}
static inline BOOL A12S2OTrim(id model, NSUInteger mode) {
    if (!model) return NO;
    BOOL ok = A12S2ONilSetter(model, @"setDescriptor:", NO);
    if (mode >= 1) ok = A12S2ONilSetter(model, @"setModel:", YES) && ok;
    if (mode >= 2) ok = A12S2ONilSetter(model, @"setModelAttributes:", NO) && ok;
    if (mode >= 3) ok = A12S2ONilSetter(model, @"setProgram:", NO) && ok;
    A12S2ORelief();
    return ok;
}

static inline id A12S2OLoadQuiet(
    id model,
    NSString *label,
    NSMutableArray<NSString *> *report,
    NSDictionary **nameMapOut,
    double *loadMSOut) {
    if (!model) return nil;
    A12S2OLive([NSString stringWithFormat:@"%@ LOAD START", label], NO);
    dispatch_semaphore_t watchdog = dispatch_semaphore_create(0);
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)),
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            if (dispatch_semaphore_wait(watchdog, DISPATCH_TIME_NOW) != 0) {
                A12S2OLive([NSString stringWithFormat:@"%@ LOAD STALL >2000ms", label], YES);
            }
        });

    NSError *error = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        model, NSSelectorFromString(@"loadWithQoS:options:error:"), 25u, @{}, &error);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    dispatch_semaphore_signal(watchdog);
    if (loadMSOut) *loadMSOut = ms;
    if (!loaded) {
        A12S2OReport(report, [NSString stringWithFormat:@"%@ LOAD FAIL ms=%.2f error=%@",
            label, ms, A12S2Error(error)]);
        return nil;
    }

    id lowered = [model respondsToSelector:NSSelectorFromString(@"model")]
        ? ((id(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"model")) : nil;
    NSDictionary *desc = A12S2ANEFDescription(lowered ?: model);
    NSDictionary *names = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSUInteger count = MAX(names.count, procedures.count);
    if (!lowered || count != 10) {
        double unloadMS = 0.0;
        A12S2LUnloadModel(model, &unloadMS);
        A12S2OReport(report, [NSString stringWithFormat:
            @"%@ LOAD INVALID ms=%.2f lowered=%@ procedures=%lu unload=%.2f",
            label, ms, lowered ? @"yes" : @"no", (unsigned long)count, unloadMS]);
        return nil;
    }
    if (nameMapOut) *nameMapOut = names;
    return lowered;
}

static inline BOOL A12S2OEvalSelfO(
    id model,
    id lowered,
    NSDictionary *names,
    A12ANESurface *input,
    A12ANESurface *output,
    NSString *label,
    double *evalMSOut) {
    NSNumber *pid = A12S2KProcedureID(names, @"procedure_self_o");
    if (!pid || !input || !output) return NO;
    A12S2KZero(output);
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = A12S2KEvaluateProcedure(
        model, lowered, pid.unsignedIntegerValue, input, output, label, detail);
    if (evalMSOut) *evalMSOut = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    return ok;
}

static inline BOOL A12S2OAccessHandle(
    NSDictionary *entry,
    NSString *label,
    NSUInteger trimMode,
    A12ANESurface *input,
    A12ANESurface *output,
    NSMutableArray<NSString *> *report,
    double *loadMSOut,
    double *evalMSOut,
    double *unloadMSOut) {
    id model = entry[@"model"];
    NSDictionary *names = nil;
    double load = 0.0, eval = 0.0, unload = 0.0;
    id lowered = A12S2OLoadQuiet(model, label, report, &names, &load);
    if (!lowered) return NO;
    BOOL evalOK = A12S2OEvalSelfO(model, lowered, names, input, output, label, &eval);
    BOOL unloadOK = A12S2LUnloadModel(model, &unload);
    BOOL trimOK = unloadOK && A12S2OTrim(model, trimMode);
    if (loadMSOut) *loadMSOut = load;
    if (evalMSOut) *evalMSOut = eval;
    if (unloadMSOut) *unloadMSOut = unload;
    if (!evalOK || !unloadOK || !trimOK) {
        A12S2OReport(report, [NSString stringWithFormat:
            @"%@ ACCESS FAIL load=%.2f eval=%.2f unload=%.2f evalOK=%@ unloadOK=%@ trimOK=%@",
            label, load, eval, unload, evalOK ? @"yes" : @"no",
            unloadOK ? @"yes" : @"no", trimOK ? @"yes" : @"no"]);
        return NO;
    }
    return YES;
}

static inline BOOL A12S2OTrimPreflight(
    id model,
    A12ANESurface *input,
    A12ANESurface *output,
    NSMutableArray<NSString *> *report,
    NSUInteger *modeOut) {
    // The caller supplies a currently loaded model. Try aggressive -> conservative.
    for (NSInteger mode = 3; mode >= 0; --mode) {
        double unload1 = 0.0;
        if (!A12S2LUnloadModel(model, &unload1)) continue;
        if (!A12S2OTrim(model, (NSUInteger)mode)) continue;

        NSDictionary *names = nil;
        double reload = 0.0, eval = 0.0, unload2 = 0.0;
        NSString *label = [NSString stringWithFormat:@"trim-preflight-m%ld", (long)mode];
        id lowered = A12S2OLoadQuiet(model, label, report, &names, &reload);
        BOOL evalOK = lowered && A12S2OEvalSelfO(model, lowered, names, input, output, label, &eval);
        BOOL unloadOK = lowered && A12S2LUnloadModel(model, &unload2);
        BOOL trimOK = unloadOK && A12S2OTrim(model, (NSUInteger)mode);
        A12S2OReport(report, [NSString stringWithFormat:
            @"TRIM PREFLIGHT mode=%@ reload=%.2f eval=%.2f unload=%.2f result=%@",
            A12S2OTrimName((NSUInteger)mode), reload, eval, unload2,
            (evalOK && unloadOK && trimOK) ? @"PASS" : @"FAIL"]);
        if (evalOK && unloadOK && trimOK) {
            if (modeOut) *modeOut = (NSUInteger)mode;
            return YES;
        }

        // A failed aggressive trim may leave this object unusable. Do not attempt
        // a weaker mode on the damaged object; caller will treat this as failure.
        break;
    }
    return NO;
}

static inline NSString *A12S2OClassify(double ms) {
    if (ms > 0.0 && ms < 50.0) return @"HOT";
    if (ms >= 80.0) return @"COLD";
    return @"GRAY";
}

static inline NSString *A12ANEStage2OProbe(void) {
    @autoreleasepool {
        A12S2OBegin();
        NSMutableArray<NSString *> *report = [NSMutableArray array];
        A12S2OReport(report, @"ANE multiprocedure hot-cache reuse-distance Stage 2O");
        A12S2OReport(report, @"reference=Stage2N first runtime load ~146-161ms; immediate reload ~13-14ms");
        A12S2OReport(report, @"question=does runtime-hot state survive 21 distinct intervening streamed model identities?");
        A12S2OReport(report, @"policy=one loaded model max; every access is load+self_o-eval+unload; target=b00; interveners=b01...b27");
        A12S2OReport(report, @"classification=HOT <50ms; COLD >=80ms; GRAY otherwise; thresholds diagnostic only");

        __block volatile unsigned int uiWarnings = 0;
        __block volatile unsigned int dispatchWarnings = 0;
        __block volatile unsigned long dispatchFlags = 0;
        id warningToken = [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
            object:nil queue:nil usingBlock:^(__unused NSNotification *note) {
                ++uiWarnings;
                A12S2OLive([NSString stringWithFormat:@"MEMORY WARNING ui=%u footprint=%.1fMB",
                    uiWarnings, A12S2OFootprintMB()], YES);
            }];
        dispatch_queue_t pressureQueue = dispatch_queue_create(
            "com.invisiblestrangler.AnimaXS.s2o-pressure", DISPATCH_QUEUE_SERIAL);
        dispatch_source_t pressureSource = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            pressureQueue);
        if (pressureSource) {
            dispatch_source_set_event_handler(pressureSource, ^{
                dispatchFlags = dispatch_source_get_data(pressureSource);
                ++dispatchWarnings;
                A12S2OLive([NSString stringWithFormat:
                    @"MEMORY PRESSURE dispatch=%u flags=0x%lx footprint=%.1fMB",
                    dispatchWarnings, dispatchFlags, A12S2OFootprintMB()], YES);
            });
            dispatch_resume(pressureSource);
        }
        void (^cleanupObservers)(void) = ^{
            if (warningToken) [NSNotificationCenter.defaultCenter removeObserver:warningToken];
            if (pressureSource) dispatch_source_cancel(pressureSource);
        };

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *surfaceError = nil;
        A12ANESurface *input = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        A12ANESurface *output = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        if (!input || !output) {
            A12S2OReport(report, [NSString stringWithFormat:@"RESULT=FAIL stage=surfaces error=%@", A12S2Error(surfaceError)]);
            cleanupObservers();
            return A12S2OFinish(report);
        }
        A12S2KFillDeterministic(input);

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(report);
        NSURL *probeRoot = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:@"AnimaXS-ANE-S2O-%@", NSUUID.UUID.UUIDString]]
            isDirectory:YES];
        NSMutableArray<NSDictionary *> *handles = [NSMutableArray arrayWithCapacity:28];
        NSUInteger trimMode = NSNotFound;
        BOOL setupOK = YES;

        A12S2OReport(report, [NSString stringWithFormat:@"SETUP BEGIN runtimeIR=%@ footprint=%.1fMB", runtimeVersion ?: @"nil", A12S2OFootprintMB()]);
        for (NSUInteger block = 0; block < 28 && setupOK; ++block) {
            @autoreleasepool {
                NSString *label = [NSString stringWithFormat:@"setup-b%02lu", (unsigned long)block];
                NSURL *blockRoot = [probeRoot URLByAppendingPathComponent:label isDirectory:YES];
                NSMutableArray<NSString *> *detail = [NSMutableArray array];
                NSMutableDictionary *plist = nil, *weights = nil;
                A12S2OLive([NSString stringWithFormat:@"SETUP b%02lu BUILD START handles=%lu footprint=%.1fMB",
                    (unsigned long)block, (unsigned long)handles.count, A12S2OFootprintMB()], NO);
                if (!A12S2NBuildBlock(block, blockRoot, runtimeVersion, detail, &plist, &weights)) {
                    A12S2OReport(report, [NSString stringWithFormat:@"SETUP b%02lu FAIL stage=build", (unsigned long)block]);
                    setupOK = NO;
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    continue;
                }

                id lowered = nil;
                NSDictionary *names = nil;
                double compileMS = 0.0, loadMS = 0.0;
                BOOL cacheHit = NO;
                id model = A12S2LLoadTenProcedureModel(
                    plist, weights, label, detail,
                    &lowered, &names, &compileMS, &loadMS, &cacheHit);
                if (!model || !lowered || names.count != 10) {
                    A12S2OReport(report, [NSString stringWithFormat:
                        @"SETUP b%02lu FAIL stage=load cacheHit=%@ compile=%.2f load=%.2f",
                        (unsigned long)block, cacheHit ? @"yes" : @"no", compileMS, loadMS]);
                    setupOK = NO;
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    continue;
                }

                double evalMS = 0.0;
                BOOL evalOK = A12S2OEvalSelfO(model, lowered, names, input, output, label, &evalMS);
                BOOL ready = NO;
                double unloadMS = 0.0;
                if (block == 0) {
                    // Preflight starts from the loaded/evaluated b00 handle and leaves it
                    // unloaded+trimmed in the selected reusable state.
                    ready = evalOK && A12S2OTrimPreflight(model, input, output, report, &trimMode);
                } else {
                    BOOL unloadOK = evalOK && A12S2LUnloadModel(model, &unloadMS);
                    ready = unloadOK && trimMode != NSNotFound && A12S2OTrim(model, trimMode);
                }

                if (ready) {
                    [handles addObject:@{@"block": @(block), @"model": model}];
                } else {
                    A12S2OReport(report, [NSString stringWithFormat:
                        @"SETUP b%02lu FAIL stage=eval-unload-trim eval=%@ trimMode=%@",
                        (unsigned long)block, evalOK ? @"PASS" : @"FAIL",
                        trimMode == NSNotFound ? @"none" : A12S2OTrimName(trimMode)]);
                    setupOK = NO;
                }
                [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                A12S2OLive([NSString stringWithFormat:
                    @"SETUP b%02lu %@ cacheHit=%@ compile=%.2f load=%.2f eval=%.2f handles=%lu trim=%@",
                    (unsigned long)block, ready ? @"PASS" : @"FAIL", cacheHit ? @"yes" : @"no",
                    compileMS, loadMS, evalMS, (unsigned long)handles.count,
                    trimMode == NSNotFound ? @"none" : A12S2OTrimName(trimMode)], YES);
            }
            A12S2ORelief();
            if (uiWarnings || dispatchWarnings) {
                A12S2OReport(report, [NSString stringWithFormat:
                    @"SETUP PRESSURE after=b%02lu ui=%u dispatch=%u flags=0x%lx footprint=%.1fMB",
                    (unsigned long)block, uiWarnings, dispatchWarnings, dispatchFlags, A12S2OFootprintMB()]);
                setupOK = NO;
            }
        }

        if (!setupOK || handles.count != 28 || trimMode == NSNotFound) {
            [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
            cleanupObservers();
            A12S2OReport(report, [NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC stage=setup handles=%lu trim=%@ footprint=%.1fMB",
                (unsigned long)handles.count,
                trimMode == NSNotFound ? @"none" : A12S2OTrimName(trimMode),
                A12S2OFootprintMB()]);
            return A12S2OFinish(report);
        }
        A12S2OReport(report, [NSString stringWithFormat:
            @"SETUP PASS handles=28 trim=%@ footprint=%.1fMB warnings=%u/%u",
            A12S2OTrimName(trimMode), A12S2OFootprintMB(), uiWarnings, dispatchWarnings]);

        // All setup loads are complete and all handles are unloaded. Fresh pressure
        // counters below therefore describe the reuse-distance measurement itself.
        unsigned int measurementUIBase = uiWarnings;
        unsigned int measurementDispatchBase = dispatchWarnings;

        NSArray<NSNumber *> *distances = @[@0,@1,@2,@4,@6,@8,@10,@12,@14,@16,@18,@20,@21,@22,@24,@27];
        NSMutableArray<NSDictionary *> *results = [NSMutableArray arrayWithCapacity:distances.count];
        BOOL measurementOK = YES;
        A12S2OReport(report, [NSString stringWithFormat:
            @"MEASURE BEGIN distances=%@ key6p2ReuseDistance=21 footprint=%.1fMB",
            distances, A12S2OFootprintMB()]);

        for (NSNumber *distanceNumber in distances) {
            NSUInteger distance = distanceNumber.unsignedIntegerValue;
            NSString *trial = [NSString stringWithFormat:@"D%02lu", (unsigned long)distance];
            A12S2OReport(report, [NSString stringWithFormat:@"%@ START", trial]);

            // Re-establish b00 as most-recently used immediately before the N
            // intervening identities. The warm-up cost itself is recorded because it
            // tells us whether the previous trial had evicted b00.
            double warmLoad=0.0, warmEval=0.0, warmUnload=0.0;
            if (!A12S2OAccessHandle(handles[0], [trial stringByAppendingString:@"-warm-b00"], trimMode,
                input, output, report, &warmLoad, &warmEval, &warmUnload)) {
                measurementOK = NO;
                break;
            }

            double interposerLoadSum = 0.0;
            double interposerLoadMin = DBL_MAX;
            double interposerLoadMax = 0.0;
            NSUInteger interposerHot = 0, interposerCold = 0, interposerGray = 0;
            for (NSUInteger i = 1; i <= distance; ++i) {
                double load=0.0, eval=0.0, unload=0.0;
                NSString *label = [NSString stringWithFormat:@"%@-i%02lu-b%02lu", trial,
                    (unsigned long)i, (unsigned long)i];
                if (!A12S2OAccessHandle(handles[i], label, trimMode, input, output, report,
                    &load, &eval, &unload)) {
                    measurementOK = NO;
                    break;
                }
                interposerLoadSum += load;
                interposerLoadMin = MIN(interposerLoadMin, load);
                interposerLoadMax = MAX(interposerLoadMax, load);
                NSString *cls = A12S2OClassify(load);
                if ([cls isEqualToString:@"HOT"]) ++interposerHot;
                else if ([cls isEqualToString:@"COLD"]) ++interposerCold;
                else ++interposerGray;
            }
            if (!measurementOK) break;

            double targetLoad=0.0, targetEval=0.0, targetUnload=0.0;
            if (!A12S2OAccessHandle(handles[0], [trial stringByAppendingString:@"-target-b00"], trimMode,
                input, output, report, &targetLoad, &targetEval, &targetUnload)) {
                measurementOK = NO;
                break;
            }
            NSString *classification = A12S2OClassify(targetLoad);
            double interposerAvg = distance ? interposerLoadSum / (double)distance : 0.0;
            if (distance == 0) interposerLoadMin = 0.0;
            NSDictionary *result = @{
                @"distance": @(distance),
                @"targetLoad": @(targetLoad),
                @"classification": classification,
                @"warmLoad": @(warmLoad),
            };
            [results addObject:result];
            A12S2OReport(report, [NSString stringWithFormat:
                @"%@ RESULT distance=%lu targetLoad=%.2f targetEval=%.2f class=%@ warmLoad=%.2f interposerAvg=%.2f min=%.2f max=%.2f hot/cold/gray=%lu/%lu/%lu footprint=%.1fMB",
                trial, (unsigned long)distance, targetLoad, targetEval, classification, warmLoad,
                interposerAvg, interposerLoadMin, interposerLoadMax,
                (unsigned long)interposerHot, (unsigned long)interposerCold, (unsigned long)interposerGray,
                A12S2OFootprintMB()]);

            if (pressureSource) dispatch_sync(pressureQueue, ^{});
            if (uiWarnings > measurementUIBase || dispatchWarnings > measurementDispatchBase) {
                A12S2OReport(report, [NSString stringWithFormat:
                    @"%@ PRESSURE uiDelta=%u dispatchDelta=%u flags=0x%lx",
                    trial, uiWarnings-measurementUIBase,
                    dispatchWarnings-measurementDispatchBase, dispatchFlags]);
                measurementOK = NO;
                break;
            }
        }

        NSInteger lastHot = -1;
        NSInteger firstCold = -1;
        double distance21MS = 0.0;
        NSString *distance21Class = @"MISSING";
        NSMutableString *compact = [NSMutableString string];
        for (NSDictionary *r in results) {
            NSInteger d = [r[@"distance"] integerValue];
            double ms = [r[@"targetLoad"] doubleValue];
            NSString *cls = r[@"classification"];
            if ([cls isEqualToString:@"HOT"]) lastHot = MAX(lastHot, d);
            if ([cls isEqualToString:@"COLD"] && firstCold < 0) firstCold = d;
            if (d == 21) { distance21MS = ms; distance21Class = cls; }
            if (compact.length) [compact appendString:@" "];
            [compact appendFormat:@"d%ld=%.1f/%@", (long)d, ms, cls];
        }
        A12S2OReport(report, [NSString stringWithFormat:@"MATRIX %@", compact]);
        A12S2OReport(report, [NSString stringWithFormat:
            @"BOUNDARY lastHotDistance=%@ firstColdDistance=%@ tested=%lu",
            lastHot >= 0 ? [NSString stringWithFormat:@"%ld", (long)lastHot] : @"none",
            firstCold >= 0 ? [NSString stringWithFormat:@"%ld", (long)firstCold] : @"none",
            (unsigned long)results.count]);
        A12S2OReport(report, [NSString stringWithFormat:
            @"SCHEDULER_6P2 reuseDistance=21 targetLoad=%.2f class=%@ implication=%@",
            distance21MS, distance21Class,
            [distance21Class isEqualToString:@"HOT"] ? @"hot-cache-survives-streaming-cycle" :
            ([distance21Class isEqualToString:@"COLD"] ? @"hot-cache-does-not-survive-streaming-cycle" : @"inconclusive")]);

        [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
        cleanupObservers();
        A12S2ORelief();
        if (!measurementOK) {
            A12S2OReport(report, [NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC stage=measurement completed=%lu footprint=%.1fMB",
                (unsigned long)results.count, A12S2OFootprintMB()]);
        } else if (results.count == distances.count) {
            A12S2OReport(report, @"RESULT=PASS hot-cache-reuse-distance-matrix-complete");
        } else {
            A12S2OReport(report, [NSString stringWithFormat:@"RESULT=DIAGNOSTIC incomplete=%lu/%lu",
                (unsigned long)results.count, (unsigned long)distances.count]);
        }
        return A12S2OFinish(report);
    }
}

#endif
