#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>
#import <os/log.h>
#import <stdio.h>
#import <unistd.h>
#define A12ANEStage2MProbe A12ANEStage2MProbeOriginal
#import "A12ANEMultiProcResidencyStage2M.h"
#undef A12ANEStage2MProbe

// Live-instrumented Stage 2M wrapper. The underlying experiment semantics stay
// the same, but important checkpoints are emitted immediately to stderr/Xcode,
// unified logging, and a synchronously-flushed cache file. If jetsam kills the
// app, the next launch prints the prior incomplete live log before starting.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2MProbe(void) {
    return @"ANE multiprocedure cache-hit residency Stage 2M live\nRESULT=SKIP simulator";
}
#else

static inline NSString *A12S2MLiveDirectory(void) {
    NSString *cache = [NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    if (!cache) return NSTemporaryDirectory();
    return [cache stringByAppendingPathComponent:@"AnimaXS-ANE"];
}

static inline NSString *A12S2MLiveLatestPath(void) {
    return [A12S2MLiveDirectory() stringByAppendingPathComponent:@"stage2m-live-latest.log"];
}

static inline NSString *A12S2MLivePreviousPath(void) {
    return [A12S2MLiveDirectory() stringByAppendingPathComponent:@"stage2m-live-previous.log"];
}

static inline NSString *A12S2MLiveFinalPath(void) {
    return [A12S2MLiveDirectory() stringByAppendingPathComponent:@"stage2m-final-latest.log"];
}

static inline void A12S2MLiveConsoleOnly(NSString *line) {
    if (![line isKindOfClass:NSString.class]) return;
    const char *utf8 = line.UTF8String ?: "";
    fprintf(stderr, "[ANE-S2M-LIVE] %s\n", utf8);
    fflush(stderr);
    os_log_with_type(OS_LOG_DEFAULT, OS_LOG_TYPE_DEFAULT,
                     "[ANE-S2M-LIVE] %{public}s", utf8);
}

static inline void A12S2MLiveAppendDisk(NSString *line) {
    if (![line isKindOfClass:NSString.class]) return;
    @synchronized (NSFileManager.defaultManager) {
        NSString *directory = A12S2MLiveDirectory();
        [NSFileManager.defaultManager createDirectoryAtPath:directory
            withIntermediateDirectories:YES attributes:nil error:nil];
        NSString *path = A12S2MLiveLatestPath();
        if (![NSFileManager.defaultManager fileExistsAtPath:path]) {
            [NSFileManager.defaultManager createFileAtPath:path contents:nil attributes:nil];
        }
        NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:path];
        if (!handle) return;
        @try {
            [handle seekToEndOfFile];
            NSData *data = [[line stringByAppendingString:@"\n"]
                dataUsingEncoding:NSUTF8StringEncoding];
            if (data) [handle writeData:data];
            [handle synchronizeFile];
        } @catch (__unused NSException *exception) {
        }
        [handle closeFile];
    }
}

static inline void A12S2MLiveRaw(NSString *line) {
    A12S2MLiveConsoleOnly(line);
    A12S2MLiveAppendDisk(line);
}

static inline void A12S2MLiveEmit(
    NSMutableArray<NSString *> *lines,
    NSString *line) {
    if (line) [lines addObject:line];
    A12S2MLiveRaw(line ?: @"(nil)");
}

static inline void A12S2MLiveAppendTail(
    NSMutableArray<NSString *> *lines,
    NSArray<NSString *> *detail,
    NSUInteger maxLines,
    NSString *prefix) {
    NSUInteger start = detail.count > maxLines ? detail.count - maxLines : 0;
    for (NSUInteger i = start; i < detail.count; ++i) {
        A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@%@",
            prefix ?: @"", detail[i]]);
    }
}

static inline void A12S2MLiveBegin(void) {
    NSString *directory = A12S2MLiveDirectory();
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:directory withIntermediateDirectories:YES
                   attributes:nil error:nil];

    NSString *latest = A12S2MLiveLatestPath();
    NSData *oldData = [NSData dataWithContentsOfFile:latest];
    if (oldData.length > 0) {
        NSString *old = [[NSString alloc] initWithData:oldData encoding:NSUTF8StringEncoding] ?: @"";
        [oldData writeToFile:A12S2MLivePreviousPath() options:NSDataWritingAtomic error:nil];
        if (![old containsString:@"LIVE SESSION COMPLETE"]) {
            A12S2MLiveConsoleOnly(@"RECOVERED PREVIOUS INCOMPLETE LOG BEGIN");
            const char *bytes = old.UTF8String ?: "";
            fprintf(stderr, "%s", bytes);
            if (![old hasSuffix:@"\n"]) fprintf(stderr, "\n");
            fflush(stderr);
            A12S2MLiveConsoleOnly(@"RECOVERED PREVIOUS INCOMPLETE LOG END");
        }
    }
    [[NSData data] writeToFile:latest options:NSDataWritingAtomic error:nil];
    A12S2MLiveRaw([NSString stringWithFormat:
        @"LIVE SESSION BEGIN timestamp=%.3f latest=%@ previous=%@ final=%@",
        NSDate.timeIntervalSinceReferenceDate,
        A12S2MLiveLatestPath(), A12S2MLivePreviousPath(), A12S2MLiveFinalPath()]);
}

static inline NSString *A12S2MLiveFinalize(NSMutableArray<NSString *> *lines) {
    NSString *report = [lines componentsJoinedByString:@"\n"];
    NSData *data = [report dataUsingEncoding:NSUTF8StringEncoding];
    NSError *error = nil;
    BOOL saved = data && [data writeToFile:A12S2MLiveFinalPath()
                                    options:NSDataWritingAtomic error:&error];
    A12S2MLiveRaw([NSString stringWithFormat:
        @"LIVE SESSION COMPLETE finalSaved=%@ finalPath=%@ lines=%lu error=%@",
        saved ? @"yes" : @"no", A12S2MLiveFinalPath(),
        (unsigned long)lines.count, A12S2Error(error)]);
    return report;
}

static inline BOOL A12S2MLiveDispatchAllTen(
    id memoryModel,
    id loweredModel,
    NSDictionary *nameMap,
    NSMutableArray<NSString *> *detail,
    NSMutableArray<NSString *> *lines,
    NSString *label,
    double *wallMSOut) {
    if (!A12S2MNameMapHasAllTen(nameMap, detail, label)) {
        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"%@ dispatch10 PRECHECK=FAIL", label]);
        return NO;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"%@ dispatch10 FAIL no-metal-device", label]);
        return NO;
    }

    NSTimeInterval allStart = NSDate.timeIntervalSinceReferenceDate;
    NSUInteger ordinal = 0;
    for (NSDictionary *trial in A12S2MProcedureTrials()) {
        @autoreleasepool {
            NSString *name = trial[@"name"];
            NSNumber *pid = A12S2KProcedureID(nameMap, name);
            NSUInteger inputChannels = [trial[@"in"] unsignedIntegerValue];
            NSUInteger outputChannels = [trial[@"out"] unsignedIntegerValue];
            NSUInteger spatial = [trial[@"spatial"] unsignedIntegerValue];
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"%@ dispatch10 procedure=%lu/10 name=%@ START",
                label, (unsigned long)(ordinal + 1), name]);

            NSError *surfaceError = nil;
            A12ANESurface *input = [[A12ANESurface alloc]
                initWithDevice:device channels:inputChannels spatial:spatial error:&surfaceError];
            A12ANESurface *output = [[A12ANESurface alloc]
                initWithDevice:device channels:outputChannels spatial:spatial error:&surfaceError];
            if (!pid || !input || !output) {
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ dispatch10 procedure=%@ SETUP=FAIL pid=%@ error=%@",
                    label, name, pid, A12S2Error(surfaceError)]);
                return NO;
            }
            A12S2KFillDeterministic(input);
            A12S2KZero(output);
            NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
            BOOL ok = A12S2KEvaluateProcedure(
                memoryModel, loweredModel, pid.unsignedIntegerValue,
                input, output, name, detail);
            double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"%@ dispatch10 procedure=%lu/10 name=%@ result=%@ wall=%.2fms",
                label, (unsigned long)(ordinal + 1), name,
                ok ? @"PASS" : @"FAIL", ms]);
            if (!ok) return NO;
            ++ordinal;
        }
    }
    if (wallMSOut) {
        *wallMSOut = (NSDate.timeIntervalSinceReferenceDate - allStart) * 1000.0;
    }
    return YES;
}

static inline NSString *A12ANEStage2MProbe(void) {
    @autoreleasepool {
        A12S2MLiveBegin();
        NSMutableArray<NSString *> *lines = [NSMutableArray array];
        A12S2MLiveEmit(lines, @"ANE multiprocedure cache-hit residency Stage 2M LIVE");
        A12S2MLiveEmit(lines, @"goal=isolate pure resident-program ceiling from Stage2L compile/materialization pressure");
        A12S2MLiveEmit(lines, @"logging=stderr-flush + unified-log + stage2m-live-latest.log; final=stage2m-final-latest.log");
        A12S2MLiveEmit(lines, @"phaseA=compile/load/dispatch10/unload each block; verify reconstructed cache hit; retain descriptor-cleared unloaded handle");
        A12S2MLiveEmit(lines, @"phaseB=reload retained handles only; no donor lowering, weight map, materialization, or compile; sentinel newest+self_o(block0)");
        A12S2MLiveEmit(lines, @"phaseC=post-cleanup warm reload/self_o/unload cycles for representative handles");

        __block volatile unsigned int liveUIWarnings = 0;
        __block volatile unsigned int liveDispatchWarnings = 0;
        id liveWarningToken = [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
            object:nil queue:nil usingBlock:^(__unused NSNotification *note) {
                ++liveUIWarnings;
                A12S2MLiveRaw([NSString stringWithFormat:
                    @"SYSTEM MEMORY WARNING uiCount=%u", (unsigned int)liveUIWarnings]);
            }];
        dispatch_queue_t livePressureQueue = dispatch_queue_create(
            "com.invisiblestrangler.AnimaXS.s2m-live-pressure", DISPATCH_QUEUE_SERIAL);
        dispatch_source_t livePressureSource = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            livePressureQueue);
        if (livePressureSource) {
            dispatch_source_set_event_handler(livePressureSource, ^{
                unsigned long flags = dispatch_source_get_data(livePressureSource);
                ++liveDispatchWarnings;
                A12S2MLiveRaw([NSString stringWithFormat:
                    @"SYSTEM DISPATCH MEMORY PRESSURE count=%u flags=0x%lx",
                    (unsigned int)liveDispatchWarnings, flags]);
            });
            dispatch_resume(livePressureSource);
        }

        void (^cleanupLiveObservers)(void) = ^{
            if (liveWarningToken) {
                [NSNotificationCenter.defaultCenter removeObserver:liveWarningToken];
            }
            if (livePressureSource) dispatch_source_cancel(livePressureSource);
        };

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"runtimeIRVersion=%@ class=%@",
            runtimeVersion ?: @"(nil)", runtimeVersion ? NSStringFromClass(runtimeVersion.class) : @"(nil)"]);

        NSURL *probeRoot = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2M-LIVE-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableArray<NSDictionary *> *handles = [NSMutableArray arrayWithCapacity:28];
        BOOL phaseAPass = YES;
        NSString *phaseAFailure = nil;

        A12S2MLiveEmit(lines, @"PHASE A BEGIN targetBlocks=28");
        for (NSUInteger block = 0; block < 28; ++block) {
            @autoreleasepool {
                NSString *blockLabel = [NSString stringWithFormat:@"phaseA-b%02lu", (unsigned long)block];
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ START completedHandles=%lu", blockLabel, (unsigned long)handles.count]);
                NSMutableArray<NSString *> *detail = [NSMutableArray array];

                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ donors START", blockLabel]);
                NSArray<NSMutableDictionary *> *donors = A12S2LFindDonors(block, detail);
                if (donors.count != 8) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-missing-donors", (unsigned long)block];
                    A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ donors FAIL count=%lu",
                        blockLabel, (unsigned long)donors.count]);
                    A12S2MLiveAppendTail(lines, detail, 12, @"phaseA detail: ");
                    break;
                }
                uint64_t donorBytes = A12S2LDonorWeightBytes(donors);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ donors PASS donorBytes=%.1fMB", blockLabel,
                    (double)donorBytes / (1024.0 * 1024.0)]);

                NSURL *blockRoot = [probeRoot URLByAppendingPathComponent:
                    [NSString stringWithFormat:@"phaseA-b%02lu", (unsigned long)block]
                    isDirectory:YES];
                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ container-build START", blockLabel]);
                NSMutableDictionary *canonical = A12S2HBuildCanonicalCombined(
                    donors, blockRoot, runtimeVersion, detail);
                NSMutableDictionary *full10 = canonical ? A12S2JSplitQKV(canonical, detail) : nil;
                NSMutableDictionary *plist = full10 ? A12S2HSubset(
                    full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9],
                    detail, [blockLabel stringByAppendingString:@"-full10"]) : nil;
                NSArray<NSDictionary *> *donorMap = donors ? A12S2JDonorMap(donors) : nil;
                NSMutableDictionary *weights = [NSMutableDictionary dictionary];
                BOOL normalized = plist && donorMap.count == 10 && A12S2JNormalizeWeightsDedup(
                    plist, donorMap, blockRoot, weights, detail);
                if (!normalized) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-container-build", (unsigned long)block];
                    A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ container-build FAIL", blockLabel]);
                    A12S2MLiveAppendTail(lines, detail, 16, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }
                uint64_t weightMapBytes = A12S2LWeightMapBytes(weights);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ container-build PASS weightMap=%.1fMB uniqueWeights=%lu",
                    blockLabel, (double)weightMapBytes / (1024.0 * 1024.0),
                    (unsigned long)weights.count]);

                id lowered = nil;
                NSDictionary *nameMap = nil;
                double compileMS = 0.0, loadMS = 0.0;
                BOOL initialCacheHit = NO;
                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ compile-load START", blockLabel]);
                id memoryModel = A12S2LLoadTenProcedureModel(
                    plist, weights, blockLabel, detail,
                    &lowered, &nameMap, &compileMS, &loadMS, &initialCacheHit);
                if (!memoryModel) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-load", (unsigned long)block];
                    A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ compile-load FAIL", blockLabel]);
                    A12S2MLiveAppendTail(lines, detail, 18, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ compile-load PASS cacheHit=%@ compile=%.1fms load=%.2fms",
                    blockLabel, initialCacheHit ? @"yes" : @"no", compileMS, loadMS]);

                double dispatch10MS = 0.0;
                BOOL dispatch10 = A12S2MLiveDispatchAllTen(
                    memoryModel, lowered, nameMap, detail, lines, blockLabel, &dispatch10MS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ unload START dispatch10=%@ wall=%.2fms",
                    blockLabel, dispatch10 ? @"PASS" : @"FAIL", dispatch10MS]);
                double unloadMS = 0.0;
                BOOL unloaded = A12S2LUnloadModel(memoryModel, &unloadMS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ unload result=%@ ms=%.2f", blockLabel,
                    unloaded ? @"PASS" : @"FAIL", unloadMS]);

                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ reconstructed-cache-check START", blockLabel]);
                BOOL reconstructedHit = unloaded && A12S2MReconstructedCacheHit(
                    plist, weights, blockLabel, detail);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ reconstructed-cache-check result=%@",
                    blockLabel, reconstructedHit ? @"PASS" : @"FAIL"]);

                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"phaseA b%02lu SUMMARY donor=%.1fMB weightMap=%.1fMB initialCacheHit=%@ compile=%.1fms load=%.2fms dispatch10=%@ dispatch10Wall=%.2fms unload=%@ unloadMS=%.2f reconstructedCacheHit=%@",
                    (unsigned long)block,
                    (double)donorBytes / (1024.0 * 1024.0),
                    (double)weightMapBytes / (1024.0 * 1024.0),
                    initialCacheHit ? @"yes" : @"no", compileMS, loadMS,
                    dispatch10 ? @"PASS" : @"FAIL", dispatch10MS,
                    unloaded ? @"PASS" : @"FAIL", unloadMS,
                    reconstructedHit ? @"yes" : @"no"]);

                if (!dispatch10 || !unloaded || !reconstructedHit) {
                    phaseAPass = NO;
                    phaseAFailure = [NSString stringWithFormat:@"b%02lu-verify", (unsigned long)block];
                    A12S2MLiveAppendTail(lines, detail, 14, @"phaseA detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                [handles addObject:@{
                    @"block": @(block), @"model": memoryModel, @"donorBytes": @(donorBytes),
                }];
                [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ COMPLETE retainedUnloadedHandles=%lu",
                    blockLabel, (unsigned long)handles.count]);
            }
        }

        if (!phaseAPass || handles.count != 28) {
            [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC phaseA failure=%@ handles=%lu expected=28",
                phaseAFailure ?: @"unknown", (unsigned long)handles.count]);
            cleanupLiveObservers();
            return A12S2MLiveFinalize(lines);
        }
        A12S2MLiveEmit(lines, @"PHASE A PASS precompiled=28 reconstructedCacheHits=28 unloadedHandles=28");

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *surfaceError = nil;
        A12ANESurface *input = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        A12ANESurface *output = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        if (!input || !output) {
            [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"RESULT=FAIL stage=phaseB-surface error=%@", A12S2Error(surfaceError)]);
            cleanupLiveObservers();
            return A12S2MLiveFinalize(lines);
        }
        A12S2KFillDeterministic(input);
        A12S2KZero(output);

        A12S2MLiveEmit(lines, @"HANDLE RELOAD PREFLIGHT BEGIN b00 descriptorRebuild=no weightMap=no");
        {
            NSMutableArray<NSString *> *detail = [NSMutableArray array];
            NSDictionary *firstHandle = handles.firstObject;
            id model = firstHandle[@"model"];
            NSDictionary *nameMap = nil;
            double loadMS = 0.0;
            id lowered = A12S2MReloadHandle(
                model, @"handle-preflight-b00", detail, &nameMap, &loadMS);
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"handle-preflight-b00 reload result=%@ load=%.2fms",
                lowered ? @"PASS" : @"FAIL", loadMS]);
            double evalMS = 0.0;
            BOOL evalOK = lowered && A12S2MRunSelfO(
                model, lowered, nameMap, input, output,
                @"handle-preflight-b00-self_o", detail, &evalMS);
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"handle-preflight-b00 self_o result=%@ eval=%.2fms",
                evalOK ? @"PASS" : @"FAIL", evalMS]);
            double unloadMS = 0.0;
            BOOL unloadOK = lowered && A12S2LUnloadModel(model, &unloadMS);
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"handle-preflight-b00 unload result=%@ ms=%.2f",
                unloadOK ? @"PASS" : @"FAIL", unloadMS]);
            if (!lowered || !evalOK || !unloadOK) {
                A12S2MLiveAppendTail(lines, detail, 14, @"preflight detail: ");
                [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"RESULT=DIAGNOSTIC stage=unloaded-handle-reload load=%.2f eval=%.2f unload=%.2f",
                    loadMS, evalMS, unloadMS]);
                cleanupLiveObservers();
                return A12S2MLiveFinalize(lines);
            }
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"handleReloadPreflight=PASS b00 load=%.2fms eval=%.2fms unload=%.2fms descriptorRebuild=no weightMap=no",
                loadMS, evalMS, unloadMS]);
        }

        usleep(250000);
        __block volatile unsigned int uiWarnings = 0;
        __block volatile unsigned int dispatchWarnings = 0;
        __block volatile unsigned long dispatchFlags = 0;
        id warningToken = [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
            object:nil queue:nil usingBlock:^(__unused NSNotification *note) {
                ++uiWarnings;
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
                ++dispatchWarnings;
            });
            dispatch_resume(pressureSource);
        }

        A12S2MLiveEmit(lines, @"PHASE B BEGIN reloadOnly=yes tempWeightMap=0 compile=0");
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
            NSString *label = [NSString stringWithFormat:@"phaseB-b%02lu", (unsigned long)block];
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"%@ START residentBefore=%lu", label, (unsigned long)resident.count]);

            NSDictionary *nameMap = nil;
            double loadMS = 0.0;
            A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ reload START", label]);
            id lowered = A12S2MReloadHandle(memoryModel, label, detail, &nameMap, &loadMS);
            if (!lowered) {
                onsetBlock = block;
                onsetReason = @"handle-reload-failure";
                onsetModelAdmitted = NO;
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ reload FAIL load=%.2fms", label, loadMS]);
                A12S2MLiveAppendTail(lines, detail, 14, @"phaseB detail: ");
                break;
            }
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"%@ reload PASS load=%.2fms", label, loadMS]);

            [resident addObject:@{
                @"block": @(block), @"model": memoryModel,
                @"lowered": lowered, @"names": nameMap,
            }];
            cumulativeDonorBytes += [handle[@"donorBytes"] unsignedLongLongValue];

            double evalMS = 0.0;
            A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ newest-self_o START", label]);
            BOOL evalOK = A12S2MRunSelfO(
                memoryModel, lowered, nameMap, input, output,
                [label stringByAppendingString:@"-self_o"], detail, &evalMS);
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"%@ newest-self_o result=%@ eval=%.2fms",
                label, evalOK ? @"PASS" : @"FAIL", evalMS]);

            BOOL sentinelOK = YES;
            double sentinelMS = 0.0;
            if (block > 0) {
                NSDictionary *first = resident.firstObject;
                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ b00-sentinel START", label]);
                sentinelOK = A12S2MRunSelfO(
                    first[@"model"], first[@"lowered"], first[@"names"],
                    input, output, @"phaseB-b00-sentinel", detail, &sentinelMS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ b00-sentinel result=%@ eval=%.2fms",
                    label, sentinelOK ? @"PASS" : @"FAIL", sentinelMS]);
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

            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"phaseB b%02lu SUMMARY resident=%lu cumulativeDonor=%.1fMB reloadOnly=yes tempWeightMap=0 compile=0 load=%.2fms eval=%.2fms sentinel=%.2fms freshPressure=%@ pressure(ui=%u dispatch=%u flags=0x%lx) thresholds(load=%.1f eval=%.1f)",
                (unsigned long)block, (unsigned long)resident.count,
                (double)cumulativeDonorBytes / (1024.0 * 1024.0),
                loadMS, evalMS, sentinelMS, freshPressure ? @"yes" : @"no",
                uiNow, dispatchNow, flagsNow, loadThreshold, evalThreshold]);

            if (!evalOK || !sentinelOK || freshPressure || loadPathology || evalPathology) {
                onsetBlock = block;
                onsetModelAdmitted = YES;
                if (!evalOK) onsetReason = @"newest-eval-failure";
                else if (!sentinelOK) onsetReason = @"oldest-sentinel-failure";
                else if (freshPressure) onsetReason = @"fresh-memory-pressure";
                else if (loadPathology) onsetReason = @"pathological-load";
                else onsetReason = @"pathological-eval";
                A12S2MLiveAppendTail(lines, detail, 10, @"phaseB detail: ");
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"phaseB STOP-ONSET b%02lu reason=%@ residentNow=%lu safeBeforeOnset=%lu",
                    (unsigned long)block, onsetReason,
                    (unsigned long)resident.count,
                    (unsigned long)(resident.count > 0 ? resident.count - 1 : 0)]);
                break;
            }
            if (healthyLoads.count < 8) [healthyLoads addObject:@(loadMS)];
            if (healthyEvals.count < 8) [healthyEvals addObject:@(MAX(evalMS, sentinelMS))];
        }

        NSUInteger admitted = resident.count;
        NSUInteger lastSafe = onsetBlock == NSNotFound
            ? admitted
            : (onsetModelAdmitted ? (admitted > 0 ? admitted - 1 : 0) : admitted);
        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"phaseB residencyResult admitted=%lu lastSafe=%lu onsetBlock=%@ reason=%@ cumulativeDonor=%.1fMB uiWarnings=%u dispatchWarnings=%u flags=0x%lx",
            (unsigned long)admitted, (unsigned long)lastSafe,
            onsetBlock == NSNotFound ? @"none" :
                [NSString stringWithFormat:@"%lu", (unsigned long)onsetBlock],
            onsetReason ?: @"none",
            (double)cumulativeDonorBytes / (1024.0 * 1024.0),
            (unsigned int)uiWarnings, (unsigned int)dispatchWarnings,
            (unsigned long)dispatchFlags]);

        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"PHASE B CLEANUP BEGIN resident=%lu", (unsigned long)resident.count]);
        double unloadTotal = 0.0;
        NSUInteger unloadFailures = 0;
        for (NSDictionary *record in resident.reverseObjectEnumerator) {
            NSUInteger block = [record[@"block"] unsignedIntegerValue];
            A12S2MLiveEmit(lines, [NSString stringWithFormat:@"phaseB cleanup b%02lu unload START", (unsigned long)block]);
            double ms = 0.0;
            BOOL ok = A12S2LUnloadModel(record[@"model"], &ms);
            unloadTotal += ms;
            if (!ok) ++unloadFailures;
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"phaseB cleanup b%02lu unload=%@ ms=%.2f",
                (unsigned long)block, ok ? @"PASS" : @"FAIL", ms]);
        }
        A12S2MLiveEmit(lines, [NSString stringWithFormat:
            @"phaseB cleanupSummary unloads=%lu failures=%lu total=%.2fms avg=%.2fms",
            (unsigned long)resident.count, (unsigned long)unloadFailures,
            unloadTotal, resident.count ? unloadTotal / (double)resident.count : 0.0]);

        if (warningToken) [NSNotificationCenter.defaultCenter removeObserver:warningToken];
        if (pressureSource) dispatch_source_cancel(pressureSource);

        usleep(500000);
        A12S2MLiveEmit(lines, @"PHASE C BEGIN sampleBlocks=0,9,27 cyclesEach=3");
        BOOL phaseCPass = YES;
        NSArray<NSNumber *> *sampleBlocks = @[@0, @9, @27];
        for (NSNumber *rawBlock in sampleBlocks) {
            NSUInteger block = rawBlock.unsignedIntegerValue;
            if (block >= handles.count) continue;
            id memoryModel = handles[block][@"model"];
            for (NSUInteger cycle = 0; cycle < 3; ++cycle) {
                NSMutableArray<NSString *> *detail = [NSMutableArray array];
                NSString *label = [NSString stringWithFormat:@"phaseC-b%02lu-c%lu",
                    (unsigned long)block, (unsigned long)cycle];
                A12S2MLiveEmit(lines, [NSString stringWithFormat:@"%@ START", label]);
                NSDictionary *nameMap = nil;
                double loadMS = 0.0;
                id lowered = A12S2MReloadHandle(memoryModel, label, detail, &nameMap, &loadMS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ reload result=%@ load=%.2fms", label,
                    lowered ? @"PASS" : @"FAIL", loadMS]);
                double evalMS = 0.0;
                BOOL evalOK = lowered && A12S2MRunSelfO(
                    memoryModel, lowered, nameMap, input, output,
                    [label stringByAppendingString:@"-self_o"], detail, &evalMS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"%@ self_o result=%@ eval=%.2fms", label,
                    evalOK ? @"PASS" : @"FAIL", evalMS]);
                double unloadMS = 0.0;
                BOOL unloadOK = lowered && A12S2LUnloadModel(memoryModel, &unloadMS);
                A12S2MLiveEmit(lines, [NSString stringWithFormat:
                    @"phaseC b%02lu cycle=%lu load=%.2fms eval=%.2fms unload=%.2fms result=%@",
                    (unsigned long)block, (unsigned long)cycle,
                    loadMS, evalMS, unloadMS,
                    (lowered && evalOK && unloadOK) ? @"PASS" : @"FAIL"]);
                if (!lowered || !evalOK || !unloadOK) {
                    phaseCPass = NO;
                    A12S2MLiveAppendTail(lines, detail, 10, @"phaseC detail: ");
                    break;
                }
            }
            if (!phaseCPass) break;
        }

        [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
        if (!phaseCPass) {
            A12S2MLiveEmit(lines, @"RESULT=FAIL stage=phaseC-warm-cycle");
        } else if (unloadFailures > 0) {
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"RESULT=FAIL stage=phaseB-cleanup unloadFailures=%lu",
                (unsigned long)unloadFailures]);
        } else if (onsetBlock == NSNotFound && admitted == 28) {
            A12S2MLiveEmit(lines,
                @"RESULT=PASS full28-reload-only-multiprocedure-blocks-resident phaseA=PASS phaseC=PASS");
        } else if (onsetBlock != NSNotFound) {
            A12S2MLiveEmit(lines, [NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC clean-residency-boundary lastSafe=%lu onsetAt=%lu reason=%@ phaseA=PASS phaseC=PASS",
                (unsigned long)lastSafe, (unsigned long)onsetBlock, onsetReason ?: @"unknown"]);
        } else {
            A12S2MLiveEmit(lines, @"RESULT=FAIL stage=phaseB-incomplete");
        }

        cleanupLiveObservers();
        return A12S2MLiveFinalize(lines);
    }
}

#endif
