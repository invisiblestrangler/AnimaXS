#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>
#import "A12ANEBlock0Stage2K.h"

// Stage 2L: measure the real residency ceiling of the proven 10-procedure/block
// representation. Stage 2K proved bit-exact dispatch for block 0. This probe
// progressively constructs and loads the same representation for blocks 0...27,
// keeps every admitted model resident, and stops at the first fresh pressure
// signal, load pathology, or evaluation failure.
//
// Important: construction data is scoped to an inner autorelease pool and the
// _ANEInMemoryModel descriptor is cleared after load so the ~189 MB temporary
// in-memory weight map for a block is not retained by the app. Only the loaded
// ANE model/program plus its small runtime metadata survives into the next step.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2LProbe(void) {
    return @"ANE multiprocedure progressive residency Stage 2L\nRESULT=SKIP simulator";
}
#else

static inline NSArray<NSMutableDictionary *> *A12S2LFindDonors(
    NSUInteger block,
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
        [lines addObject:[NSString stringWithFormat:@"b%02lu donorCache=FAIL error=%@",
            (unsigned long)block, A12S2Error(listError)]];
        return nil;
    }

    NSMutableArray<NSMutableDictionary *> *found = [NSMutableArray arrayWithCapacity:8];
    for (NSDictionary *baseSpec in A12S2Specs()) {
        NSString *prefix = [NSString stringWithFormat:@"ane-u8-row-v1-b%lu-%@-",
            (unsigned long)block, baseSpec[@"needle"]];
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
            [lines addObject:[NSString stringWithFormat:@"b%02lu donor=%@ MISSING prefix=%@",
                (unsigned long)block, baseSpec[@"name"], prefix]];
            return nil;
        }
        NSMutableDictionary *spec = [baseSpec mutableCopy];
        NSURL *chosen = matches.firstObject;
        spec[@"url"] = chosen;
        spec[@"netURL"] = [chosen URLByAppendingPathComponent:@"model.espresso.net"];
        spec[@"metadataKey"] = A12S2MetadataKey(spec);
        [found addObject:spec];
    }
    return found;
}

static inline uint64_t A12S2LDonorWeightBytes(NSArray<NSDictionary *> *donors) {
    uint64_t total = 0;
    for (NSDictionary *donor in donors) {
        NSURL *url = [donor[@"url"] isKindOfClass:NSURL.class] ? donor[@"url"] : nil;
        NSURL *weight = [url URLByAppendingPathComponent:@"model.espresso.weights"];
        NSNumber *size = nil;
        [weight getResourceValue:&size forKey:NSURLFileSizeKey error:nil];
        total += size.unsignedLongLongValue;
    }
    return total;
}

static inline uint64_t A12S2LWeightMapBytes(NSDictionary *weightMap) {
    uint64_t total = 0;
    for (NSString *name in weightMap) {
        NSDictionary *entry = [weightMap[name] isKindOfClass:NSDictionary.class] ? weightMap[name] : nil;
        NSData *data = [entry[@"data"] isKindOfClass:NSData.class] ? entry[@"data"] : nil;
        total += data.length;
    }
    return total;
}

static inline double A12S2LMedian(NSArray<NSNumber *> *values) {
    if (values.count == 0) return 0.0;
    NSArray<NSNumber *> *sorted = [values sortedArrayUsingSelector:@selector(compare:)];
    NSUInteger mid = sorted.count / 2;
    if (sorted.count & 1u) return sorted[mid].doubleValue;
    return 0.5 * (sorted[mid - 1].doubleValue + sorted[mid].doubleValue);
}

static inline void A12S2LAppendTail(
    NSMutableArray<NSString *> *dst,
    NSArray<NSString *> *src,
    NSUInteger maxLines,
    NSString *prefix) {
    NSUInteger start = src.count > maxLines ? src.count - maxLines : 0;
    for (NSUInteger i = start; i < src.count; ++i) {
        [dst addObject:[NSString stringWithFormat:@"%@%@", prefix ?: @"", src[i]]];
    }
}

static inline id A12S2LLoadTenProcedureModel(
    NSDictionary *plist,
    NSDictionary *weights,
    NSString *label,
    NSMutableArray<NSString *> *lines,
    id *loweredModelOut,
    NSDictionary **nameMapOut,
    double *compileMSOut,
    double *loadMSOut,
    BOOL *cacheHitOut) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        [lines addObject:[NSString stringWithFormat:@"%@ serialize=FAIL error=%@",
            label, A12S2Error(serializeError)]];
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
        [lines addObject:[NSString stringWithFormat:@"%@ model=FAIL", label]];
        return nil;
    }

    BOOL cacheHit = [memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"compiledModelExists")) : NO;
    NSString *localPath = [memoryModel respondsToSelector:NSSelectorFromString(@"localModelPath")]
        ? ((id(*)(id,SEL))objc_msgSend)(memoryModel, NSSelectorFromString(@"localModelPath")) : nil;
    double compileMS = 0.0;
    if (!cacheHit) {
        if (![localPath isKindOfClass:NSString.class] || localPath.length == 0) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ materialize=FAIL stage=localModelPath", label]];
            return nil;
        }

        NSFileManager *fm = NSFileManager.defaultManager;
        NSError *ioError = nil;
        if (![fm createDirectoryAtPath:localPath
            withIntermediateDirectories:YES attributes:nil error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ materialize=FAIL stage=create-dir error=%@ path=%@",
                label, A12S2Error(ioError), localPath]];
            return nil;
        }

        NSString *netPath = [localPath stringByAppendingPathComponent:@"net.plist"];
        if (![netData writeToFile:netPath options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:
                @"%@ materialize=FAIL stage=net-plist error=%@ path=%@",
                label, A12S2Error(ioError), netPath]];
            return nil;
        }

        NSUInteger writtenFiles = 0;
        uint64_t writtenBytes = 0;
        for (NSString *name in weights) {
            NSDictionary *entry = [weights[name] isKindOfClass:NSDictionary.class]
                ? weights[name] : nil;
            NSData *data = [entry[@"data"] isKindOfClass:NSData.class]
                ? entry[@"data"] : nil;
            if (!data) {
                [lines addObject:[NSString stringWithFormat:
                    @"%@ materialize=FAIL stage=weight-data weight=%@", label, name]];
                return nil;
            }

            NSString *weightPath = [localPath stringByAppendingPathComponent:name];
            ioError = nil;
            if (![data writeToFile:weightPath options:NSDataWritingAtomic error:&ioError]) {
                [lines addObject:[NSString stringWithFormat:
                    @"%@ materialize=FAIL stage=weight-write weight=%@ error=%@ path=%@",
                    label, name, A12S2Error(ioError), weightPath]];
                return nil;
            }

            ioError = nil;
            NSDictionary *attributes = [fm attributesOfItemAtPath:weightPath error:&ioError];
            NSNumber *size = [attributes[NSFileSize] isKindOfClass:NSNumber.class]
                ? attributes[NSFileSize] : nil;
            if (!size || size.unsignedLongLongValue != (uint64_t)data.length) {
                [lines addObject:[NSString stringWithFormat:
                    @"%@ materialize=FAIL stage=weight-verify weight=%@ expected=%llu actual=%@ error=%@",
                    label, name, (unsigned long long)data.length,
                    size ?: @"missing", A12S2Error(ioError)]];
                return nil;
            }
            ++writtenFiles;
            writtenBytes += data.length;
        }

        [lines addObject:[NSString stringWithFormat:
            @"%@ materialize=PASS cacheHit=no files=%lu bytes=%llu localPath=%@",
            label, (unsigned long)writtenFiles,
            (unsigned long long)writtenBytes, localPath]];

        NSError *compileError = nil;
        NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
            25u, @{}, &compileError);
        compileMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
        if (!compiled) {
            [lines addObject:[NSString stringWithFormat:@"%@ compile=FAIL ms=%.2f error=%@",
                label, compileMS, A12S2Error(compileError)]];
            return nil;
        }
    } else {
        [lines addObject:[NSString stringWithFormat:
            @"%@ materialize=SKIP cacheHit=yes localPath=%@",
            label, localPath ?: @"(nil)"]];
    }

    NSError *loadError = nil;
    NSTimeInterval loadStart = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - loadStart) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:@"%@ load=FAIL ms=%.2f error=%@",
            label, loadMS, A12S2Error(loadError)]];
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
    if (count != 10 || !loweredModel) {
        NSError *unloadError = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &unloadError);
        [lines addObject:[NSString stringWithFormat:@"%@ procedureCount=FAIL count=%lu names=%@",
            label, (unsigned long)count, nameMap]];
        return nil;
    }

    // On a cache hit _ANEInMemoryModel may still retain its in-memory descriptor,
    // which owns the entire temporary weight map. The loaded program/model has
    // already been created, so drop the descriptor before the next block.
    SEL setDescriptor = NSSelectorFromString(@"setDescriptor:");
    if ([memoryModel respondsToSelector:setDescriptor]) {
        ((void(*)(id,SEL,id))objc_msgSend)(memoryModel, setDescriptor, nil);
    }

    if (loweredModelOut) *loweredModelOut = loweredModel;
    if (nameMapOut) *nameMapOut = nameMap;
    if (compileMSOut) *compileMSOut = compileMS;
    if (loadMSOut) *loadMSOut = loadMS;
    if (cacheHitOut) *cacheHitOut = cacheHit;
    return memoryModel;
}

static inline BOOL A12S2LUnloadModel(id memoryModel, double *msOut) {
    if (!memoryModel) return YES;
    NSError *error = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &error);
    if (msOut) *msOut = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    return ok;
}

static inline NSString *A12ANEStage2LProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE multiprocedure progressive residency Stage 2L",
            @"goal=measure safe resident-block ceiling for real 10-procedure/block representation",
            @"policy=keep admitted blocks resident; stop on fresh pressure, load pathology, or eval failure",
            @"sentinel=evaluate newest self_o + block0 self_o after every admission",
            nil];

        __block volatile unsigned int uiWarnings = 0;
        __block volatile unsigned int dispatchWarnings = 0;
        __block volatile unsigned long dispatchFlags = 0;

        id warningToken = [NSNotificationCenter.defaultCenter
            addObserverForName:UIApplicationDidReceiveMemoryWarningNotification
            object:nil queue:nil usingBlock:^(__unused NSNotification *note) {
                uiWarnings += 1;
            }];
        dispatch_queue_t pressureQueue = dispatch_queue_create(
            "com.invisiblestrangler.AnimaXS.s2l-pressure", DISPATCH_QUEUE_SERIAL);
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

        NSString *runtimeVersion = A12S2GCurrentANEIRVersion(lines);
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *surfaceError = nil;
        A12ANESurface *input = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        A12ANESurface *output = [[A12ANESurface alloc]
            initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
        if (!input || !output) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=surface error=%@",
                A12S2Error(surfaceError)]];
            if (warningToken) [NSNotificationCenter.defaultCenter removeObserver:warningToken];
            if (pressureSource) dispatch_source_cancel(pressureSource);
            return [lines componentsJoinedByString:@"\n"];
        }
        A12S2KFillDeterministic(input);
        A12S2KZero(output);

        NSURL *probeRoot = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:
                @"AnimaXS-ANE-S2L-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableArray<NSDictionary *> *resident = [NSMutableArray arrayWithCapacity:28];
        NSMutableArray<NSNumber *> *healthyLoads = [NSMutableArray array];
        NSMutableArray<NSNumber *> *healthyEvals = [NSMutableArray array];
        uint64_t cumulativeDonorBytes = 0;
        NSUInteger onsetBlock = NSNotFound;
        NSString *onsetReason = nil;

        for (NSUInteger block = 0; block < 28; ++block) {
            @autoreleasepool {
                NSMutableArray<NSString *> *detail = [NSMutableArray array];
                NSArray<NSMutableDictionary *> *donors = A12S2LFindDonors(block, detail);
                if (donors.count != 8) {
                    onsetBlock = block;
                    onsetReason = @"missing-donors";
                    A12S2LAppendTail(lines, detail, 12, @"detail: ");
                    break;
                }
                uint64_t donorBytes = A12S2LDonorWeightBytes(donors);
                NSURL *blockRoot = [probeRoot URLByAppendingPathComponent:
                    [NSString stringWithFormat:@"b%02lu", (unsigned long)block] isDirectory:YES];

                NSMutableDictionary *canonical = A12S2HBuildCanonicalCombined(
                    donors, blockRoot, runtimeVersion, detail);
                NSMutableDictionary *full10 = canonical ? A12S2JSplitQKV(canonical, detail) : nil;
                NSMutableDictionary *modelPlist = full10 ? A12S2HSubset(
                    full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9],
                    detail, [NSString stringWithFormat:@"b%02lu-full10", (unsigned long)block]) : nil;
                NSArray<NSDictionary *> *donorMap = donors ? A12S2JDonorMap(donors) : nil;
                NSMutableDictionary *weights = [NSMutableDictionary dictionary];
                BOOL normalized = modelPlist && donorMap.count == 10 && A12S2JNormalizeWeightsDedup(
                    modelPlist, donorMap, blockRoot, weights, detail);
                if (!normalized) {
                    onsetBlock = block;
                    onsetReason = @"container-build";
                    A12S2LAppendTail(lines, detail, 16, @"detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                uint64_t materializedBytes = A12S2LWeightMapBytes(weights);
                id lowered = nil;
                NSDictionary *nameMap = nil;
                double compileMS = 0.0, loadMS = 0.0;
                BOOL cacheHit = NO;
                NSString *label = [NSString stringWithFormat:@"b%02lu", (unsigned long)block];
                id memoryModel = A12S2LLoadTenProcedureModel(
                    modelPlist, weights, label, detail,
                    &lowered, &nameMap, &compileMS, &loadMS, &cacheHit);
                if (!memoryModel) {
                    onsetBlock = block;
                    onsetReason = @"load-failure";
                    A12S2LAppendTail(lines, detail, 16, @"detail: ");
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                NSNumber *selfOID = A12S2KProcedureID(nameMap, @"procedure_self_o");
                if (!selfOID) {
                    double unloadMS = 0;
                    A12S2LUnloadModel(memoryModel, &unloadMS);
                    onsetBlock = block;
                    onsetReason = @"missing-self-o";
                    [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];
                    break;
                }

                A12S2KZero(output);
                NSTimeInterval evalStart = NSDate.timeIntervalSinceReferenceDate;
                BOOL evalOK = A12S2KEvaluateProcedure(
                    memoryModel, lowered, selfOID.unsignedIntegerValue,
                    input, output, [NSString stringWithFormat:@"b%02lu-self_o", (unsigned long)block], detail);
                double evalWallMS = (NSDate.timeIntervalSinceReferenceDate - evalStart) * 1000.0;

                NSDictionary *record = @{
                    @"block": @(block),
                    @"model": memoryModel,
                    @"lowered": lowered,
                    @"names": nameMap,
                };
                [resident addObject:record];
                cumulativeDonorBytes += donorBytes;

                BOOL sentinelOK = YES;
                double sentinelMS = 0.0;
                if (block > 0 && resident.count > 0) {
                    NSDictionary *first = resident.firstObject;
                    NSNumber *firstID = A12S2KProcedureID(first[@"names"], @"procedure_self_o");
                    A12S2KZero(output);
                    NSTimeInterval sentinelStart = NSDate.timeIntervalSinceReferenceDate;
                    sentinelOK = firstID && A12S2KEvaluateProcedure(
                        first[@"model"], first[@"lowered"], firstID.unsignedIntegerValue,
                        input, output, @"b00-sentinel", detail);
                    sentinelMS = (NSDate.timeIntervalSinceReferenceDate - sentinelStart) * 1000.0;
                }

                unsigned int uiSnapshot = uiWarnings;
                unsigned int dispatchSnapshot = dispatchWarnings;
                unsigned long flagsSnapshot = dispatchFlags;

                double baselineLoad = A12S2LMedian(healthyLoads);
                double baselineEval = A12S2LMedian(healthyEvals);
                double loadThreshold = MAX(500.0, baselineLoad > 0.0 ? baselineLoad * 8.0 : 500.0);
                double evalThreshold = MAX(100.0, baselineEval > 0.0 ? baselineEval * 10.0 : 100.0);
                BOOL loadPathology = healthyLoads.count >= 3 && loadMS > loadThreshold;
                BOOL evalPathology = healthyEvals.count >= 3 &&
                    (evalWallMS > evalThreshold || (block > 0 && sentinelMS > evalThreshold));
                BOOL pressure = uiSnapshot > 0 || dispatchSnapshot > 0;

                [lines addObject:[NSString stringWithFormat:
                    @"b%02lu resident=%lu donor=%.1fMB cumulativeDonor=%.1fMB tempMaterialized=%.1fMB cacheHit=%@ compile=%.1fms load=%.2fms eval=%.2fms sentinel=%.2fms pressure(ui=%u dispatch=%u flags=0x%lx) thresholds(load=%.1f eval=%.1f)",
                    (unsigned long)block, (unsigned long)resident.count,
                    (double)donorBytes / (1024.0 * 1024.0),
                    (double)cumulativeDonorBytes / (1024.0 * 1024.0),
                    (double)materializedBytes / (1024.0 * 1024.0),
                    cacheHit ? @"yes" : @"no", compileMS, loadMS,
                    evalWallMS, sentinelMS, uiSnapshot, dispatchSnapshot,
                    flagsSnapshot, loadThreshold, evalThreshold]];

                [NSFileManager.defaultManager removeItemAtURL:blockRoot error:nil];

                if (!evalOK || !sentinelOK) {
                    onsetBlock = block;
                    onsetReason = !evalOK ? @"newest-eval-failure" : @"oldest-sentinel-failure";
                    A12S2LAppendTail(lines, detail, 10, @"detail: ");
                    break;
                }
                if (pressure || loadPathology || evalPathology) {
                    onsetBlock = block;
                    if (pressure) onsetReason = @"fresh-memory-pressure";
                    else if (loadPathology) onsetReason = @"pathological-load";
                    else onsetReason = @"pathological-eval";
                    [lines addObject:[NSString stringWithFormat:
                        @"STOP-ONSET b%02lu reason=%@ residentNow=%lu lastSafe=%lu",
                        (unsigned long)block, onsetReason,
                        (unsigned long)resident.count,
                        (unsigned long)(resident.count > 0 ? resident.count - 1 : 0)]];
                    break;
                }

                if (healthyLoads.count < 8) [healthyLoads addObject:@(loadMS)];
                if (healthyEvals.count < 8) [healthyEvals addObject:@(MAX(evalWallMS, sentinelMS))];
            }
        }

        NSUInteger admitted = resident.count;
        NSUInteger lastSafe = onsetBlock == NSNotFound ? admitted : (admitted > 0 ? admitted - 1 : 0);
        [lines addObject:[NSString stringWithFormat:
            @"residencyResult admitted=%lu lastSafe=%lu onsetBlock=%@ reason=%@ cumulativeDonor=%.1fMB uiWarnings=%u dispatchWarnings=%u flags=0x%lx",
            (unsigned long)admitted, (unsigned long)lastSafe,
            onsetBlock == NSNotFound ? @"none" : [NSString stringWithFormat:@"%lu", (unsigned long)onsetBlock],
            onsetReason ?: @"none",
            (double)cumulativeDonorBytes / (1024.0 * 1024.0),
            (unsigned int)uiWarnings, (unsigned int)dispatchWarnings,
            (unsigned long)dispatchFlags]];

        double unloadTotal = 0.0;
        NSUInteger unloadFailures = 0;
        for (NSDictionary *record in resident.reverseObjectEnumerator) {
            double ms = 0.0;
            if (!A12S2LUnloadModel(record[@"model"], &ms)) ++unloadFailures;
            unloadTotal += ms;
        }
        [lines addObject:[NSString stringWithFormat:
            @"cleanup unloads=%lu failures=%lu total=%.2fms avg=%.2fms",
            (unsigned long)resident.count, (unsigned long)unloadFailures,
            unloadTotal, resident.count ? unloadTotal / (double)resident.count : 0.0]];

        [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
        if (warningToken) [NSNotificationCenter.defaultCenter removeObserver:warningToken];
        if (pressureSource) dispatch_source_cancel(pressureSource);

        if (onsetBlock == NSNotFound && admitted == 28 && unloadFailures == 0) {
            [lines addObject:@"RESULT=PASS full28-multiprocedure-blocks-resident"];
        } else if (onsetBlock != NSNotFound) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=DIAGNOSTIC residency-boundary lastSafe=%lu onsetAt=%lu reason=%@",
                (unsigned long)lastSafe, (unsigned long)onsetBlock, onsetReason ?: @"unknown"]];
        } else {
            [lines addObject:@"RESULT=FAIL stage=residency-probe-cleanup"];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
