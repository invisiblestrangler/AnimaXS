#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <stdio.h>
#import <unistd.h>
#import "A12ANEMultiProcResidencyStage2L.h"

// Stage 2N isolates the post-Stage2K loader regression.
// Proven reference points from device evidence:
//   Stage 2J freshly compiled full10 load: ~86.84 ms
//   Stage 2K cache-hit full10 load:       ~15.83 ms
//   Stage 2M cache-hit loads:             ~156-180+ ms
//
// This probe intentionally uses distinct real blocks for cases A-F so one case
// cannot warm the exact compiled model identity used by the next case.
// No diffusion, no progressive residency, and at most one ANE model is loaded.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2NProbe(void) {
    return @"ANE multiprocedure loader parity Stage 2N\nRESULT=SKIP simulator";
}
#else

static inline NSString *A12S2NRootPath(void) {
    NSString *cache = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    return [(cache ?: NSTemporaryDirectory()) stringByAppendingPathComponent:@"AnimaXS-ANE"];
}
static inline NSString *A12S2NLivePath(void) {
    return [A12S2NRootPath() stringByAppendingPathComponent:@"stage2n-live-latest.log"];
}
static inline NSString *A12S2NFinalPath(void) {
    return [A12S2NRootPath() stringByAppendingPathComponent:@"stage2n-final-latest.log"];
}
static inline FILE **A12S2NFileSlot(void) { static FILE *f = NULL; return &f; }
static inline void A12S2NLive(NSString *line) {
    if (!line) return;
    fprintf(stderr, "[ANE-S2N] %s\n", line.UTF8String ?: "");
    fflush(stderr);
    FILE *f = *A12S2NFileSlot();
    if (f) {
        fprintf(f, "%s\n", line.UTF8String ?: "");
        fflush(f);
        fsync(fileno(f));
    }
}
static inline void A12S2NReport(NSMutableArray<NSString *> *report, NSString *line) {
    if (line) [report addObject:line];
    A12S2NLive(line);
}
static inline void A12S2NBegin(void) {
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:A12S2NRootPath() withIntermediateDirectories:YES attributes:nil error:nil];
    FILE **slot = A12S2NFileSlot();
    if (*slot) { fclose(*slot); *slot = NULL; }
    *slot = fopen(A12S2NLivePath().fileSystemRepresentation, "w");
    A12S2NLive([NSString stringWithFormat:@"LIVE BEGIN live=%@ final=%@", A12S2NLivePath(), A12S2NFinalPath()]);
}
static inline NSString *A12S2NFinish(NSMutableArray<NSString *> *report) {
    NSString *text = [report componentsJoinedByString:@"\n"];
    [text writeToFile:A12S2NFinalPath() atomically:YES encoding:NSUTF8StringEncoding error:nil];
    A12S2NLive([NSString stringWithFormat:@"LIVE COMPLETE final=%@", A12S2NFinalPath()]);
    FILE **slot = A12S2NFileSlot();
    if (*slot) { fclose(*slot); *slot = NULL; }
    return text;
}

static inline NSString *A12S2NBool(BOOL v) { return v ? @"yes" : @"no"; }
static inline id A12S2NGetObject(id obj, NSString *selector) {
    SEL s = NSSelectorFromString(selector);
    return obj && [obj respondsToSelector:s] ? ((id(*)(id,SEL))objc_msgSend)(obj, s) : nil;
}
static inline uint64_t A12S2NGetU64(id obj, NSString *selector) {
    SEL s = NSSelectorFromString(selector);
    return obj && [obj respondsToSelector:s] ? ((uint64_t(*)(id,SEL))objc_msgSend)(obj, s) : 0;
}
static inline NSString *A12S2NPostLoadState(id model) {
    id descriptor = A12S2NGetObject(model, @"descriptor");
    id lowered = A12S2NGetObject(model, @"model");
    id attributes = A12S2NGetObject(model, @"modelAttributes");
    id program = A12S2NGetObject(model, @"program");
    id modelURL = A12S2NGetObject(model, @"modelURL");
    uint64_t handle = A12S2NGetU64(model, @"programHandle");
    uint64_t state = A12S2NGetU64(model, @"state");
    return [NSString stringWithFormat:
        @"descriptor=%@ model=%@ attributes=%@ program=%@ programHandle=%llu state=%llu modelURL=%@",
        A12S2NBool(descriptor != nil), A12S2NBool(lowered != nil), A12S2NBool(attributes != nil),
        A12S2NBool(program != nil), (unsigned long long)handle, (unsigned long long)state,
        modelURL ?: @"(nil)"];
}

static inline BOOL A12S2NBuildBlock(
    NSUInteger block,
    NSURL *root,
    NSString *runtimeVersion,
    NSMutableArray<NSString *> *detail,
    NSMutableDictionary **plistOut,
    NSMutableDictionary **weightsOut) {
    NSArray<NSMutableDictionary *> *donors = A12S2LFindDonors(block, detail);
    if (donors.count != 8) return NO;
    NSMutableDictionary *canonical = A12S2HBuildCanonicalCombined(donors, root, runtimeVersion, detail);
    NSMutableDictionary *full10 = canonical ? A12S2JSplitQKV(canonical, detail) : nil;
    NSMutableDictionary *plist = full10 ? A12S2HSubset(
        full10, @[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9], detail,
        [NSString stringWithFormat:@"s2n-b%02lu-full10", (unsigned long)block]) : nil;
    NSArray<NSDictionary *> *donorMap = donors ? A12S2JDonorMap(donors) : nil;
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    BOOL ok = plist && donorMap.count == 10 &&
        A12S2JNormalizeWeightsDedup(plist, donorMap, root, weights, detail);
    if (!ok) return NO;
    if (plistOut) *plistOut = plist;
    if (weightsOut) *weightsOut = weights;
    return YES;
}

static inline id A12S2NNewModel(
    NSDictionary *plist,
    NSDictionary *weights,
    NSString *label,
    NSMutableArray<NSString *> *report,
    BOOL *cacheHitOut) {
    NSError *serializeError = nil;
    NSData *netData = [NSPropertyListSerialization dataWithPropertyList:plist
        format:NSPropertyListBinaryFormat_v1_0 options:0 error:&serializeError];
    if (!netData) {
        A12S2NReport(report, [NSString stringWithFormat:@"%@ MODEL FAIL serialize=%@", label, A12S2Error(serializeError)]);
        return nil;
    }
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class memoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    id descriptor = descriptorClass ? ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass, NSSelectorFromString(@"modelWithNetworkDescription:weights:optionsPlist:"),
        netData, weights, nil) : nil;
    id model = descriptor && memoryModelClass ? ((id(*)(Class,SEL,id))objc_msgSend)(
        memoryModelClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!model) {
        A12S2NReport(report, [NSString stringWithFormat:@"%@ MODEL FAIL descriptor/model", label]);
        return nil;
    }
    BOOL hit = [model respondsToSelector:NSSelectorFromString(@"compiledModelExists")]
        ? ((BOOL(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"compiledModelExists")) : NO;
    if (cacheHitOut) *cacheHitOut = hit;
    A12S2NReport(report, [NSString stringWithFormat:@"%@ MODEL cacheHit=%@", label, A12S2NBool(hit)]);
    return model;
}

static inline BOOL A12S2NLoad(
    id model,
    NSString *label,
    NSMutableArray<NSString *> *report,
    double *loadMSOut,
    id *loweredOut,
    NSDictionary **namesOut) {
    A12S2NReport(report, [NSString stringWithFormat:@"%@ LOAD START", label]);
    dispatch_semaphore_t watchdog = dispatch_semaphore_create(0);
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2 * NSEC_PER_SEC)),
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            if (dispatch_semaphore_wait(watchdog, DISPATCH_TIME_NOW) != 0) {
                A12S2NLive([NSString stringWithFormat:@"%@ LOAD STALL >2000ms", label]);
            }
        });
    NSError *loadError = nil;
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        model, NSSelectorFromString(@"loadWithQoS:options:error:"), 25u, @{}, &loadError);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    dispatch_semaphore_signal(watchdog);
    if (loadMSOut) *loadMSOut = ms;
    A12S2NReport(report, [NSString stringWithFormat:@"%@ LOAD %@ ms=%.2f error=%@",
        label, loaded ? @"PASS" : @"FAIL", ms, A12S2Error(loadError)]);
    if (!loaded) return NO;
    id lowered = A12S2NGetObject(model, @"model");
    NSDictionary *desc = A12S2ANEFDescription(lowered ?: model);
    NSDictionary *names = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSUInteger count = MAX(names.count, procedures.count);
    A12S2NReport(report, [NSString stringWithFormat:@"%@ POST %@ procedureCount=%lu",
        label, A12S2NPostLoadState(model), (unsigned long)count]);
    if (!lowered || count != 10) return NO;
    if (loweredOut) *loweredOut = lowered;
    if (namesOut) *namesOut = names;
    return YES;
}

static inline BOOL A12S2NQuickSelfO(
    id model,
    id lowered,
    NSDictionary *names,
    NSString *label,
    NSMutableArray<NSString *> *report,
    double *evalMSOut) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *surfaceError = nil;
    A12ANESurface *input = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
    A12ANESurface *output = [[A12ANESurface alloc] initWithDevice:device channels:2048 spatial:1024 error:&surfaceError];
    NSNumber *pid = A12S2KProcedureID(names, @"procedure_self_o");
    if (!input || !output || !pid) {
        A12S2NReport(report, [NSString stringWithFormat:@"%@ EVAL FAIL surface/pid error=%@", label, A12S2Error(surfaceError)]);
        return NO;
    }
    A12S2KFillDeterministic(input);
    A12S2KZero(output);
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = A12S2KEvaluateProcedure(model, lowered, pid.unsignedIntegerValue, input, output, label, detail);
    double ms = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (evalMSOut) *evalMSOut = ms;
    A12S2NReport(report, [NSString stringWithFormat:@"%@ EVAL %@ ms=%.2f", label, ok ? @"PASS" : @"FAIL", ms]);
    return ok;
}

static inline BOOL A12S2NUnload(id model, NSString *label, NSMutableArray<NSString *> *report, double *msOut) {
    double ms = 0.0;
    BOOL ok = A12S2LUnloadModel(model, &ms);
    if (msOut) *msOut = ms;
    A12S2NReport(report, [NSString stringWithFormat:@"%@ UNLOAD %@ ms=%.2f", label, ok ? @"PASS" : @"FAIL", ms]);
    return ok;
}

static inline BOOL A12S2NPrepareCase(
    NSUInteger block,
    NSString *caseName,
    NSString *runtimeVersion,
    NSURL *probeRoot,
    NSMutableArray<NSString *> *report,
    NSMutableDictionary **plistOut,
    NSMutableDictionary **weightsOut,
    NSURL **caseRootOut) {
    NSURL *root = [probeRoot URLByAppendingPathComponent:[NSString stringWithFormat:@"%@-b%02lu", caseName, (unsigned long)block] isDirectory:YES];
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    NSMutableDictionary *plist = nil, *weights = nil;
    A12S2NReport(report, [NSString stringWithFormat:@"%@ BUILD START block=%lu", caseName, (unsigned long)block]);
    BOOL ok = A12S2NBuildBlock(block, root, runtimeVersion, detail, &plist, &weights);
    A12S2NReport(report, [NSString stringWithFormat:@"%@ BUILD %@ weights=%lu bytes=%.1fMB",
        caseName, ok ? @"PASS" : @"FAIL", (unsigned long)weights.count,
        (double)A12S2LWeightMapBytes(weights)/(1024.0*1024.0)]);
    if (!ok) {
        NSUInteger start = detail.count > 12 ? detail.count - 12 : 0;
        for (NSUInteger i=start; i<detail.count; ++i) A12S2NReport(report, [@"detail: " stringByAppendingString:detail[i]]);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        return NO;
    }
    if (plistOut) *plistOut = plist;
    if (weightsOut) *weightsOut = weights;
    if (caseRootOut) *caseRootOut = root;
    return YES;
}

static inline BOOL A12S2NRunABCD(
    NSUInteger block,
    NSString *caseName,
    NSUInteger style, // 0=exact-K prep+fresh, 1=fresh direct, 2=touch localPath, 3=current-L helper
    NSString *runtimeVersion,
    NSURL *probeRoot,
    NSMutableArray<NSString *> *report,
    NSMutableDictionary *metrics) {
    @autoreleasepool {
        NSMutableDictionary *plist=nil, *weights=nil; NSURL *root=nil;
        if (!A12S2NPrepareCase(block, caseName, runtimeVersion, probeRoot, report, &plist, &weights, &root)) return NO;
        NSMutableArray<NSString *> *detail = [NSMutableArray array];
        if (style == 0) {
            A12S2NReport(report, [NSString stringWithFormat:@"%@ PREP exact-StageK A12S2EPreparedCompileOnly START", caseName]);
            A12S2ECompileResult prep = A12S2EPreparedCompileOnly(plist, weights, [caseName stringByAppendingString:@"-prep"], detail);
            NSString *last = detail.lastObject ?: @"(no detail)";
            A12S2NReport(report, [NSString stringWithFormat:@"%@ PREP result=%ld tail=%@", caseName, (long)prep, last]);
            if (prep != A12S2ECompilePass) { [NSFileManager.defaultManager removeItemAtURL:root error:nil]; return NO; }
        }

        id model=nil, lowered=nil; NSDictionary *names=nil; double load=0, eval=0, unload=0; BOOL hit=NO;
        if (style == 3) {
            double compileMS=0; BOOL helperHit=NO;
            model = A12S2LLoadTenProcedureModel(plist, weights, caseName, detail, &lowered, &names, &compileMS, &load, &helperHit);
            hit = helperHit;
            A12S2NReport(report, [NSString stringWithFormat:@"%@ L-HELPER result=%@ cacheHit=%@ compile=%.2f load=%.2f POST %@",
                caseName, model ? @"PASS" : @"FAIL", A12S2NBool(hit), compileMS, load, model ? A12S2NPostLoadState(model) : @"(nil)"]);
        } else {
            model = A12S2NNewModel(plist, weights, caseName, report, &hit);
            if (model && style == 2) {
                NSString *path = A12S2NGetObject(model, @"localModelPath");
                A12S2NReport(report, [NSString stringWithFormat:@"%@ TOUCH localModelPath=%@", caseName, path ?: @"(nil)"]);
            }
            if (model && hit) A12S2NLoad(model, caseName, report, &load, &lowered, &names);
        }
        BOOL ok = model && hit && lowered && names.count == 10;
        if (ok) ok = A12S2NQuickSelfO(model, lowered, names, caseName, report, &eval);
        if (model && lowered) ok = A12S2NUnload(model, caseName, report, &unload) && ok;
        metrics[[caseName stringByAppendingString:@".load1"]] = @(load);
        metrics[[caseName stringByAppendingString:@".eval1"]] = @(eval);
        metrics[[caseName stringByAppendingString:@".unload1"]] = @(unload);
        A12S2NReport(report, [NSString stringWithFormat:@"%@ SUMMARY block=%lu cacheHit=%@ load=%.2f eval=%.2f unload=%.2f result=%@",
            caseName, (unsigned long)block, A12S2NBool(hit), load, eval, unload, ok ? @"PASS" : @"FAIL"]);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        return ok;
    }
}

static inline BOOL A12S2NRunReloadCase(
    NSUInteger block,
    NSString *caseName,
    BOOL sameObject,
    NSString *runtimeVersion,
    NSURL *probeRoot,
    NSMutableArray<NSString *> *report,
    NSMutableDictionary *metrics) {
    @autoreleasepool {
        NSMutableDictionary *plist=nil, *weights=nil; NSURL *root=nil;
        if (!A12S2NPrepareCase(block, caseName, runtimeVersion, probeRoot, report, &plist, &weights, &root)) return NO;
        __block id retainedModel=nil;
        __block BOOL hit1=NO; __block double load1=0, eval1=0, unload1=0;
        __block BOOL firstOK=NO;
        @autoreleasepool {
            id model=A12S2NNewModel(plist, weights, [caseName stringByAppendingString:@"-first"], report, &hit1);
            id lowered=nil; NSDictionary *names=nil;
            firstOK = model && hit1 && A12S2NLoad(model, [caseName stringByAppendingString:@"-first"], report, &load1, &lowered, &names);
            if (firstOK) firstOK = A12S2NQuickSelfO(model, lowered, names, [caseName stringByAppendingString:@"-first"], report, &eval1);
            if (model && lowered) firstOK = A12S2NUnload(model, [caseName stringByAppendingString:@"-first"], report, &unload1) && firstOK;
            if (sameObject) retainedModel = model;
        }
        BOOL hit2=YES; id model2=retainedModel;
        if (!sameObject) model2=A12S2NNewModel(plist, weights, [caseName stringByAppendingString:@"-second-new"], report, &hit2);
        double load2=0, eval2=0, unload2=0; id lowered2=nil; NSDictionary *names2=nil;
        NSString *secondLabel=[caseName stringByAppendingString:(sameObject ? @"-second-same" : @"-second-new")];
        BOOL secondOK=model2 && hit2 && A12S2NLoad(model2, secondLabel, report, &load2, &lowered2, &names2);
        if (secondOK) secondOK=A12S2NQuickSelfO(model2, lowered2, names2, secondLabel, report, &eval2);
        if (model2 && lowered2) secondOK=A12S2NUnload(model2, secondLabel, report, &unload2) && secondOK;
        metrics[[caseName stringByAppendingString:@".load1"]]=@(load1);
        metrics[[caseName stringByAppendingString:@".load2"]]=@(load2);
        A12S2NReport(report, [NSString stringWithFormat:@"%@ SUMMARY block=%lu reuse=%@ first=%.2f second=%.2f delta=%.2f result=%@",
            caseName, (unsigned long)block, sameObject ? @"same-object" : @"new-object",
            load1, load2, load2-load1, (firstOK && secondOK) ? @"PASS" : @"FAIL"]);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        return firstOK && secondOK;
    }
}

static inline NSString *A12ANEStage2NProbe(void) {
    @autoreleasepool {
        A12S2NBegin();
        NSMutableArray<NSString *> *report=[NSMutableArray array];
        NSMutableDictionary *metrics=[NSMutableDictionary dictionary];
        A12S2NReport(report, @"ANE multiprocedure loader parity Stage 2N");
        A12S2NReport(report, @"reference=Stage2J fresh-compiled full10 load 86.84ms; Stage2K cache-hit full10 load 15.83ms; Stage2M cache-hit ~156-180+ms");
        A12S2NReport(report, @"matrix=A exact-K prep+fresh; B fresh-direct; C fresh+localModelPath; D current-L-helper; E new-object reload; F same-object reload");
        A12S2NReport(report, @"isolation=distinct real blocks 0..5; max one loaded model; all cases require compiledModelExists=yes");
        NSString *runtimeVersion=A12S2GCurrentANEIRVersion(report);
        NSURL *probeRoot=[NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"AnimaXS-ANE-S2N-%@", NSUUID.UUID.UUIDString]] isDirectory:YES];

        BOOL a=A12S2NRunABCD(0,@"A-exactK",0,runtimeVersion,probeRoot,report,metrics);
        BOOL b=A12S2NRunABCD(1,@"B-freshDirect",1,runtimeVersion,probeRoot,report,metrics);
        BOOL c=A12S2NRunABCD(2,@"C-touchLocalPath",2,runtimeVersion,probeRoot,report,metrics);
        BOOL d=A12S2NRunABCD(3,@"D-currentL",3,runtimeVersion,probeRoot,report,metrics);
        BOOL e=A12S2NRunReloadCase(4,@"E-newObjectReload",NO,runtimeVersion,probeRoot,report,metrics);
        BOOL f=A12S2NRunReloadCase(5,@"F-sameObjectReload",YES,runtimeVersion,probeRoot,report,metrics);

        double A=[metrics[@"A-exactK.load1"] doubleValue];
        double B=[metrics[@"B-freshDirect.load1"] doubleValue];
        double C=[metrics[@"C-touchLocalPath.load1"] doubleValue];
        double D=[metrics[@"D-currentL.load1"] doubleValue];
        double E1=[metrics[@"E-newObjectReload.load1"] doubleValue];
        double E2=[metrics[@"E-newObjectReload.load2"] doubleValue];
        double F1=[metrics[@"F-sameObjectReload.load1"] doubleValue];
        double F2=[metrics[@"F-sameObjectReload.load2"] doubleValue];
        A12S2NReport(report, [NSString stringWithFormat:@"MATRIX loadMS A=%.2f B=%.2f C=%.2f D=%.2f E(first/new2)=%.2f/%.2f F(first/same2)=%.2f/%.2f", A,B,C,D,E1,E2,F1,F2]);
        A12S2NReport(report, [NSString stringWithFormat:@"SIGNALS exactKFast=%@ localPathPenalty=%.2f LHelperPenaltyVsB=%.2f newReloadDelta=%.2f sameReloadDelta=%.2f",
            A12S2NBool(A>0 && A<50.0), C-B, D-B, E2-E1, F2-F1]);
        BOOL all=a&&b&&c&&d&&e&&f;
        A12S2NReport(report, [NSString stringWithFormat:@"RESULT=%@ loader-parity-matrix-complete", all ? @"PASS" : @"DIAGNOSTIC"]);
        [NSFileManager.defaultManager removeItemAtURL:probeRoot error:nil];
        return A12S2NFinish(report);
    }
}

#endif
