#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <UIKit/UIKit.h>
#import <objc/message.h>
#import <mach/mach.h>
#import <malloc/malloc.h>
#import <stdio.h>
#import <unistd.h>

#define A12ANEStage2MProbe A12ANEStage2MProbeOriginal
#import "A12ANEMultiProcResidencyStage2M.h"
#undef A12ANEStage2MProbe

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEStage2MProbe(void) {
    return @"ANE Stage2M memory-safe rerun\nRESULT=SKIP simulator";
}
#else

// Process phys_footprint is used only to diagnose the jetsam/process-memory issue.
// It is NOT an estimate of logical ANE residency.
static inline uint64_t A12S2MMemFootprint(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self_, TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS
        ? (uint64_t)info.phys_footprint : 0;
}
static inline double A12S2MMemMB(void) { return (double)A12S2MMemFootprint()/(1024.0*1024.0); }
static inline size_t A12S2MMemRelief(void) { return malloc_zone_pressure_relief(NULL, 0); }

static inline NSString *A12S2MMemRoot(void) {
    NSString *cache = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    return [(cache ?: NSTemporaryDirectory()) stringByAppendingPathComponent:@"AnimaXS-ANE"];
}
static inline NSString *A12S2MMemLivePath(void) { return [A12S2MMemRoot() stringByAppendingPathComponent:@"stage2m-live-latest.log"]; }
static inline NSString *A12S2MMemPrevPath(void) { return [A12S2MMemRoot() stringByAppendingPathComponent:@"stage2m-live-previous.log"]; }
static inline NSString *A12S2MMemFinalPath(void) { return [A12S2MMemRoot() stringByAppendingPathComponent:@"stage2m-final-latest.log"]; }
static inline FILE **A12S2MMemFileSlot(void) { static FILE *f = NULL; return &f; }
static inline void A12S2MMemLog(NSString *line, BOOL sync) {
    if (!line) return;
    fprintf(stderr, "[ANE-S2M] %s\n", line.UTF8String ?: ""); fflush(stderr);
    FILE *f = *A12S2MMemFileSlot();
    if (f) { fprintf(f, "%s\n", line.UTF8String ?: ""); fflush(f); if (sync) fsync(fileno(f)); }
}
static inline void A12S2MMemBegin(void) {
    NSFileManager *fm = NSFileManager.defaultManager;
    [fm createDirectoryAtPath:A12S2MMemRoot() withIntermediateDirectories:YES attributes:nil error:nil];
    FILE **slot = A12S2MMemFileSlot(); if (*slot) { fclose(*slot); *slot = NULL; }
    if ([fm fileExistsAtPath:A12S2MMemLivePath()]) {
        [fm removeItemAtPath:A12S2MMemPrevPath() error:nil];
        [fm moveItemAtPath:A12S2MMemLivePath() toPath:A12S2MMemPrevPath() error:nil];
        fprintf(stderr, "[ANE-S2M] previous incomplete/live log preserved: %s\n", A12S2MMemPrevPath().UTF8String ?: "");
    }
    *slot = fopen(A12S2MMemLivePath().fileSystemRepresentation, "w");
    A12S2MMemLog([NSString stringWithFormat:@"LIVE BEGIN footprint=%.1fMB live=%@ final=%@", A12S2MMemMB(), A12S2MMemLivePath(), A12S2MMemFinalPath()], YES);
}
static inline void A12S2MMemReport(NSMutableArray<NSString *> *report, NSString *line) {
    if (line) [report addObject:line]; A12S2MMemLog(line, YES);
}
static inline NSString *A12S2MMemFinish(NSMutableArray<NSString *> *report) {
    NSString *text = [report componentsJoinedByString:@"\n"];
    [text writeToFile:A12S2MMemFinalPath() atomically:YES encoding:NSUTF8StringEncoding error:nil];
    A12S2MMemLog([NSString stringWithFormat:@"LIVE COMPLETE footprint=%.1fMB final=%@", A12S2MMemMB(), A12S2MMemFinalPath()], YES);
    FILE **slot = A12S2MMemFileSlot(); if (*slot) { fclose(*slot); *slot = NULL; }
    return text;
}
static inline void A12S2MMemTail(NSMutableArray<NSString *> *report, NSArray<NSString *> *detail, NSUInteger n, NSString *prefix) {
    NSUInteger start = detail.count > n ? detail.count - n : 0;
    for (NSUInteger i=start; i<detail.count; ++i) A12S2MMemReport(report, [NSString stringWithFormat:@"%@%@", prefix ?: @"", detail[i]]);
}

// Loader already clears descriptor. These modes additionally clear post-load fields
// that unloadWithQoS leaves attached to _ANEInMemoryModel.
static inline NSString *A12S2MTrimName(NSUInteger mode) {
    return mode==3 ? @"model+attributes+program" : mode==2 ? @"model+attributes" : mode==1 ? @"model" : @"descriptor-only";
}
static inline BOOL A12S2MNilSetter(id obj, NSString *name, BOOL required, NSMutableArray<NSString *> *detail) {
    SEL s = NSSelectorFromString(name);
    if (![obj respondsToSelector:s]) { if (required) [detail addObject:[NSString stringWithFormat:@"missing selector %@", name]]; return !required; }
    ((void(*)(id,SEL,id))objc_msgSend)(obj,s,nil); return YES;
}
static inline BOOL A12S2MTrim(id model, NSUInteger mode, NSMutableArray<NSString *> *detail, NSString *label, BOOL verbose) {
    uint64_t before=A12S2MMemFootprint();
    BOOL ok=A12S2MNilSetter(model,@"setDescriptor:",NO,detail);
    if (mode>=1) ok=A12S2MNilSetter(model,@"setModel:",YES,detail)&&ok;
    if (mode>=2) ok=A12S2MNilSetter(model,@"setModelAttributes:",NO,detail)&&ok;
    if (mode>=3) ok=A12S2MNilSetter(model,@"setProgram:",NO,detail)&&ok;
    size_t relieved=A12S2MMemRelief(); uint64_t after=A12S2MMemFootprint();
    if (verbose) A12S2MMemLog([NSString stringWithFormat:@"%@ trim=%@ %@ footprint %.1f->%.1fMB delta=%.1fMB relief=%.1fMB", label,A12S2MTrimName(mode),ok?@"PASS":@"FAIL",(double)before/(1024.0*1024.0),(double)after/(1024.0*1024.0),(double)((int64_t)after-(int64_t)before)/(1024.0*1024.0),(double)relieved/(1024.0*1024.0)],YES);
    return ok;
}
static inline BOOL A12S2MQuickEval(id model,id lowered,NSDictionary *names,NSMutableArray<NSString *> *detail,NSString *label,double *msOut) {
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError *e=nil;
    A12ANESurface *in=[[A12ANESurface alloc] initWithDevice:dev channels:2048 spatial:1024 error:&e];
    A12ANESurface *out=[[A12ANESurface alloc] initWithDevice:dev channels:2048 spatial:1024 error:&e];
    if(!in||!out){[detail addObject:[NSString stringWithFormat:@"%@ surface fail %@",label,A12S2Error(e)]];return NO;}
    A12S2KFillDeterministic(in); return A12S2MRunSelfO(model,lowered,names,in,out,label,detail,msOut);
}

// Find the most aggressive trim that still supports load + real dispatch.
static inline id A12S2MTrimPreflight(NSDictionary *plist,NSDictionary *weights,NSUInteger *modeOut,NSMutableArray<NSString *> *detail) {
    for(NSInteger mode=3;mode>=0;--mode){
        NSString *label=[NSString stringWithFormat:@"trim-preflight-m%ld",(long)mode];
        A12S2MMemLog([NSString stringWithFormat:@"%@ START footprint=%.1fMB",label,A12S2MMemMB()],YES);
        id lowered=nil; NSDictionary *names=nil; double compile=0,load=0; BOOL hit=NO;
        id model=A12S2LLoadTenProcedureModel(plist,weights,label,detail,&lowered,&names,&compile,&load,&hit);
        if(!model) continue;
        double u1=0; if(!A12S2LUnloadModel(model,&u1)) continue;
        if(!A12S2MTrim(model,(NSUInteger)mode,detail,label,YES)) continue;
        NSDictionary *names2=nil; double reload=0; id lowered2=A12S2MReloadHandle(model,[label stringByAppendingString:@"-reload"],detail,&names2,&reload);
        double eval=0; BOOL evalOK=lowered2&&A12S2MQuickEval(model,lowered2,names2,detail,[label stringByAppendingString:@"-self_o"],&eval);
        double u2=0; BOOL unloadOK=lowered2&&A12S2LUnloadModel(model,&u2);
        BOOL trimAgain=unloadOK&&A12S2MTrim(model,(NSUInteger)mode,detail,[label stringByAppendingString:@"-again"],YES);
        A12S2MMemLog([NSString stringWithFormat:@"%@ result=%@ cacheHit=%@ load=%.2f reload=%.2f eval=%.2f unload=%.2f footprint=%.1fMB",label,(evalOK&&unloadOK&&trimAgain)?@"PASS":@"FAIL",hit?@"yes":@"no",load,reload,eval,u2,A12S2MMemMB()],YES);
        if(evalOK&&unloadOK&&trimAgain){*modeOut=(NSUInteger)mode;return model;}
    }
    return nil;
}

static inline NSString *A12ANEStage2MProbe(void) {
    @autoreleasepool {
        A12S2MMemBegin(); NSMutableArray<NSString *> *report=[NSMutableArray array];
        A12S2MMemReport(report,@"ANE Stage2M memory-safe rerun");
        A12S2MMemReport(report,@"reason=prior run hit jetsam in phaseA after retaining 14 unloaded handles; test reloadable post-unload trimming before residency sweep");
        A12S2MMemReport(report,@"processFootprint=jetsam health only; NOT logical ANE residency");

        __block volatile unsigned int uiWarnings=0,dispatchWarnings=0; __block volatile unsigned long dispatchFlags=0;
        id warning=[NSNotificationCenter.defaultCenter addObserverForName:UIApplicationDidReceiveMemoryWarningNotification object:nil queue:nil usingBlock:^(__unused NSNotification *n){++uiWarnings;A12S2MMemLog([NSString stringWithFormat:@"MEMORY WARNING ui=%u footprint=%.1fMB",uiWarnings,A12S2MMemMB()],YES);}];
        dispatch_queue_t pq=dispatch_queue_create("com.invisiblestrangler.AnimaXS.s2m-mem",DISPATCH_QUEUE_SERIAL);
        dispatch_source_t ps=dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE,0,DISPATCH_MEMORYPRESSURE_WARN|DISPATCH_MEMORYPRESSURE_CRITICAL,pq);
        if(ps){dispatch_source_set_event_handler(ps,^{dispatchFlags=dispatch_source_get_data(ps);++dispatchWarnings;A12S2MMemLog([NSString stringWithFormat:@"MEMORY PRESSURE dispatch=%u flags=0x%lx footprint=%.1fMB",dispatchWarnings,dispatchFlags,A12S2MMemMB()],YES);});dispatch_resume(ps);}
        void (^cleanup)(void)=^{if(warning)[NSNotificationCenter.defaultCenter removeObserver:warning];if(ps)dispatch_source_cancel(ps);};

        uint64_t baseline=A12S2MMemFootprint(); NSString *runtime=A12S2GCurrentANEIRVersion(report);
        A12S2MMemReport(report,[NSString stringWithFormat:@"baseline=%.1fMB runtimeIR=%@",(double)baseline/(1024.0*1024.0),runtime?:@"nil"]);
        NSURL *root=[NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"S2M-MEM-%@",NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableArray<NSDictionary *> *handles=[NSMutableArray arrayWithCapacity:28]; NSUInteger trimMode=NSNotFound; NSString *phaseAFailure=nil;

        A12S2MMemReport(report,@"PHASE A BEGIN");
        for(NSUInteger b=0;b<28;++b){
            unsigned int u0=uiWarnings,d0=dispatchWarnings; __block BOOL ok=YES; __block NSString *why=nil; __block uint64_t donorBytes=0; __block BOOL cacheHit=NO; __block double loadMS=0,evalMS=0,unloadMS=0;
            A12S2MMemLog([NSString stringWithFormat:@"A b%02lu START handles=%lu footprint=%.1fMB",(unsigned long)b,(unsigned long)handles.count,A12S2MMemMB()],YES);
            @autoreleasepool {
                NSMutableArray<NSString *> *detail=[NSMutableArray array]; NSString *label=[NSString stringWithFormat:@"A-b%02lu",(unsigned long)b];
                NSArray *donors=A12S2LFindDonors(b,detail); if(donors.count!=8){ok=NO;why=@"donors";}
                donorBytes=ok?A12S2LDonorWeightBytes(donors):0; NSURL *br=[root URLByAppendingPathComponent:label isDirectory:YES];
                NSMutableDictionary *weights=[NSMutableDictionary dictionary]; NSMutableDictionary *plist=nil;
                if(ok){
                    A12S2MMemLog([NSString stringWithFormat:@"A b%02lu build START footprint=%.1fMB",(unsigned long)b,A12S2MMemMB()],NO);
                    NSMutableDictionary *c=A12S2HBuildCanonicalCombined(donors,br,runtime,detail); NSMutableDictionary *f=c?A12S2JSplitQKV(c,detail):nil;
                    plist=f?A12S2HSubset(f,@[@0,@1,@2,@3,@4,@5,@6,@7,@8,@9],detail,[label stringByAppendingString:@"-full10"]):nil;
                    NSArray *map=A12S2JDonorMap(donors); ok=plist&&map.count==10&&A12S2JNormalizeWeightsDedup(plist,map,br,weights,detail); if(!ok)why=@"build";
                    A12S2MMemLog([NSString stringWithFormat:@"A b%02lu build %@ weightMap=%.1fMB footprint=%.1fMB",(unsigned long)b,ok?@"PASS":@"FAIL",(double)A12S2LWeightMapBytes(weights)/(1024.0*1024.0),A12S2MMemMB()],YES);
                }
                if(ok&&(uiWarnings>u0||dispatchWarnings>d0)){ok=NO;why=@"pressure-during-build";}
                id model=nil;
                if(ok&&b==0){model=A12S2MTrimPreflight(plist,weights,&trimMode,detail);if(!model){ok=NO;why=@"no-reloadable-trim";}else cacheHit=YES;}
                else if(ok){
                    id lowered=nil;NSDictionary *names=nil;double compile=0;model=A12S2LLoadTenProcedureModel(plist,weights,label,detail,&lowered,&names,&compile,&loadMS,&cacheHit);
                    if(!model){ok=NO;why=@"load";}else{
                        evalMS=0;BOOL eval=A12S2MQuickEval(model,lowered,names,detail,[label stringByAppendingString:@"-self_o"],&evalMS);
                        BOOL unloaded=A12S2LUnloadModel(model,&unloadMS);BOOL trimmed=unloaded&&trimMode!=NSNotFound&&A12S2MTrim(model,trimMode,detail,label,NO);
                        if(!eval||!unloaded||!trimmed){ok=NO;why=!eval?@"eval":(!unloaded?@"unload":@"trim");}
                    }
                }
                if(ok&&model)[handles addObject:@{@"block":@(b),@"model":model,@"donorBytes":@(donorBytes)}];
                if(!ok)A12S2MMemTail(report,detail,12,@"A detail: "); [NSFileManager.defaultManager removeItemAtURL:br error:nil];
            }
            size_t relief=A12S2MMemRelief(); double footprint=A12S2MMemMB(),growth=footprint-(double)baseline/(1024.0*1024.0);
            A12S2MMemReport(report,[NSString stringWithFormat:@"A b%02lu POST result=%@ handles=%lu trim=%@ cacheHit=%@ load=%.2f eval=%.2f unload=%.2f footprint=%.1fMB growth=%.1fMB relief=%.1fMB warnings=%u/%u",(unsigned long)b,ok?@"PASS":@"FAIL",(unsigned long)handles.count,trimMode==NSNotFound?@"none":A12S2MTrimName(trimMode),cacheHit?@"yes":@"no",loadMS,evalMS,unloadMS,footprint,growth,(double)relief/(1024.0*1024.0),uiWarnings,dispatchWarnings]);
            if(!ok||uiWarnings>u0||dispatchWarnings>d0){phaseAFailure=[NSString stringWithFormat:@"b%02lu-%@",(unsigned long)b,why?:@"memory-pressure"];break;}
        }
        if(handles.count!=28){[NSFileManager.defaultManager removeItemAtURL:root error:nil];cleanup();A12S2MMemReport(report,[NSString stringWithFormat:@"RESULT=DIAGNOSTIC phaseA failure=%@ handles=%lu trim=%@ footprint=%.1fMB",phaseAFailure?:@"incomplete",(unsigned long)handles.count,trimMode==NSNotFound?@"none":A12S2MTrimName(trimMode),A12S2MMemMB()]);return A12S2MMemFinish(report);}
        A12S2MMemReport(report,[NSString stringWithFormat:@"PHASE A PASS handles=28 trim=%@ footprint=%.1fMB",A12S2MTrimName(trimMode),A12S2MMemMB()]);

        id<MTLDevice> dev=MTLCreateSystemDefaultDevice();NSError *se=nil;A12ANESurface *input=[[A12ANESurface alloc]initWithDevice:dev channels:2048 spatial:1024 error:&se];A12ANESurface *output=[[A12ANESurface alloc]initWithDevice:dev channels:2048 spatial:1024 error:&se];
        if(!input||!output){cleanup();A12S2MMemReport(report,[NSString stringWithFormat:@"RESULT=FAIL surfaces %@",A12S2Error(se)]);return A12S2MMemFinish(report);}A12S2KFillDeterministic(input);

        A12S2MMemReport(report,[NSString stringWithFormat:@"PHASE B BEGIN footprint=%.1fMB",A12S2MMemMB()]);
        NSMutableArray *resident=[NSMutableArray array];NSUInteger onset=NSNotFound;NSString *reason=nil;BOOL admittedOnset=NO;uint64_t cumulative=0;
        for(NSUInteger b=0;b<28;++b){
            unsigned int u0=uiWarnings,d0=dispatchWarnings;NSMutableArray *detail=[NSMutableArray array];id model=handles[b][@"model"];NSDictionary *names=nil;double load=0;
            A12S2MMemLog([NSString stringWithFormat:@"B b%02lu reload START resident=%lu footprint=%.1fMB",(unsigned long)b,(unsigned long)resident.count,A12S2MMemMB()],YES);
            id lowered=A12S2MReloadHandle(model,[NSString stringWithFormat:@"B-b%02lu",(unsigned long)b],detail,&names,&load);if(!lowered){onset=b;reason=@"reload-failure";break;}
            [resident addObject:@{@"block":@(b),@"model":model,@"lowered":lowered,@"names":names}];cumulative+=[handles[b][@"donorBytes"] unsignedLongLongValue];
            double eval=0,sentinel=0;BOOL eok=A12S2MRunSelfO(model,lowered,names,input,output,@"B-newest",detail,&eval),sok=YES;if(b>0){NSDictionary *first=resident.firstObject;sok=A12S2MRunSelfO(first[@"model"],first[@"lowered"],first[@"names"],input,output,@"B-b00-sentinel",detail,&sentinel);}
            if(ps)dispatch_sync(pq,^{});BOOL pressure=uiWarnings>u0||dispatchWarnings>d0;
            A12S2MMemReport(report,[NSString stringWithFormat:@"B b%02lu resident=%lu donor=%.1fMB load=%.2f eval=%.2f sentinel=%.2f pressure=%@ footprint=%.1fMB warnings=%u/%u flags=0x%lx",(unsigned long)b,(unsigned long)resident.count,(double)cumulative/(1024.0*1024.0),load,eval,sentinel,pressure?@"yes":@"no",A12S2MMemMB(),uiWarnings,dispatchWarnings,dispatchFlags]);
            if(!eok||!sok||pressure){onset=b;admittedOnset=YES;reason=!eok?@"newest-eval":(!sok?@"sentinel":@"memory-pressure");break;}
        }
        NSUInteger admitted=resident.count,lastSafe=onset==NSNotFound?admitted:(admittedOnset?(admitted?admitted-1:0):admitted);
        A12S2MMemReport(report,[NSString stringWithFormat:@"B RESULT admitted=%lu lastSafe=%lu onset=%@ reason=%@ footprint=%.1fMB",(unsigned long)admitted,(unsigned long)lastSafe,onset==NSNotFound?@"none":[NSString stringWithFormat:@"%lu",(unsigned long)onset],reason?:@"none",A12S2MMemMB()]);
        NSUInteger cleanupFailures=0;for(NSDictionary *r in resident.reverseObjectEnumerator){double u=0;BOOL uok=A12S2LUnloadModel(r[@"model"],&u);NSMutableArray *d=[NSMutableArray array];BOOL tok=uok&&A12S2MTrim(r[@"model"],trimMode,d,@"B-cleanup",NO);if(!uok||!tok)++cleanupFailures;A12S2MMemRelief();A12S2MMemReport(report,[NSString stringWithFormat:@"B cleanup b%02lu unload=%@ %.2f trim=%@ footprint=%.1fMB",(unsigned long)[r[@"block"] unsignedIntegerValue],uok?@"PASS":@"FAIL",u,tok?@"PASS":@"FAIL",A12S2MMemMB()]);}
        // The residency records retain lowered models/name maps. Drop them after unload+trim
        // or they would defeat the handle trimming and contaminate Phase C memory.
        [resident removeAllObjects]; A12S2MMemRelief();
        A12S2MMemReport(report,[NSString stringWithFormat:@"B cleanup RELEASED residentRecords footprint=%.1fMB",A12S2MMemMB()]);

        A12S2MMemReport(report,@"PHASE C BEGIN samples=0,9,27 cycles=3");BOOL cpass=YES;
        for(NSNumber *bn in @[@0,@9,@27]){NSUInteger b=bn.unsignedIntegerValue;id model=handles[b][@"model"];for(NSUInteger c=0;c<3;++c){NSMutableArray *d=[NSMutableArray array];NSDictionary *names=nil;double load=0;id lowered=A12S2MReloadHandle(model,@"C",d,&names,&load);double eval=0;BOOL e=lowered&&A12S2MRunSelfO(model,lowered,names,input,output,@"C-self_o",d,&eval);double u=0;BOOL uok=lowered&&A12S2LUnloadModel(model,&u);BOOL t=uok&&A12S2MTrim(model,trimMode,d,@"C-trim",NO);A12S2MMemRelief();A12S2MMemReport(report,[NSString stringWithFormat:@"C b%02lu c%lu load=%.2f eval=%.2f unload=%.2f trim=%@ footprint=%.1fMB result=%@",(unsigned long)b,(unsigned long)c,load,eval,u,t?@"PASS":@"FAIL",A12S2MMemMB(),(lowered&&e&&uok&&t)?@"PASS":@"FAIL"]);if(!lowered||!e||!uok||!t){cpass=NO;break;}}if(!cpass)break;}

        [NSFileManager.defaultManager removeItemAtURL:root error:nil];cleanup();
        if(!cpass)A12S2MMemReport(report,@"RESULT=FAIL phaseC");
        else if(cleanupFailures)A12S2MMemReport(report,[NSString stringWithFormat:@"RESULT=FAIL cleanupFailures=%lu",(unsigned long)cleanupFailures]);
        else if(onset==NSNotFound&&admitted==28)A12S2MMemReport(report,@"RESULT=PASS full28 clean residency");
        else A12S2MMemReport(report,[NSString stringWithFormat:@"RESULT=DIAGNOSTIC clean-residency lastSafe=%lu onset=%lu reason=%@",(unsigned long)lastSafe,(unsigned long)onset,reason?:@"unknown"]);
        return A12S2MMemFinish(report);
    }
}
#endif
