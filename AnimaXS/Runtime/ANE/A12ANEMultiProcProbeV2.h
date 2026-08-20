#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>

// Experiment branch only.
//
// V1 was accidentally implemented as a constructor inside A12ANEBridge.h.
// That constructor has now been removed. This V2 probe is called explicitly
// from AnimaXSApp on physical devices, so there is no ambiguity about whether
// it was linked or executed.
//
// V2 answers one narrow question with a controlled pair:
//   A) Can _ANEInMemoryModel compile/load the V5 identity graph when its entry
//      point is the conventional `main`?
//   B) If A passes, can the same MIL contain `main` + `double`, and if so how
//      many private ANE procedures does the loaded model expose?
//
// No diffusion/model pack is touched and no IOSurface evaluation is needed yet.
// If B exposes >=2 procedures, the next probe will test procedureIndex dispatch.

#if TARGET_OS_SIMULATOR
static inline NSString *A12MPV2Run(void) {
    return @"ANE private multi-procedure MIL POC v2\nRESULT=SKIP simulator";
}
#else

static inline NSString *A12MPV2ErrorText(NSError *error) {
    if (!error) return @"unknown";
    NSError *under = error.userInfo[NSUnderlyingErrorKey];
    if (under) {
        return [NSString stringWithFormat:@"%@ | underlying=%@",
            error.localizedDescription ?: @"error",
            under.localizedDescription ?: [under description]];
    }
    return error.localizedDescription ?: [error description];
}

static inline NSUInteger A12MPV2ProcedureCount(id memoryModel,
                                                NSDictionary **nameMapOut,
                                                NSArray **proceduresOut) {
    NSDictionary *attrs = @{};
    @try {
        id value = ((id(*)(id,SEL))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"modelAttributes"));
        if ([value isKindOfClass:NSDictionary.class]) attrs = value;
    } @catch (__unused NSException *exception) {}

    NSDictionary *desc = [attrs[@"ANEFModelDescription"] isKindOfClass:NSDictionary.class]
        ? attrs[@"ANEFModelDescription"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    if (nameMapOut) *nameMapOut = nameMap;
    if (proceduresOut) *proceduresOut = procedures;
    return MAX(procedures.count, nameMap.count);
}

static inline BOOL A12MPV2CompileLoadVariant(NSData *milData,
                                              NSData *weightData,
                                              NSString *label,
                                              NSMutableArray<NSString *> *lines,
                                              NSUInteger *procedureCountOut) {
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class inMemoryClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !inMemoryClass) {
        [lines addObject:[NSString stringWithFormat:
            @"variant=%@ RESULT=FAIL stage=class-discovery", label]];
        return NO;
    }

    NSDictionary *weightMap = @{
        @"@model_path/weights/weight.bin": @{
            @"offset": @0,
            @"data": weightData
        }
    };
    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
        milData, weightMap, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        inMemoryClass,
        NSSelectorFromString(@"inMemoryModelWithDescriptor:"),
        descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:
            @"variant=%@ RESULT=FAIL stage=model-construction", label]];
        return NO;
    }

    BOOL cacheHit = NO;
    SEL cachedSelector = NSSelectorFromString(@"compiledModelExists");
    if ([memoryModel respondsToSelector:cachedSelector]) {
        cacheHit = ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel, cachedSelector);
    }

    double compileMS = 0.0;
    if (!cacheHit) {
        NSError *compileError = nil;
        CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        BOOL compiled = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel,
            NSSelectorFromString(@"compileWithQoS:options:error:"),
            25u, @{}, &compileError);
        compileMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0;
        if (!compiled) {
            [lines addObject:[NSString stringWithFormat:
                @"variant=%@ cacheHit=no compile=FAIL compileMs=%.1f error=%@",
                label, compileMS, A12MPV2ErrorText(compileError)]];
            return NO;
        }
    }

    NSError *loadError = nil;
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel,
        NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:
            @"variant=%@ cacheHit=%@ compile=PASS compileMs=%.1f load=FAIL loadMs=%.1f error=%@",
            label, cacheHit ? @"yes" : @"no", compileMS, loadMS,
            A12MPV2ErrorText(loadError)]];
        return NO;
    }

    NSDictionary *nameMap = nil;
    NSArray *procedures = nil;
    NSUInteger procedureCount = A12MPV2ProcedureCount(memoryModel, &nameMap, &procedures);
    if (procedureCountOut) *procedureCountOut = procedureCount;
    [lines addObject:[NSString stringWithFormat:
        @"variant=%@ cacheHit=%@ compile=PASS compileMs=%.1f load=PASS loadMs=%.1f procedureCount=%lu names=%@ procedures=%lu",
        label, cacheHit ? @"yes" : @"no", compileMS, loadMS,
        (unsigned long)procedureCount, nameMap ?: @{},
        (unsigned long)procedures.count]];

    NSError *unloadError = nil;
    CFAbsoluteTime t2 = CFAbsoluteTimeGetCurrent();
    BOOL unloaded = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        memoryModel,
        NSSelectorFromString(@"unloadWithQoS:error:"),
        25u, &unloadError);
    double unloadMS = (CFAbsoluteTimeGetCurrent() - t2) * 1000.0;
    [lines addObject:[NSString stringWithFormat:
        @"variant=%@ unload=%@ unloadMs=%.1f%@",
        label, unloaded ? @"PASS" : @"FAIL", unloadMS,
        unloaded ? @"" : [NSString stringWithFormat:@" error=%@",
            A12MPV2ErrorText(unloadError)]]];
    return YES;
}

static inline NSString *A12MPV2Run(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE private multi-procedure MIL POC v2",
            @"control=single-main vs main+double; same V5 graph/weights; no diffusion",
            nil];

        void *aneHandle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        if (!aneHandle) {
            [lines addObject:@"RESULT=FAIL stage=framework-load"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *templateURL = [[NSBundle mainBundle]
            URLForResource:@"Conv2048W8Template" withExtension:@"bundle"];
        NSURL *assetURL = [templateURL
            URLByAppendingPathComponent:@"V5TwoProcedure.mlmodelc" isDirectory:YES];
        NSError *ioError = nil;
        NSString *source = [NSString stringWithContentsOfURL:
            [assetURL URLByAppendingPathComponent:@"model.mil"]
            encoding:NSUTF8StringEncoding error:&ioError];
        NSData *weights = [NSData dataWithContentsOfURL:
            [assetURL URLByAppendingPathComponent:@"weights/weight.bin"]
            options:0 error:&ioError];
        if (!source || !weights) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL stage=fixture-read error=%@",
                A12MPV2ErrorText(ioError)]];
            return [lines componentsJoinedByString:@"\n"];
        }

        source = [source stringByReplacingOccurrencesOfString:@"[1, 64, 1, 64]"
                                                   withString:@"[1, 64, 1, 128]"];

        NSRange doubleStart = [source rangeOfString:@"    func double<ios18>"];
        NSRange identityStart = [source rangeOfString:@"    func identity<ios18>"];
        NSRange programEnd = [source rangeOfString:@"\n}"
                                         options:NSBackwardsSearch];
        if (doubleStart.location == NSNotFound ||
            identityStart.location == NSNotFound ||
            programEnd.location == NSNotFound ||
            !(doubleStart.location < identityStart.location &&
              identityStart.location < programEnd.location)) {
            [lines addObject:@"RESULT=FAIL stage=fixture-parse"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSString *prefix = [source substringToIndex:doubleStart.location];
        NSString *doubleFunction = [source substringWithRange:NSMakeRange(
            doubleStart.location, identityStart.location - doubleStart.location)];
        NSString *identityFunction = [source substringWithRange:NSMakeRange(
            identityStart.location, programEnd.location - identityStart.location)];
        NSString *mainFunction = [identityFunction
            stringByReplacingOccurrencesOfString:@"func identity<ios18>"
                                        withString:@"func main<ios18>"];

        NSString *singleMainText = [NSString stringWithFormat:@"%@%@\n}\n",
            prefix, mainFunction];
        NSString *mainPlusDoubleText = [NSString stringWithFormat:@"%@%@%@\n}\n",
            prefix, mainFunction, doubleFunction];
        NSData *singleMainMIL = [singleMainText dataUsingEncoding:NSUTF8StringEncoding];
        NSData *mainPlusDoubleMIL = [mainPlusDoubleText dataUsingEncoding:NSUTF8StringEncoding];
        if (!singleMainMIL || !mainPlusDoubleMIL) {
            [lines addObject:@"RESULT=FAIL stage=mil-encoding"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSUInteger singleCount = 0;
        BOOL singleOK = A12MPV2CompileLoadVariant(
            singleMainMIL, weights, @"single-main", lines, &singleCount);
        if (!singleOK) {
            [lines addObject:@"RESULT=INCONCLUSIVE single-main-rejected"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSUInteger multiCount = 0;
        BOOL multiOK = A12MPV2CompileLoadVariant(
            mainPlusDoubleMIL, weights, @"main+double", lines, &multiCount);
        if (!multiOK) {
            [lines addObject:@"RESULT=RAW_MULTIFUNCTION_REJECTED singleMain=PASS mainPlusDouble=FAIL"];
        } else if (multiCount >= 2) {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=PASS privateProcedures=%lu next=procedureIndex-dispatch",
                (unsigned long)multiCount]];
        } else {
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=FAIL multifunction-loaded-but-procedureCount=%lu",
                (unsigned long)multiCount]];
        }
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
