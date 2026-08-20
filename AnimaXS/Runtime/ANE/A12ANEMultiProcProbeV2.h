#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <TargetConditionals.h>

// Experiment branch only. Imported by AnimaXS-Bridging-Header.h after
// A12ANEBridge.h, so the page-valid A12MP IOSurface helpers are already visible.
//
// V1 fed ANEC a valid public Core ML multifunction MIL program whose functions
// were named `double` and `identity`, but the low-level in-memory compiler
// rejected it as InvalidMILProgram before procedure discovery. V2 separates the
// two possible causes in one physical-device launch:
//   A) the in-memory ANEC path requires a `main` function;
//   B) raw multi-function MIL is unsupported/rejected by that path.
//
// It compiles the same V5 identity graph twice from the same weight blob:
//   1) single function renamed to `main`;
//   2) `main` + `double` in one program, with `main` first.
// Only if (2) loads with >=2 procedures do we allocate 16 KiB IOSurfaces and
// dispatch procedure indices 0/1 to validate identity vs x2.

#if !TARGET_OS_SIMULATOR

static inline NSString *A12MPV2DescribeError(NSError *error) {
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
                                                NSDictionary **outNameMap,
                                                NSArray **outProcedures) {
    NSDictionary *attrs = nil;
    @try {
        attrs = ((id(*)(id,SEL))objc_msgSend)(memoryModel,
            NSSelectorFromString(@"modelAttributes"));
    } @catch (__unused NSException *exception) {
        attrs = @{};
    }
    NSDictionary *desc = [attrs[@"ANEFModelDescription"] isKindOfClass:NSDictionary.class]
        ? attrs[@"ANEFModelDescription"] : @{};
    NSArray *procedures = [desc[@"ANEFModelProcedures"] isKindOfClass:NSArray.class]
        ? desc[@"ANEFModelProcedures"] : @[];
    NSDictionary *nameMap = [desc[@"kANEFModelProcedureNameToIDMapKey"] isKindOfClass:NSDictionary.class]
        ? desc[@"kANEFModelProcedureNameToIDMapKey"] : @{};
    if (outNameMap) *outNameMap = nameMap;
    if (outProcedures) *outProcedures = procedures;
    return MAX(procedures.count, nameMap.count);
}

static inline id A12MPV2CompileAndLoad(NSData *milData,
                                       NSData *weightData,
                                       NSString *label,
                                       NSMutableArray<NSString *> *lines) {
    Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class inMemoryClass = NSClassFromString(@"_ANEInMemoryModel");
    if (!descriptorClass || !inMemoryClass) {
        [lines addObject:[NSString stringWithFormat:@"variant=%@ RESULT=FAIL stage=class-discovery", label]];
        return nil;
    }

    NSDictionary *weights = @{
        @"@model_path/weights/weight.bin": @{ @"offset": @0, @"data": weightData }
    };
    id descriptor = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        descriptorClass,
        NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
        milData, weights, nil);
    id memoryModel = descriptor ? ((id(*)(Class,SEL,id))objc_msgSend)(
        inMemoryClass, NSSelectorFromString(@"inMemoryModelWithDescriptor:"), descriptor) : nil;
    if (!descriptor || !memoryModel) {
        [lines addObject:[NSString stringWithFormat:@"variant=%@ RESULT=FAIL stage=model-construction", label]];
        return nil;
    }

    BOOL cacheHit = NO;
    if ([memoryModel respondsToSelector:NSSelectorFromString(@"compiledModelExists")]) {
        cacheHit = ((BOOL(*)(id,SEL))objc_msgSend)(memoryModel,
            NSSelectorFromString(@"compiledModelExists"));
    }

    NSString *tmpDir = nil;
    double compileMS = 0.0;
    if (!cacheHit) {
        NSString *hexID = ((id(*)(id,SEL))objc_msgSend)(memoryModel,
            NSSelectorFromString(@"hexStringIdentifier"));
        if (![hexID isKindOfClass:NSString.class] || hexID.length == 0) {
            [lines addObject:[NSString stringWithFormat:@"variant=%@ RESULT=FAIL stage=cache-key", label]];
            return nil;
        }
        tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexID];
        NSString *weightsDir = [tmpDir stringByAppendingPathComponent:@"weights"];
        NSFileManager *fm = NSFileManager.defaultManager;
        [fm removeItemAtPath:tmpDir error:NULL];
        NSError *ioError = nil;
        if (![fm createDirectoryAtPath:weightsDir withIntermediateDirectories:YES attributes:nil error:&ioError] ||
            ![milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                        options:NSDataWritingAtomic error:&ioError] ||
            ![weightData writeToFile:[weightsDir stringByAppendingPathComponent:@"weight.bin"]
                           options:NSDataWritingAtomic error:&ioError]) {
            [lines addObject:[NSString stringWithFormat:
                @"variant=%@ RESULT=FAIL stage=materialize error=%@",
                label, A12MPV2DescribeError(ioError)]];
            return nil;
        }

        NSError *compileError = nil;
        NSTimeInterval t0 = NSDate.timeIntervalSinceReferenceDate;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"compileWithQoS:options:error:"),
            25u, @{}, &compileError);
        compileMS = (NSDate.timeIntervalSinceReferenceDate - t0) * 1000.0;
        [fm removeItemAtPath:tmpDir error:NULL];
        if (!ok) {
            [lines addObject:[NSString stringWithFormat:
                @"variant=%@ cacheHit=no compile=FAIL compileMs=%.1f error=%@",
                label, compileMS, A12MPV2DescribeError(compileError)]];
            return nil;
        }
    }

    NSError *loadError = nil;
    NSTimeInterval t1 = NSDate.timeIntervalSinceReferenceDate;
    BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError **))objc_msgSend)(
        memoryModel, NSSelectorFromString(@"loadWithQoS:options:error:"),
        25u, @{}, &loadError);
    double loadMS = (NSDate.timeIntervalSinceReferenceDate - t1) * 1000.0;
    if (!loaded) {
        [lines addObject:[NSString stringWithFormat:
            @"variant=%@ cacheHit=%@ compile=PASS compileMs=%.1f load=FAIL loadMs=%.1f error=%@",
            label, cacheHit ? @"yes" : @"no", compileMS, loadMS,
            A12MPV2DescribeError(loadError)]];
        return nil;
    }

    NSDictionary *nameMap = nil;
    NSArray *procedures = nil;
    NSUInteger count = A12MPV2ProcedureCount(memoryModel, &nameMap, &procedures);
    [lines addObject:[NSString stringWithFormat:
        @"variant=%@ cacheHit=%@ compile=PASS compileMs=%.1f load=PASS loadMs=%.1f procedureCount=%lu names=%@ procedures=%lu",
        label, cacheHit ? @"yes" : @"no", compileMS, loadMS,
        (unsigned long)count, nameMap ?: @{}, (unsigned long)procedures.count]];
    return memoryModel;
}

static inline void A12MPV2Unload(id model,
                                 NSString *label,
                                 NSMutableArray<NSString *> *lines) {
    if (!model) return;
    NSError *error = nil;
    NSTimeInterval t0 = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError **))objc_msgSend)(
        model, NSSelectorFromString(@"unloadWithQoS:error:"), 25u, &error);
    double ms = (NSDate.timeIntervalSinceReferenceDate - t0) * 1000.0;
    [lines addObject:[NSString stringWithFormat:@"variant=%@ unload=%@ unloadMs=%.1f%@",
        label, ok ? @"PASS" : @"FAIL", ms,
        ok ? @"" : [NSString stringWithFormat:@" error=%@", A12MPV2DescribeError(error)]]];
}

static inline NSString *A12MPV2DispatchTwoProcedures(id memoryModel) {
    NSMutableArray<NSString *> *lines = [NSMutableArray array];
    NSDictionary *nameMap = nil;
    NSArray *procedures = nil;
    NSUInteger count = A12MPV2ProcedureCount(memoryModel, &nameMap, &procedures);
    if (count < 2) {
        return [NSString stringWithFormat:
            @"dispatch=SKIP reason=procedureCount<2 count=%lu names=%@",
            (unsigned long)count, nameMap ?: @{}];
    }

    id loweredModel = ((id(*)(id,SEL))objc_msgSend)(memoryModel,
        NSSelectorFromString(@"model"));
    Class requestClass = NSClassFromString(@"_ANERequest");
    Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!loweredModel || !requestClass || !surfaceClass) {
        return @"dispatch=FAIL stage=runtime-objects";
    }

    A12MP_IOSurfaceAPI io = A12MPGetIOSurfaceAPI();
    if (!io.ok) return @"dispatch=FAIL stage=iosurface-api";
    const NSUInteger elementCount = 64u * 128u;
    const NSUInteger surfaceBytes = elementCount * sizeof(uint16_t); // 16 KiB
    A12MP_IOSurfaceRef inputSurface = A12MPMakeSurface(io, surfaceBytes);
    A12MP_IOSurfaceRef outputSurface = A12MPMakeSurface(io, surfaceBytes);
    if (!inputSurface || !outputSurface) {
        if (inputSurface) CFRelease(inputSurface);
        if (outputSurface) CFRelease(outputSurface);
        return @"dispatch=FAIL stage=iosurface-create";
    }

    io.lock(inputSurface, 0, NULL);
    uint16_t *inputBits = (uint16_t *)io.baseAddress(inputSurface);
    for (NSUInteger i = 0; i < elementCount; ++i) inputBits[i] = 0x3C00u;
    io.unlock(inputSurface, 0, NULL);

    id inputObject = ((id(*)(Class,SEL,void *))objc_msgSend)(
        surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)inputSurface);
    id outputObject = ((id(*)(Class,SEL,void *))objc_msgSend)(
        surfaceClass, NSSelectorFromString(@"objectWithIOSurface:"), (void *)outputSurface);
    if (!inputObject || !outputObject) {
        CFRelease(inputSurface); CFRelease(outputSurface);
        return @"dispatch=FAIL stage=iosurface-wrap";
    }

    SEL requestSelector = NSSelectorFromString(
        @"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:");
    if (![requestClass respondsToSelector:requestSelector]) {
        CFRelease(inputSurface); CFRelease(outputSurface);
        return @"dispatch=FAIL stage=request-selector";
    }

    NSMutableArray<NSString *> *classes = [NSMutableArray arrayWithCapacity:2];
    for (unsigned int p = 0; p < 2; ++p) {
        NSArray *inputIndices = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
            loweredModel, NSSelectorFromString(@"inputSymbolIndicesForProcedureIndex:"), p);
        NSArray *outputIndices = ((id(*)(id,SEL,unsigned int))objc_msgSend)(
            loweredModel, NSSelectorFromString(@"outputSymbolIndicesForProcedureIndex:"), p);
        if (![inputIndices isKindOfClass:NSArray.class] || inputIndices.count != 1 ||
            ![outputIndices isKindOfClass:NSArray.class] || outputIndices.count != 1) {
            [lines addObject:[NSString stringWithFormat:
                @"procedure%u=FAIL stage=symbol-indices inputs=%@ outputs=%@",
                p, inputIndices, outputIndices]];
            continue;
        }

        io.lock(outputSurface, 0, NULL);
        memset(io.baseAddress(outputSurface), 0, surfaceBytes);
        io.unlock(outputSurface, 0, NULL);

        id request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            requestClass, requestSelector,
            @[inputObject], inputIndices, @[outputObject], outputIndices,
            nil, nil, @(p));
        if (!request) {
            [lines addObject:[NSString stringWithFormat:@"procedure%u=FAIL stage=request", p]];
            continue;
        }

        NSError *evalError = nil;
        NSTimeInterval t0 = NSDate.timeIntervalSinceReferenceDate;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError **))objc_msgSend)(
            memoryModel, NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
            25u, @{}, request, &evalError);
        double evalMS = (NSDate.timeIntervalSinceReferenceDate - t0) * 1000.0;
        if (!ok) {
            [lines addObject:[NSString stringWithFormat:
                @"procedure%u=FAIL stage=evaluate evalMs=%.1f error=%@",
                p, evalMS, A12MPV2DescribeError(evalError)]];
            continue;
        }

        io.lock(outputSurface, 0, NULL);
        const uint16_t *bits = (const uint16_t *)io.baseAddress(outputSurface);
        NSUInteger ones = 0, twos = 0;
        for (NSUInteger i = 0; i < elementCount; ++i) {
            ones += bits[i] == 0x3C00u;
            twos += bits[i] == 0x4000u;
        }
        io.unlock(outputSurface, 0, NULL);

        NSString *kind = @"other";
        if (ones == elementCount) kind = @"identity";
        else if (twos == elementCount) kind = @"double";
        [classes addObject:kind];
        [lines addObject:[NSString stringWithFormat:
            @"procedure%u=PASS evalMs=%.1f class=%@ ones=%lu twos=%lu/%lu indices=%@->%@",
            p, evalMS, kind, (unsigned long)ones, (unsigned long)twos,
            (unsigned long)elementCount, inputIndices, outputIndices]];
    }

    CFRelease(inputSurface);
    CFRelease(outputSurface);
    NSSet *observed = [NSSet setWithArray:classes];
    BOOL exactPair = observed.count == 2 &&
        [observed containsObject:@"identity"] && [observed containsObject:@"double"];
    [lines addObject:exactPair
        ? @"dispatch=PASS identity+double"
        : [NSString stringWithFormat:@"dispatch=FAIL classes=%@", classes]];
    return [lines componentsJoinedByString:@"\n"];
}

static inline NSString *A12MPV2Run(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE private multi-procedure MIL POC v2",
            @"control=single main vs main+double; same V5 weights; spatial 64->128",
            nil];

        void *aneHandle = dlopen(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
            RTLD_NOW | RTLD_LOCAL);
        if (!aneHandle) {
            [lines addObject:@"RESULT=FAIL stage=framework-load"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSURL *templateURL = [[NSBundle mainBundle] URLForResource:@"Conv2048W8Template"
                                                     withExtension:@"bundle"];
        NSURL *assetURL = [templateURL URLByAppendingPathComponent:@"V5TwoProcedure.mlmodelc"
                                                       isDirectory:YES];
        NSError *ioError = nil;
        NSString *source = [NSString stringWithContentsOfURL:
            [assetURL URLByAppendingPathComponent:@"model.mil"]
            encoding:NSUTF8StringEncoding error:&ioError];
        NSData *weights = [NSData dataWithContentsOfURL:
            [assetURL URLByAppendingPathComponent:@"weights/weight.bin"]
            options:0 error:&ioError];
        if (!source || !weights) {
            [lines addObject:[NSString stringWithFormat:@"RESULT=FAIL stage=fixture-read error=%@",
                A12MPV2DescribeError(ioError)]];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSString *patched = [source stringByReplacingOccurrencesOfString:@"[1, 64, 1, 64]"
                                                               withString:@"[1, 64, 1, 128]"];
        NSRange doubleStart = [patched rangeOfString:@"    func double<ios18>"];
        NSRange identityStart = [patched rangeOfString:@"    func identity<ios18>"];
        NSRange programClose = [patched rangeOfString:@"\n}" options:NSBackwardsSearch];
        if (doubleStart.location == NSNotFound || identityStart.location == NSNotFound ||
            programClose.location == NSNotFound || !(doubleStart.location < identityStart.location) ||
            !(identityStart.location < programClose.location)) {
            [lines addObject:@"RESULT=FAIL stage=fixture-split"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSString *header = [patched substringToIndex:doubleStart.location];
        NSString *doubleFunction = [patched substringWithRange:NSMakeRange(
            doubleStart.location, identityStart.location - doubleStart.location)];
        NSString *identityFunction = [patched substringWithRange:NSMakeRange(
            identityStart.location, programClose.location - identityStart.location)];
        identityFunction = [identityFunction stringByReplacingOccurrencesOfString:
            @"func identity<ios18>" withString:@"func main<ios18>"];

        NSString *singleText = [NSString stringWithFormat:@"%@%@\n}", header, identityFunction];
        NSString *multiText = [NSString stringWithFormat:@"%@%@%@\n}",
            header, identityFunction, doubleFunction];
        NSData *singleMIL = [singleText dataUsingEncoding:NSUTF8StringEncoding];
        NSData *multiMIL = [multiText dataUsingEncoding:NSUTF8StringEncoding];
        [lines addObject:[NSString stringWithFormat:
            @"sourceBytes=%lu singleBytes=%lu multiBytes=%lu weightsBytes=%lu",
            (unsigned long)[source lengthOfBytesUsingEncoding:NSUTF8StringEncoding],
            (unsigned long)singleMIL.length, (unsigned long)multiMIL.length,
            (unsigned long)weights.length]];

        id single = A12MPV2CompileAndLoad(singleMIL, weights, @"single-main", lines);
        if (!single) {
            [lines addObject:@"RESULT=INCONCLUSIVE route=in-memory-MIL baseline-single-main-failed"];
            return [lines componentsJoinedByString:@"\n"];
        }
        NSDictionary *singleNames = nil;
        NSArray *singleProcedures = nil;
        NSUInteger singleCount = A12MPV2ProcedureCount(single, &singleNames, &singleProcedures);
        A12MPV2Unload(single, @"single-main", lines);
        if (singleCount == 0) {
            [lines addObject:@"RESULT=INCONCLUSIVE route=in-memory-MIL single-main-loaded-but-no-procedure-metadata"];
            return [lines componentsJoinedByString:@"\n"];
        }

        id multi = A12MPV2CompileAndLoad(multiMIL, weights, @"main+double", lines);
        if (!multi) {
            [lines addObject:@"RESULT=ROUTE_BLOCKED single-main=PASS main+double=FAIL conclusion=raw-multifunction-MIL-rejected-by-ANEC-path"];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSDictionary *multiNames = nil;
        NSArray *multiProcedures = nil;
        NSUInteger multiCount = A12MPV2ProcedureCount(multi, &multiNames, &multiProcedures);
        if (multiCount < 2) {
            A12MPV2Unload(multi, @"main+double", lines);
            [lines addObject:[NSString stringWithFormat:
                @"RESULT=ROUTE_BLOCKED single-main=PASS main+double=PASS procedureCount=%lu conclusion=compiler-kept-only-one-procedure",
                (unsigned long)multiCount]];
            return [lines componentsJoinedByString:@"\n"];
        }

        NSString *dispatch = A12MPV2DispatchTwoProcedures(multi);
        [lines addObject:dispatch];
        BOOL dispatchPass = [dispatch rangeOfString:@"dispatch=PASS identity+double"].location != NSNotFound;
        A12MPV2Unload(multi, @"main+double", lines);
        [lines addObject:dispatchPass
            ? @"RESULT=PASS privateProcedures>=2 oneLoadedModel=yes procedureDispatch=identity+double"
            : @"RESULT=FAIL stage=procedure-dispatch"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

__attribute__((constructor))
static void A12MPV2InstallLaunchProbe(void) {
    if (NSProcessInfo.processInfo.environment[@"XCTestConfigurationFilePath"] != nil) return;
    // V1 fires at +1.0 s. Delay V2 slightly so its result block is not interleaved.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.8 * NSEC_PER_SEC)),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSString *result = A12MPV2Run();
            NSLog(@"\n========== ANIMAXS_ANE_MULTIPROC_POC_V2 ==========\n%@\n====================================================\n", result);
        }
    });
}

#endif // !TARGET_OS_SIMULATOR
