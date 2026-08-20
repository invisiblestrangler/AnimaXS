#import <Foundation/Foundation.h>
#import "A12ANEBridge.h"

// Experimental production bridge for the device-proven Stage2J/K 10-procedure
// representation. This intentionally reuses the validated experiment helpers
// while the old eight-model production backend remains untouched for A/B.
// Once the physical-device A/B is accepted, these helpers can be moved out of
// Experiment/ without changing the Swift scheduler API below.
#import "../../../Experiment/A12ANEMultiProcResidencyStage2MMemSafe.h"

#if !TARGET_OS_SIMULATOR

static inline NSString *A12MPErrorText(NSArray<NSString *> *detail, NSString *fallback) {
    if (detail.count) {
        NSUInteger start = detail.count > 6 ? detail.count - 6 : 0;
        return [[detail subarrayWithRange:NSMakeRange(start, detail.count - start)] componentsJoinedByString:@" | "];
    }
    return fallback ?: @"multiprocedure ANE operation failed";
}

static inline NSMutableDictionary * _Nullable A12ANEMultiProcCreateLoadedHandle(
    NSUInteger block,
    double * _Nullable compileMSOut,
    double * _Nullable loadMSOut,
    BOOL * _Nullable cacheHitOut) {
    @autoreleasepool {
        if (block >= 28) return nil;
        NSMutableArray<NSString *> *detail = [NSMutableArray array];
        NSString *runtime = A12S2GCurrentANEIRVersion(detail);
        NSURL *root = [NSURL fileURLWithPath:[NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSString stringWithFormat:@"AnimaXS-ANE-MP-PROD-b%02lu-%@",
                (unsigned long)block, NSUUID.UUID.UUIDString]] isDirectory:YES];
        NSMutableDictionary *plist = nil;
        NSMutableDictionary *weights = nil;
        if (!A12S2NBuildBlock(block, root, runtime, detail, &plist, &weights)) {
            [NSFileManager.defaultManager removeItemAtURL:root error:nil];
            return nil;
        }

        id lowered = nil;
        NSDictionary *names = nil;
        double compileMS = 0.0, loadMS = 0.0;
        BOOL cacheHit = NO;
        NSString *label = [NSString stringWithFormat:@"prod-mp-b%02lu", (unsigned long)block];
        id model = A12S2LLoadTenProcedureModel(
            plist, weights, label, detail,
            &lowered, &names, &compileMS, &loadMS, &cacheHit);
        BOOL valid = model && lowered && A12S2MNameMapHasAllTen(names, detail, label);
        [NSFileManager.defaultManager removeItemAtURL:root error:nil];
        if (!valid) {
            if (model) {
                double ignored = 0.0;
                A12S2LUnloadModel(model, &ignored);
            }
            return nil;
        }

        NSMutableDictionary *state = [NSMutableDictionary dictionaryWithDictionary:@{
            @"block": @(block),
            @"model": model,
            @"lowered": lowered,
            @"names": names,
            @"loaded": @YES,
            @"lastError": @"",
        }];
        if (compileMSOut) *compileMSOut = compileMS;
        if (loadMSOut) *loadMSOut = loadMS;
        if (cacheHitOut) *cacheHitOut = cacheHit;
        A12S2MMemRelief();
        return state;
    }
}

static inline BOOL A12ANEMultiProcHandleIsLoaded(NSMutableDictionary * _Nullable handle) {
    return handle && [handle[@"loaded"] boolValue];
}

static inline NSString *A12ANEMultiProcHandleLastError(NSMutableDictionary * _Nullable handle) {
    NSString *value = [handle[@"lastError"] isKindOfClass:NSString.class] ? handle[@"lastError"] : nil;
    return value ?: @"multiprocedure ANE operation failed";
}

static inline void A12MPSetError(NSMutableDictionary *handle, NSArray<NSString *> *detail, NSString *fallback) {
    if (!handle) return;
    handle[@"lastError"] = A12MPErrorText(detail, fallback);
}

static inline BOOL A12ANEMultiProcLoadHandle(
    NSMutableDictionary * _Nullable handle,
    double * _Nullable loadMSOut) {
    if (!handle) return NO;
    if ([handle[@"loaded"] boolValue]) {
        if (loadMSOut) *loadMSOut = 0.0;
        return YES;
    }
    id model = handle[@"model"];
    if (!model) return NO;
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    NSDictionary *names = nil;
    double loadMS = 0.0;
    id lowered = A12S2MReloadHandle(
        model,
        [NSString stringWithFormat:@"prod-mp-reload-b%02lu", (unsigned long)[handle[@"block"] unsignedIntegerValue]],
        detail, &names, &loadMS);
    if (!lowered || !A12S2MNameMapHasAllTen(names, detail, @"prod-mp-reload")) {
        A12MPSetError(handle, detail, @"multiprocedure reload failed");
        if (loadMSOut) *loadMSOut = loadMS;
        return NO;
    }
    handle[@"lowered"] = lowered;
    handle[@"names"] = names;
    handle[@"loaded"] = @YES;
    handle[@"lastError"] = @"";
    if (loadMSOut) *loadMSOut = loadMS;
    return YES;
}

static inline BOOL A12ANEMultiProcUnloadHandle(
    NSMutableDictionary * _Nullable handle,
    double * _Nullable unloadMSOut) {
    if (!handle) return YES;
    if (![handle[@"loaded"] boolValue]) {
        if (unloadMSOut) *unloadMSOut = 0.0;
        return YES;
    }
    id model = handle[@"model"];
    double unloadMS = 0.0;
    BOOL unloaded = A12S2LUnloadModel(model, &unloadMS);
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    BOOL trimmed = unloaded && A12S2MTrim(model, 3u, detail, @"prod-mp-trim", NO);
    if (unloadMSOut) *unloadMSOut = unloadMS;
    if (!unloaded || !trimmed) {
        A12MPSetError(handle, detail, unloaded ? @"multiprocedure trim failed" : @"multiprocedure unload failed");
        return NO;
    }
    [handle removeObjectForKey:@"lowered"];
    [handle removeObjectForKey:@"names"];
    handle[@"loaded"] = @NO;
    handle[@"lastError"] = @"";
    A12S2MMemRelief();
    return YES;
}

static inline BOOL A12ANEMultiProcEvaluateHandle(
    NSMutableDictionary * _Nullable handle,
    NSString *procedureName,
    A12ANESurface *input,
    A12ANESurface *output,
    double * _Nullable evalMSOut) {
    if (!handle || ![handle[@"loaded"] boolValue]) return NO;
    id model = handle[@"model"];
    id lowered = handle[@"lowered"];
    NSDictionary *names = [handle[@"names"] isKindOfClass:NSDictionary.class] ? handle[@"names"] : nil;
    NSNumber *pid = names ? A12S2KProcedureID(names, procedureName) : nil;
    if (!model || !lowered || !pid || !input || !output) {
        handle[@"lastError"] = [NSString stringWithFormat:@"invalid procedure dispatch %@", procedureName ?: @"(nil)"];
        return NO;
    }
    NSMutableArray<NSString *> *detail = [NSMutableArray array];
    NSTimeInterval start = NSDate.timeIntervalSinceReferenceDate;
    BOOL ok = A12S2KEvaluateProcedure(
        model, lowered, pid.unsignedIntegerValue,
        input, output, procedureName, detail);
    double evalMS = (NSDate.timeIntervalSinceReferenceDate - start) * 1000.0;
    if (evalMSOut) *evalMSOut = evalMS;
    if (!ok) {
        A12MPSetError(handle, detail, [NSString stringWithFormat:@"procedure %@ failed", procedureName]);
        return NO;
    }
    handle[@"lastError"] = @"";
    return YES;
}

static inline void A12ANEMultiProcDestroyHandle(NSMutableDictionary * _Nullable handle) {
    if (!handle) return;
    double ignored = 0.0;
    A12ANEMultiProcUnloadHandle(handle, &ignored);
    [handle removeAllObjects];
    A12S2MMemRelief();
}

#else

static inline NSMutableDictionary * _Nullable A12ANEMultiProcCreateLoadedHandle(
    NSUInteger block, double * _Nullable compileMSOut,
    double * _Nullable loadMSOut, BOOL * _Nullable cacheHitOut) {
    return nil;
}
static inline BOOL A12ANEMultiProcHandleIsLoaded(NSMutableDictionary * _Nullable handle) { return NO; }
static inline NSString *A12ANEMultiProcHandleLastError(NSMutableDictionary * _Nullable handle) { return @"ANE unavailable in simulator"; }
static inline BOOL A12ANEMultiProcLoadHandle(NSMutableDictionary * _Nullable handle, double * _Nullable loadMSOut) { return NO; }
static inline BOOL A12ANEMultiProcUnloadHandle(NSMutableDictionary * _Nullable handle, double * _Nullable unloadMSOut) { return YES; }
static inline BOOL A12ANEMultiProcEvaluateHandle(NSMutableDictionary * _Nullable handle, NSString *procedureName, A12ANESurface *input, A12ANESurface *output, double * _Nullable evalMSOut) { return NO; }
static inline void A12ANEMultiProcDestroyHandle(NSMutableDictionary * _Nullable handle) {}

#endif
