#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#import <stdlib.h>

// Experiment branch only.
//
// The previous V2 raw-MIL compiler probe established that even its single-main
// control is rejected by _ANEInMemoryModel with InvalidMILProgram. Do not infer
// anything further about multi-procedure support from that compiler path.
//
// This probe instead asks the physical device's Objective-C runtime what private
// AppleNeuralEngine / Espresso API surface actually exists. It is intentionally
// read-only: no model is compiled, loaded, evaluated, or unloaded.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANEPrivateRuntimeInventory(void) {
    return @"ANE private runtime inventory v1\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12InvContains(NSString *value, NSString *needle) {
    if (!value || !needle) return NO;
    return [value rangeOfString:needle options:NSCaseInsensitiveSearch].location != NSNotFound;
}

static inline BOOL A12InvRelevantClass(NSString *name, NSString *image) {
    if (A12InvContains(image, @"AppleNeuralEngine.framework") ||
        A12InvContains(image, @"Espresso.framework")) return YES;
    return A12InvContains(name, @"_ANE") ||
           A12InvContains(name, @"ANEF") ||
           A12InvContains(name, @"Espresso") ||
           A12InvContains(name, @"NeuralEngine");
}

static inline BOOL A12InvDetailClass(NSString *name) {
    NSArray<NSString *> *needles = @[
        @"Procedure", @"Program", @"Function", @"Model", @"Client",
        @"Request", @"Compiler", @"Espresso", @"Network", @"Descriptor",
        @"Instance", @"Chaining", @"InMemory", @"ANEF", @"IRTranslator",
        @"IOSurface", @"Parameter"
    ];
    for (NSString *needle in needles) if (A12InvContains(name, needle)) return YES;
    return NO;
}

static inline BOOL A12InvInterestingSelector(NSString *selector) {
    NSArray<NSString *> *needles = @[
        @"procedure", @"program", @"function", @"network", @"anef",
        @"espresso", @"compile", @"model", @"instance", @"request",
        @"chain", @"descriptor", @"mil", @"load", @"unload", @"evaluate",
        @"symbol", @"parameter", @"weight"
    ];
    for (NSString *needle in needles) if (A12InvContains(selector, needle)) return YES;
    return NO;
}

static inline NSArray<NSString *> *A12InvSortedMethods(Class cls, BOOL classMethods) {
    if (!cls) return @[];
    Class target = classMethods ? object_getClass(cls) : cls;
    unsigned int count = 0;
    Method *methods = class_copyMethodList(target, &count);
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        SEL sel = method_getName(methods[i]);
        const char *types = method_getTypeEncoding(methods[i]);
        [out addObject:[NSString stringWithFormat:@"%@ %@ types=%s",
            classMethods ? @"+" : @"-",
            NSStringFromSelector(sel),
            types ?: ""]];
    }
    free(methods);
    [out sortUsingSelector:@selector(compare:)];
    return out;
}

static inline NSArray<NSString *> *A12InvSortedIvars(Class cls) {
    if (!cls) return @[];
    unsigned int count = 0;
    Ivar *ivars = class_copyIvarList(cls, &count);
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        const char *name = ivar_getName(ivars[i]);
        const char *type = ivar_getTypeEncoding(ivars[i]);
        ptrdiff_t offset = ivar_getOffset(ivars[i]);
        [out addObject:[NSString stringWithFormat:@"ivar %s offset=%td type=%s",
            name ?: "?", offset, type ?: ""]];
    }
    free(ivars);
    [out sortUsingSelector:@selector(compare:)];
    return out;
}

static inline NSArray<NSString *> *A12InvSortedProperties(Class cls) {
    if (!cls) return @[];
    unsigned int count = 0;
    objc_property_t *properties = class_copyPropertyList(cls, &count);
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        const char *name = property_getName(properties[i]);
        const char *attrs = property_getAttributes(properties[i]);
        [out addObject:[NSString stringWithFormat:@"property %s attrs=%s",
            name ?: "?", attrs ?: ""]];
    }
    free(properties);
    [out sortUsingSelector:@selector(compare:)];
    return out;
}

static inline NSArray<NSString *> *A12InvSortedProtocols(Class cls) {
    if (!cls) return @[];
    unsigned int count = 0;
    Protocol * __unsafe_unretained *protocols = class_copyProtocolList(cls, &count);
    NSMutableArray<NSString *> *out = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; ++i) {
        const char *name = protocol_getName(protocols[i]);
        if (name) [out addObject:[NSString stringWithUTF8String:name]];
    }
    free(protocols);
    [out sortUsingSelector:@selector(compare:)];
    return out;
}

static inline NSString *A12ANEPrivateRuntimeInventory(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE private runtime inventory v1",
            @"mode=read-only Objective-C runtime inventory; no compile/load/evaluate",
            nil];

        const char *anePath = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
        const char *espressoPath = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso";
        void *aneHandle = dlopen(anePath, RTLD_NOW | RTLD_LOCAL);
        void *espressoHandle = dlopen(espressoPath, RTLD_NOW | RTLD_LOCAL);
        [lines addObject:[NSString stringWithFormat:@"dlopen AppleNeuralEngine=%@ Espresso=%@",
            aneHandle ? @"PASS" : @"FAIL", espressoHandle ? @"PASS" : @"FAIL"]];

        int total = objc_getClassList(NULL, 0);
        __unsafe_unretained Class *classes = total > 0
            ? (__unsafe_unretained Class *)malloc(sizeof(Class) * (NSUInteger)total)
            : NULL;
        int copied = classes ? objc_getClassList(classes, total) : 0;

        NSMutableDictionary<NSString *, NSValue *> *relevant = [NSMutableDictionary dictionary];
        for (int i = 0; i < copied; ++i) {
            Class cls = classes[i];
            const char *rawName = class_getName(cls);
            const char *rawImage = class_getImageName(cls);
            NSString *name = rawName ? [NSString stringWithUTF8String:rawName] : @"?";
            NSString *image = rawImage ? [NSString stringWithUTF8String:rawImage] : @"";
            if (A12InvRelevantClass(name, image)) {
                relevant[name] = [NSValue valueWithPointer:(__bridge const void *)(cls)];
            }
        }
        free(classes);

        NSArray<NSString *> *names = [[relevant allKeys] sortedArrayUsingSelector:@selector(compare:)];
        [lines addObject:[NSString stringWithFormat:@"runtimeClassCount=%d relevantClassCount=%lu",
            copied, (unsigned long)names.count]];
        [lines addObject:@"-- RELEVANT CLASSES --"];
        for (NSString *name in names) {
            Class cls = (__bridge Class)[relevant[name] pointerValue];
            const char *rawImage = class_getImageName(cls);
            NSString *image = rawImage ? [NSString stringWithUTF8String:rawImage] : @"?";
            [lines addObject:[NSString stringWithFormat:@"CLASS %@ image=%@", name, image]];
        }

        [lines addObject:@"-- DETAILED CLASS METADATA --"];
        for (NSString *name in names) {
            if (!A12InvDetailClass(name)) continue;
            Class cls = (__bridge Class)[relevant[name] pointerValue];
            Class superCls = class_getSuperclass(cls);
            const char *rawImage = class_getImageName(cls);
            NSString *image = rawImage ? [NSString stringWithUTF8String:rawImage] : @"?";
            [lines addObject:[NSString stringWithFormat:
                @"BEGIN_CLASS %@ super=%@ instanceSize=%lu image=%@",
                name,
                superCls ? NSStringFromClass(superCls) : @"(none)",
                (unsigned long)class_getInstanceSize(cls),
                image]];

            NSArray<NSString *> *protocols = A12InvSortedProtocols(cls);
            if (protocols.count) [lines addObject:[@"protocols=" stringByAppendingString:[protocols componentsJoinedByString:@","]]];
            for (NSString *entry in A12InvSortedIvars(cls)) [lines addObject:entry];
            for (NSString *entry in A12InvSortedProperties(cls)) [lines addObject:entry];
            for (NSString *entry in A12InvSortedMethods(cls, YES)) [lines addObject:entry];
            for (NSString *entry in A12InvSortedMethods(cls, NO)) [lines addObject:entry];
            [lines addObject:[@"END_CLASS " stringByAppendingString:name]];
        }

        [lines addObject:@"-- INTERESTING SELECTOR INDEX ACROSS ALL RELEVANT CLASSES --"];
        NSMutableArray<NSString *> *selectorIndex = [NSMutableArray array];
        for (NSString *name in names) {
            Class cls = (__bridge Class)[relevant[name] pointerValue];
            for (NSNumber *classSide in @[@YES, @NO]) {
                BOOL isClass = classSide.boolValue;
                Class target = isClass ? object_getClass(cls) : cls;
                unsigned int methodCount = 0;
                Method *methods = class_copyMethodList(target, &methodCount);
                for (unsigned int i = 0; i < methodCount; ++i) {
                    NSString *selector = NSStringFromSelector(method_getName(methods[i]));
                    if (!A12InvInterestingSelector(selector)) continue;
                    const char *types = method_getTypeEncoding(methods[i]);
                    [selectorIndex addObject:[NSString stringWithFormat:@"%@ %@ %@ types=%s",
                        name, isClass ? @"+" : @"-", selector, types ?: ""]];
                }
                free(methods);
            }
        }
        [selectorIndex sortUsingSelector:@selector(compare:)];
        for (NSString *entry in selectorIndex) [lines addObject:entry];

        [lines addObject:@"-- KNOWN PRIVATE CLASS PRESENCE --"];
        NSArray<NSString *> *knownClasses = @[
            @"_ANEModel", @"_ANEClient", @"_ANERequest", @"_ANEProcedureData",
            @"_ANEModelInstanceParameters", @"_ANEInMemoryModel",
            @"_ANEInMemoryModelDescriptor", @"_ANECompiler",
            @"_ANEEspressoIRTranslator", @"_ANEIOSurfaceObject"
        ];
        for (NSString *name in knownClasses) {
            Class cls = NSClassFromString(name);
            const char *rawImage = cls ? class_getImageName(cls) : NULL;
            [lines addObject:[NSString stringWithFormat:@"%@=%@ image=%@",
                name, cls ? @"YES" : @"NO",
                rawImage ? [NSString stringWithUTF8String:rawImage] : @"-"]];
        }

        [lines addObject:@"-- DLSYM PRESENCE PROBES --"];
        const char *symbols[] = {
            "ANECCompile",
            "ANEFModelProcedures",
            "kANEFModelProcedureNameToIDMapKey",
            "kANEFModelProceduresKey",
            "kANEFModelDescription",
            "kANEModelKeyEspressoTranslationOptions",
            "kANEFModelIdentityStrKey"
        };
        size_t symbolCount = sizeof(symbols) / sizeof(symbols[0]);
        for (size_t i = 0; i < symbolCount; ++i) {
            void *ptr = aneHandle ? dlsym(aneHandle, symbols[i]) : NULL;
            if (!ptr && espressoHandle) ptr = dlsym(espressoHandle, symbols[i]);
            [lines addObject:[NSString stringWithFormat:@"symbol %s=%@", symbols[i], ptr ? @"YES" : @"NO"]];
        }

        [lines addObject:[NSString stringWithFormat:
            @"RESULT=PASS classes=%lu indexedSelectors=%lu",
            (unsigned long)names.count, (unsigned long)selectorIndex.count]];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
