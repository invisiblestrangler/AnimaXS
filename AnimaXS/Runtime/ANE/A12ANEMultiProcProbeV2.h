#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// Experiment branch only.
//
// V1 established the private ANE/Espresso Objective-C API surface on the
// physical A12 device. This V2 probe is narrower: it mines the actual loaded
// framework binaries for embedded compiler/runtime vocabulary around
// multi-head / procedure support, and constructs only lightweight metadata
// objects (_ANEProcedureData / _ANEModelInstanceParameters) in memory.
//
// No model is compiled, loaded, evaluated, mapped, or unloaded here.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE targeted runtime probe v2\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12ProbeContains(NSString *value, NSString *needle) {
    if (!value || !needle) return NO;
    return [value rangeOfString:needle options:NSCaseInsensitiveSearch].location != NSNotFound;
}

static inline NSString *A12ProbeSafeDescription(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (NSException *exception) {
        return [NSString stringWithFormat:@"<description threw %@>", exception.name ?: @"exception"];
    }
}

static inline BOOL A12ProbeInterestingBinaryString(NSString *value) {
    NSArray<NSString *> *needles = @[
        @"multi_head", @"multihead", @"multi-head",
        @"procedure", @"procedurearray", @"procedure_data", @"proceduredata",
        @"anefmodel", @"anef_", @"anef.",
        @"inputnetwork", @"input_network", @"input networks", @"inputnetworks",
        @"program_gen", @"programgen", @"program generation",
        @"two_nets", @"two nets", @"multifunction", @"multi_function",
        @"networkdescription", @"network_description",
        @"procedure_name", @"procedureindex", @"procedure_index"
    ];
    for (NSString *needle in needles) {
        if (A12ProbeContains(value, needle)) return YES;
    }
    return NO;
}

static inline NSArray<NSString *> *A12ProbeFilteredStringsAtPath(NSString *path,
                                                                 NSString **errorText) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path
                                         options:NSDataReadingMappedIfSafe
                                           error:&error];
    if (!data) {
        if (errorText) *errorText = error.localizedDescription ?: @"read failed";
        return @[];
    }

    const uint8_t *bytes = (const uint8_t *)data.bytes;
    const NSUInteger length = data.length;
    NSUInteger start = NSNotFound;
    NSMutableOrderedSet<NSString *> *matches = [NSMutableOrderedSet orderedSet];

    for (NSUInteger i = 0; i <= length; ++i) {
        BOOL printable = NO;
        if (i < length) {
            const uint8_t c = bytes[i];
            printable = (c >= 0x20 && c <= 0x7e);
        }

        if (printable) {
            if (start == NSNotFound) start = i;
            continue;
        }

        if (start != NSNotFound) {
            const NSUInteger runLength = i - start;
            if (runLength >= 4 && runLength <= 1024) {
                NSString *candidate = [[NSString alloc] initWithBytes:bytes + start
                                                               length:runLength
                                                             encoding:NSASCIIStringEncoding];
                if (candidate && A12ProbeInterestingBinaryString(candidate)) {
                    [matches addObject:candidate];
                }
            }
            start = NSNotFound;
        }
    }

    NSArray<NSString *> *sorted = [[matches array] sortedArrayUsingSelector:@selector(compare:)];
    // Keep Xcode console output bounded while still returning far more than the
    // handful of strings we actually expect for these precise filters.
    if (sorted.count > 400) {
        sorted = [sorted subarrayWithRange:NSMakeRange(0, 400)];
    }
    return sorted;
}

static inline void A12ProbeAppendBinaryStrings(NSMutableArray<NSString *> *lines,
                                                NSString *label,
                                                NSString *path) {
    NSString *errorText = nil;
    NSArray<NSString *> *strings = A12ProbeFilteredStringsAtPath(path, &errorText);
    [lines addObject:[NSString stringWithFormat:@"-- BINARY STRINGS %@ count=%lu --",
        label, (unsigned long)strings.count]];
    if (errorText) {
        [lines addObject:[NSString stringWithFormat:@"readError=%@", errorText]];
        return;
    }
    for (NSString *value in strings) [lines addObject:value];
}

static inline void A12ProbeAppendProcedureMetadataObjects(NSMutableArray<NSString *> *lines) {
    [lines addObject:@"-- PROCEDURE METADATA OBJECT CONSTRUCTION --"];

    Class procedureClass = NSClassFromString(@"_ANEProcedureData");
    Class paramsClass = NSClassFromString(@"_ANEModelInstanceParameters");
    SEL procedureFactory = NSSelectorFromString(@"procedureDataWithSymbol:weightArray:");
    SEL paramsFactory = NSSelectorFromString(@"withProcedureData:procedureArray:");

    [lines addObject:[NSString stringWithFormat:@"classes procedureData=%@ modelInstanceParams=%@",
        procedureClass ? @"YES" : @"NO", paramsClass ? @"YES" : @"NO"]];

    if (!procedureClass || !paramsClass ||
        ![procedureClass respondsToSelector:procedureFactory] ||
        ![paramsClass respondsToSelector:paramsFactory]) {
        [lines addObject:@"metadataConstruction=SKIP missing class/selector"];
        return;
    }

    @try {
        id p0 = ((id(*)(Class,SEL,id,id))objc_msgSend)(
            procedureClass, procedureFactory, @"probe.identity", @[]);
        id p1 = ((id(*)(Class,SEL,id,id))objc_msgSend)(
            procedureClass, procedureFactory, @"probe.double", @[]);

        [lines addObject:[NSString stringWithFormat:@"p0 class=%@ desc=%@",
            p0 ? NSStringFromClass([p0 class]) : @"(nil)", A12ProbeSafeDescription(p0)]];
        [lines addObject:[NSString stringWithFormat:@"p1 class=%@ desc=%@",
            p1 ? NSStringFromClass([p1 class]) : @"(nil)", A12ProbeSafeDescription(p1)]];

        if (p0 && p1) {
            id params = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                paramsClass, paramsFactory, p0, @[p0, p1]);
            [lines addObject:[NSString stringWithFormat:@"params class=%@ desc=%@",
                params ? NSStringFromClass([params class]) : @"(nil)",
                A12ProbeSafeDescription(params)]];

            if (params) {
                SEL instanceNameSel = NSSelectorFromString(@"instanceName");
                SEL procedureArraySel = NSSelectorFromString(@"procedureArray");
                id instanceName = [params respondsToSelector:instanceNameSel]
                    ? ((id(*)(id,SEL))objc_msgSend)(params, instanceNameSel) : nil;
                id procedureArray = [params respondsToSelector:procedureArraySel]
                    ? ((id(*)(id,SEL))objc_msgSend)(params, procedureArraySel) : nil;
                [lines addObject:[NSString stringWithFormat:@"params.instanceName=%@",
                    A12ProbeSafeDescription(instanceName)]];
                [lines addObject:[NSString stringWithFormat:@"params.procedureArray=%@",
                    A12ProbeSafeDescription(procedureArray)]];
            }
        }
        [lines addObject:@"metadataConstruction=PASS"];
    } @catch (NSException *exception) {
        [lines addObject:[NSString stringWithFormat:@"metadataConstruction=EXCEPTION name=%@ reason=%@",
            exception.name ?: @"?", exception.reason ?: @"?"]];
    }
}

static inline void A12ProbeAppendKeyRuntimeFacts(NSMutableArray<NSString *> *lines) {
    [lines addObject:@"-- KEY RUNTIME SELECTOR PRESENCE --"];
    NSDictionary<NSString *, NSArray<NSString *> *> *selectors = @{
        @"_ANEModel": @[
            @"procedureInfoForProcedureIndex:",
            @"inputSymbolIndicesForProcedureIndex:",
            @"outputSymbolIndicesForProcedureIndex:"
        ],
        @"_ANEClient": @[
            @"loadModelNewInstance:options:modelInstParams:qos:error:",
            @"prepareChainingWithModel:options:chainingReq:qos:error:"
        ],
        @"_ANEInMemoryModelDescriptor": @[
            @"modelWithNetworkDescription:weights:optionsPlist:",
            @"modelWithMILText:weights:optionsPlist:"
        ]
    };

    for (NSString *className in [[selectors allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
        Class cls = NSClassFromString(className);
        for (NSString *selectorName in selectors[className]) {
            SEL sel = NSSelectorFromString(selectorName);
            BOOL present = NO;
            if (cls) {
                if ([selectorName hasPrefix:@"modelWith"])
                    present = class_getClassMethod(cls, sel) != NULL;
                else
                    present = class_getInstanceMethod(cls, sel) != NULL;
            }
            [lines addObject:[NSString stringWithFormat:@"%@ %@=%@",
                className, selectorName, present ? @"YES" : @"NO"]];
        }
    }

    NSArray<NSString *> *passClasses = @[
        @"EspressoPass_multi_head_program_gen",
        @"EspressoPass_multi_head_prune_undeclared",
        @"EspressoPass_style_transfer_two_nets",
        @"EspressoPass_style_transfer_two_nets_onlyanepart"
    ];
    for (NSString *className in passClasses) {
        [lines addObject:[NSString stringWithFormat:@"class %@=%@",
            className, NSClassFromString(className) ? @"YES" : @"NO"]];
    }
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE targeted runtime probe v2",
            @"mode=binary vocabulary + lightweight metadata objects; no compile/load/evaluate",
            nil];

        NSString *anePath = @"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
        NSString *espressoPath = @"/System/Library/PrivateFrameworks/Espresso.framework/Espresso";
        void *aneHandle = dlopen(anePath.UTF8String, RTLD_NOW | RTLD_LOCAL);
        void *espressoHandle = dlopen(espressoPath.UTF8String, RTLD_NOW | RTLD_LOCAL);
        [lines addObject:[NSString stringWithFormat:@"dlopen AppleNeuralEngine=%@ Espresso=%@",
            aneHandle ? @"PASS" : @"FAIL", espressoHandle ? @"PASS" : @"FAIL"]];

        A12ProbeAppendKeyRuntimeFacts(lines);
        A12ProbeAppendProcedureMetadataObjects(lines);
        A12ProbeAppendBinaryStrings(lines, @"AppleNeuralEngine", anePath);
        A12ProbeAppendBinaryStrings(lines, @"Espresso", espressoPath);

        [lines addObject:@"RESULT=PASS targeted-runtime-v2"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
