#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach-o/dyld.h>
#import <mach-o/loader.h>
#import <stdint.h>
#import <string.h>

// Experiment branch only. Read-only runtime introspection.
// No model compile/load/evaluate/mapping is performed here.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE targeted runtime probe v3\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12V3Contains(NSString *value, NSString *needle) {
    if (!value || !needle) return NO;
    return [value rangeOfString:needle options:NSCaseInsensitiveSearch].location != NSNotFound;
}

static inline NSString *A12V3Desc(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (NSException *e) {
        return [NSString stringWithFormat:@"<description threw %@: %@>", e.name ?: @"?", e.reason ?: @"?"];
    }
}

static inline BOOL A12V3InterestingString(NSString *value) {
    NSArray<NSString *> *needles = @[
        @"multi_head", @"multihead", @"multi-head", @"multi head",
        @"procedure", @"procedurearray", @"procedure_data", @"proceduredata",
        @"anefmodel", @"anef_", @"anef.",
        @"inputnetwork", @"input_network", @"input networks", @"inputnetworks",
        @"program_gen", @"programgen", @"program generation",
        @"two_nets", @"two nets", @"multifunction", @"multi_function",
        @"networkdescription", @"network_description", @"network description",
        @"procedure_name", @"procedurename", @"procedureindex", @"procedure_index",
        @"espresso translation", @"translationoptions", @"translation_options"
    ];
    for (NSString *needle in needles) if (A12V3Contains(value, needle)) return YES;
    return NO;
}

static inline void A12V3ScanBytes(const uint8_t *bytes,
                                  size_t length,
                                  NSString *sectionLabel,
                                  NSMutableOrderedSet<NSString *> *matches) {
    if (!bytes || length == 0) return;
    // Avoid pathological scans if a future OS maps a giant const section.
    if (length > (64u * 1024u * 1024u)) length = 64u * 1024u * 1024u;

    size_t start = SIZE_MAX;
    for (size_t i = 0; i <= length; ++i) {
        BOOL printable = NO;
        if (i < length) {
            uint8_t c = bytes[i];
            printable = (c >= 0x20 && c <= 0x7e);
        }
        if (printable) {
            if (start == SIZE_MAX) start = i;
            continue;
        }
        if (start != SIZE_MAX) {
            size_t run = i - start;
            if (run >= 4 && run <= 1024) {
                NSString *candidate = [[NSString alloc] initWithBytes:bytes + start
                                                               length:run
                                                             encoding:NSASCIIStringEncoding];
                if (candidate && A12V3InterestingString(candidate)) {
                    [matches addObject:[NSString stringWithFormat:@"%@ :: %@", sectionLabel, candidate]];
                }
            }
            start = SIZE_MAX;
        }
    }
}

static inline void A12V3AppendMappedImageStrings(NSMutableArray<NSString *> *lines,
                                                  NSString *imageNeedle,
                                                  NSString *label) {
    const struct mach_header_64 *header = NULL;
    intptr_t slide = 0;
    NSString *resolvedPath = nil;

    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        const char *raw = _dyld_get_image_name(i);
        if (!raw) continue;
        NSString *path = [NSString stringWithUTF8String:raw];
        if (!A12V3Contains(path, imageNeedle)) continue;
        const struct mach_header *h = _dyld_get_image_header(i);
        if (!h || h->magic != MH_MAGIC_64) continue;
        header = (const struct mach_header_64 *)h;
        slide = _dyld_get_image_vmaddr_slide(i);
        resolvedPath = path;
        break;
    }

    [lines addObject:[NSString stringWithFormat:@"-- MAPPED STRINGS %@ --", label]];
    if (!header) {
        [lines addObject:@"mappedImage=NOT_FOUND"];
        return;
    }
    [lines addObject:[NSString stringWithFormat:@"mappedImage=FOUND path=%@ slide=%lld",
        resolvedPath ?: @"?", (long long)slide]];

    NSMutableOrderedSet<NSString *> *matches = [NSMutableOrderedSet orderedSet];
    const uint8_t *cursor = (const uint8_t *)(header + 1);
    for (uint32_t commandIndex = 0; commandIndex < header->ncmds; ++commandIndex) {
        const struct load_command *lc = (const struct load_command *)cursor;
        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(struct segment_command_64)) {
            const struct segment_command_64 *seg = (const struct segment_command_64 *)lc;
            const struct section_64 *sections = (const struct section_64 *)(seg + 1);
            for (uint32_t s = 0; s < seg->nsects; ++s) {
                const struct section_64 *sec = &sections[s];
                NSString *sectName = [NSString stringWithUTF8String:sec->sectname] ?: @"?";
                BOOL scan = [sectName isEqualToString:@"__cstring"] ||
                            [sectName isEqualToString:@"__objc_methname"] ||
                            [sectName isEqualToString:@"__objc_classname"] ||
                            [sectName isEqualToString:@"__const"];
                if (!scan || sec->size == 0) continue;
                uintptr_t address = (uintptr_t)(sec->addr + (uint64_t)slide);
                NSString *sectionLabel = [NSString stringWithFormat:@"%s/%s", seg->segname, sec->sectname];
                A12V3ScanBytes((const uint8_t *)address, (size_t)sec->size, sectionLabel, matches);
            }
        }
        if (lc->cmdsize == 0) break;
        cursor += lc->cmdsize;
    }

    NSArray<NSString *> *sorted = [[matches array] sortedArrayUsingSelector:@selector(compare:)];
    [lines addObject:[NSString stringWithFormat:@"interestingStringCount=%lu", (unsigned long)sorted.count]];
    NSUInteger limit = MIN((NSUInteger)500, sorted.count);
    for (NSUInteger i = 0; i < limit; ++i) [lines addObject:sorted[i]];
    if (sorted.count > limit) [lines addObject:[NSString stringWithFormat:@"... truncated %lu entries", (unsigned long)(sorted.count - limit)]];
}

static inline void A12V3AppendExportedKey(NSMutableArray<NSString *> *lines,
                                          void *aneHandle,
                                          void *espressoHandle,
                                          const char *symbol) {
    void *ptr = aneHandle ? dlsym(aneHandle, symbol) : NULL;
    if (!ptr && espressoHandle) ptr = dlsym(espressoHandle, symbol);
    if (!ptr) {
        [lines addObject:[NSString stringWithFormat:@"export %s=ABSENT", symbol]];
        return;
    }

    id value = nil;
    @try {
        value = *(__unsafe_unretained id *)ptr;
    } @catch (__unused NSException *e) {
        value = nil;
    }
    [lines addObject:[NSString stringWithFormat:@"export %s=YES value=%@ class=%@",
        symbol,
        A12V3Desc(value),
        value ? NSStringFromClass([value class]) : @"(nil)"]];
}

static inline void A12V3AppendMetadataFactoryVariants(NSMutableArray<NSString *> *lines) {
    [lines addObject:@"-- MODEL INSTANCE PARAMETER FACTORY SEMANTICS --"];

    Class procedureClass = NSClassFromString(@"_ANEProcedureData");
    Class paramsClass = NSClassFromString(@"_ANEModelInstanceParameters");
    SEL procedureFactory = NSSelectorFromString(@"procedureDataWithSymbol:weightArray:");
    SEL paramsFactory = NSSelectorFromString(@"withProcedureData:procedureArray:");

    if (!procedureClass || !paramsClass ||
        ![procedureClass respondsToSelector:procedureFactory] ||
        ![paramsClass respondsToSelector:paramsFactory]) {
        [lines addObject:@"factorySemantics=SKIP missing private API"];
        return;
    }

    @try {
        id p0 = ((id(*)(Class,SEL,id,id))objc_msgSend)(procedureClass, procedureFactory, @"probe.identity", @[]);
        id p1 = ((id(*)(Class,SEL,id,id))objc_msgSend)(procedureClass, procedureFactory, @"probe.double", @[]);
        NSArray *array = (p0 && p1) ? @[p0, p1] : @[];
        [lines addObject:[NSString stringWithFormat:@"p0=%@", A12V3Desc(p0)]];
        [lines addObject:[NSString stringWithFormat:@"p1=%@", A12V3Desc(p1)]];

        NSArray *firstArgs = p0 ? @[p0, @"probe.instance"] : @[@"probe.instance"];
        for (id firstArg in firstArgs) {
            @try {
                id params = ((id(*)(Class,SEL,id,id))objc_msgSend)(paramsClass, paramsFactory, firstArg, array);
                SEL instanceNameSel = NSSelectorFromString(@"instanceName");
                SEL procedureArraySel = NSSelectorFromString(@"procedureArray");
                id instanceName = params && [params respondsToSelector:instanceNameSel]
                    ? ((id(*)(id,SEL))objc_msgSend)(params, instanceNameSel) : nil;
                id procedureArray = params && [params respondsToSelector:procedureArraySel]
                    ? ((id(*)(id,SEL))objc_msgSend)(params, procedureArraySel) : nil;
                [lines addObject:[NSString stringWithFormat:
                    @"firstArgClass=%@ firstArg=%@ => params=%@ instanceName=%@ procedureArray=%@",
                    NSStringFromClass([firstArg class]), A12V3Desc(firstArg), A12V3Desc(params),
                    A12V3Desc(instanceName), A12V3Desc(procedureArray)]];
            } @catch (NSException *inner) {
                [lines addObject:[NSString stringWithFormat:@"firstArg=%@ => EXCEPTION %@ %@",
                    A12V3Desc(firstArg), inner.name ?: @"?", inner.reason ?: @"?"]];
            }
        }
        [lines addObject:@"factorySemantics=PASS"];
    } @catch (NSException *e) {
        [lines addObject:[NSString stringWithFormat:@"factorySemantics=EXCEPTION %@ %@",
            e.name ?: @"?", e.reason ?: @"?"]];
    }
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE targeted runtime probe v3",
            @"mode=dyld mapped-string vocabulary + exported constants + metadata semantics; no compile/load/evaluate",
            nil];

        const char *anePath = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
        const char *espressoPath = "/System/Library/PrivateFrameworks/Espresso.framework/Espresso";
        void *aneHandle = dlopen(anePath, RTLD_NOW | RTLD_LOCAL);
        void *espressoHandle = dlopen(espressoPath, RTLD_NOW | RTLD_LOCAL);
        [lines addObject:[NSString stringWithFormat:@"dlopen AppleNeuralEngine=%@ Espresso=%@",
            aneHandle ? @"PASS" : @"FAIL", espressoHandle ? @"PASS" : @"FAIL"]];

        [lines addObject:@"-- EXPORTED ANEF / ESPRESSO KEY VALUES --"];
        A12V3AppendExportedKey(lines, aneHandle, espressoHandle, "kANEFModelProcedureNameToIDMapKey");
        A12V3AppendExportedKey(lines, aneHandle, espressoHandle, "kANEModelKeyEspressoTranslationOptions");
        A12V3AppendExportedKey(lines, aneHandle, espressoHandle, "kANEFModelIdentityStrKey");
        A12V3AppendExportedKey(lines, aneHandle, espressoHandle, "kANEFModelProceduresKey");
        A12V3AppendExportedKey(lines, aneHandle, espressoHandle, "kANEFModelDescription");

        A12V3AppendMetadataFactoryVariants(lines);
        A12V3AppendMappedImageStrings(lines, @"AppleNeuralEngine.framework/AppleNeuralEngine", @"AppleNeuralEngine");
        A12V3AppendMappedImageStrings(lines, @"Espresso.framework/Espresso", @"Espresso");

        [lines addObject:@"RESULT=PASS targeted-runtime-v3"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
