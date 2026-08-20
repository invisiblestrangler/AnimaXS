#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach-o/dyld.h>
#import <mach-o/loader.h>
#include <stdint.h>
#include <string.h>

// Experiment branch only.
//
// V3 established the concrete private-runtime vocabulary for ANE procedures.
// V4 remains diagnostic-only but does two more discriminating things:
//   1. preserves neighboring __cstring entries around the exact Espresso
//      multi-procedure parser/compiler terms instead of alphabetically sorting;
//   2. if the normal W8 cache already contains block-0 self-O, loads that one
//      known-good single-procedure Espresso model, dumps the emitted ANEF model
//      description/procedure metadata, then unloads it. No evaluation occurs.

#if TARGET_OS_SIMULATOR
static inline NSString *A12ANETargetedRuntimeProbe(void) {
    return @"ANE targeted runtime probe v4\nRESULT=SKIP simulator";
}
#else

static inline BOOL A12V4Contains(NSString *value, NSString *needle) {
    if (!value || !needle) return NO;
    return [value rangeOfString:needle options:NSCaseInsensitiveSearch].location != NSNotFound;
}

static inline NSString *A12V4Desc(id value) {
    if (!value) return @"(nil)";
    @try { return [value description] ?: @""; }
    @catch (NSException *e) {
        return [NSString stringWithFormat:@"<description threw %@: %@>", e.name ?: @"?", e.reason ?: @"?"];
    }
}

static inline const struct mach_header_64 *A12V4MappedHeader(NSString *needle,
                                                             intptr_t *slideOut,
                                                             NSString **pathOut) {
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        const char *raw = _dyld_get_image_name(i);
        if (!raw) continue;
        NSString *path = [NSString stringWithUTF8String:raw];
        if (!A12V4Contains(path, needle)) continue;
        const struct mach_header *h = _dyld_get_image_header(i);
        if (!h || h->magic != MH_MAGIC_64) continue;
        if (slideOut) *slideOut = _dyld_get_image_vmaddr_slide(i);
        if (pathOut) *pathOut = path;
        return (const struct mach_header_64 *)h;
    }
    return NULL;
}

static inline NSArray<NSString *> *A12V4CStringRuns(const uint8_t *bytes, size_t length) {
    if (!bytes || length == 0) return @[];
    if (length > (64u * 1024u * 1024u)) length = 64u * 1024u * 1024u;
    NSMutableArray<NSString *> *out = [NSMutableArray array];
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
            if (run >= 2 && run <= 2048) {
                NSString *s = [[NSString alloc] initWithBytes:bytes + start
                                                      length:run
                                                    encoding:NSASCIIStringEncoding];
                if (s) [out addObject:s];
            }
            start = SIZE_MAX;
        }
    }
    return out;
}

static inline void A12V4AppendCStringContexts(NSMutableArray<NSString *> *lines,
                                               NSString *imageNeedle,
                                               NSString *label,
                                               NSArray<NSString *> *targets) {
    intptr_t slide = 0;
    NSString *path = nil;
    const struct mach_header_64 *header = A12V4MappedHeader(imageNeedle, &slide, &path);
    [lines addObject:[NSString stringWithFormat:@"-- CSTRING CONTEXT %@ --", label]];
    if (!header) {
        [lines addObject:@"mappedImage=NOT_FOUND"];
        return;
    }
    [lines addObject:[NSString stringWithFormat:@"mappedImage=FOUND path=%@ slide=%lld",
        path ?: @"?", (long long)slide]];

    const uint8_t *cursor = (const uint8_t *)(header + 1);
    NSMutableArray<NSString *> *all = [NSMutableArray array];
    for (uint32_t commandIndex = 0; commandIndex < header->ncmds; ++commandIndex) {
        const struct load_command *lc = (const struct load_command *)cursor;
        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(struct segment_command_64)) {
            const struct segment_command_64 *seg = (const struct segment_command_64 *)lc;
            const struct section_64 *sections = (const struct section_64 *)(seg + 1);
            for (uint32_t s = 0; s < seg->nsects; ++s) {
                const struct section_64 *sec = &sections[s];
                NSString *sect = [NSString stringWithUTF8String:sec->sectname] ?: @"";
                if (![sect isEqualToString:@"__cstring"] || sec->size == 0) continue;
                uintptr_t address = (uintptr_t)(sec->addr + (uint64_t)slide);
                [all addObjectsFromArray:A12V4CStringRuns((const uint8_t *)address, (size_t)sec->size)];
            }
        }
        if (lc->cmdsize == 0) break;
        cursor += lc->cmdsize;
    }

    [lines addObject:[NSString stringWithFormat:@"cstringCount=%lu", (unsigned long)all.count]];
    const NSInteger radius = 12;
    for (NSString *target in targets) {
        NSMutableIndexSet *hits = [NSMutableIndexSet indexSet];
        [all enumerateObjectsUsingBlock:^(NSString *obj, NSUInteger idx, BOOL *stop) {
            if (A12V4Contains(obj, target)) [hits addIndex:idx];
        }];
        [lines addObject:[NSString stringWithFormat:@"TARGET %@ hits=%lu", target, (unsigned long)hits.count]];
        __block NSUInteger emitted = 0;
        [hits enumerateIndexesUsingBlock:^(NSUInteger idx, BOOL *stop) {
            if (emitted >= 4) { *stop = YES; return; }
            NSInteger lo = MAX((NSInteger)0, (NSInteger)idx - radius);
            NSInteger hi = MIN((NSInteger)all.count - 1, (NSInteger)idx + radius);
            [lines addObject:[NSString stringWithFormat:@"  HIT index=%lu context=%ld..%ld",
                (unsigned long)idx, (long)lo, (long)hi]];
            for (NSInteger j = lo; j <= hi; ++j) {
                NSString *mark = ((NSUInteger)j == idx) ? @">>" : @"  ";
                [lines addObject:[NSString stringWithFormat:@"%@ [%ld] %@", mark, (long)j, all[(NSUInteger)j]]];
            }
            emitted += 1;
        }];
    }
}

static inline NSString *A12V4ProjectionKey(NSUInteger spatial,
                                            NSUInteger inputChannels,
                                            NSUInteger outputChannels) {
    return [NSString stringWithFormat:
        @"{\"isegment\":0,\"inputs\":{\"x\":{\"shape\":[%lu,1,1,%lu,1]}},\"outputs\":{\"y\":{\"shape\":[%lu,1,1,%lu,1]}}}",
        (unsigned long)spatial, (unsigned long)inputChannels,
        (unsigned long)spatial, (unsigned long)outputChannels];
}

static inline NSDictionary *A12V4Options(void) {
    return @{ @"kANEModelKeyEspressoTranslationOptions": @{ @"compute_unit_mask": @5 },
              @"kANEFModelIdentityStrKey": @"compiled_0" };
}

static inline NSURL *A12V4FindKnownGoodProjection(void) {
    NSString *root = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    if (!root) return nil;
    NSURL *base = [[NSURL fileURLWithPath:root isDirectory:YES]
        URLByAppendingPathComponent:@"AnimaXS-ANE" isDirectory:YES];
    NSError *error = nil;
    NSArray<NSURL *> *entries = [[NSFileManager defaultManager]
        contentsOfDirectoryAtURL:base
        includingPropertiesForKeys:nil
        options:NSDirectoryEnumerationSkipsHiddenFiles
        error:&error];
    if (!entries) return nil;
    for (NSURL *url in entries) {
        NSString *name = url.lastPathComponent ?: @"";
        if ([name containsString:@"ane-u8-row-v1-b0-self_o-"] && [name hasSuffix:@".mlmodelc"]) return url;
    }
    return nil;
}

static inline void A12V4AppendKnownGoodModelDescription(NSMutableArray<NSString *> *lines) {
    [lines addObject:@"-- KNOWN-GOOD SINGLE-PROCEDURE MODEL --"];
    NSURL *url = A12V4FindKnownGoodProjection();
    if (!url) {
        [lines addObject:@"cacheModel=SKIP no b0-self_o prepared entry found"];
        return;
    }
    [lines addObject:[NSString stringWithFormat:@"cacheModel=%@", url.lastPathComponent ?: @"?"]];

    void *handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                          RTLD_NOW | RTLD_LOCAL);
    Class modelClass = NSClassFromString(@"_ANEModel");
    Class clientClass = NSClassFromString(@"_ANEClient");
    if (!handle || !modelClass || !clientClass) {
        [lines addObject:@"cacheModel=SKIP private runtime unavailable"];
        return;
    }

    @try {
        NSString *key = A12V4ProjectionKey(1024, 2048, 2048);
        id model = ((id(*)(Class,SEL,id,id))objc_msgSend)(
            modelClass, NSSelectorFromString(@"modelAtURL:key:"), url, key);
        id client = ((id(*)(Class,SEL))objc_msgSend)(
            clientClass, NSSelectorFromString(@"sharedConnection"));
        if (!model || !client) {
            [lines addObject:@"cacheModel=FAIL model/client construction"];
            return;
        }

        NSError *loadError = nil;
        BOOL loaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError **))objc_msgSend)(
            client, NSSelectorFromString(@"loadModel:options:qos:error:"),
            model, A12V4Options(), 25, &loadError);
        [lines addObject:[NSString stringWithFormat:@"load=%@ error=%@",
            loaded ? @"PASS" : @"FAIL", A12V4Desc(loadError)]];
        if (!loaded) return;

        id attrs = ((id(*)(id,SEL))objc_msgSend)(model, NSSelectorFromString(@"modelAttributes"));
        id desc = [attrs isKindOfClass:NSDictionary.class] ? attrs[@"ANEFModelDescription"] : nil;
        [lines addObject:[NSString stringWithFormat:@"ANEFModelDescription=%@", A12V4Desc(desc)]];

        SEL procSel = NSSelectorFromString(@"procedureInfoForProcedureIndex:");
        SEL inSel = NSSelectorFromString(@"inputSymbolIndicesForProcedureIndex:");
        SEL outSel = NSSelectorFromString(@"outputSymbolIndicesForProcedureIndex:");
        id proc0 = [model respondsToSelector:procSel]
            ? ((id(*)(id,SEL,unsigned int))objc_msgSend)(model, procSel, 0) : nil;
        id in0 = [model respondsToSelector:inSel]
            ? ((id(*)(id,SEL,unsigned int))objc_msgSend)(model, inSel, 0) : nil;
        id out0 = [model respondsToSelector:outSel]
            ? ((id(*)(id,SEL,unsigned int))objc_msgSend)(model, outSel, 0) : nil;
        [lines addObject:[NSString stringWithFormat:@"procedureInfo[0]=%@", A12V4Desc(proc0)]];
        [lines addObject:[NSString stringWithFormat:@"inputSymbolIndices[0]=%@", A12V4Desc(in0)]];
        [lines addObject:[NSString stringWithFormat:@"outputSymbolIndices[0]=%@", A12V4Desc(out0)]];

        NSError *unloadError = nil;
        BOOL unloaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError **))objc_msgSend)(
            client, NSSelectorFromString(@"unloadModel:options:qos:error:"),
            model, A12V4Options(), 25, &unloadError);
        [lines addObject:[NSString stringWithFormat:@"unload=%@ error=%@",
            unloaded ? @"PASS" : @"FAIL", A12V4Desc(unloadError)]];
    } @catch (NSException *e) {
        [lines addObject:[NSString stringWithFormat:@"cacheModel=EXCEPTION %@ %@",
            e.name ?: @"?", e.reason ?: @"?"]];
    }
}

static inline NSString *A12ANETargetedRuntimeProbe(void) {
    @autoreleasepool {
        NSMutableArray<NSString *> *lines = [NSMutableArray arrayWithObjects:
            @"ANE targeted runtime probe v4",
            @"mode=ordered parser-string context + one known-good ANEF description; no compile/evaluate",
            nil];

        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
               RTLD_NOW | RTLD_LOCAL);
        dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
               RTLD_NOW | RTLD_LOCAL);

        A12V4AppendCStringContexts(lines,
            @"Espresso.framework/Espresso", @"Espresso",
            @[@"InputNetworks", @"ProcedureList", @"ProcedureParams",
              @"multi_head", @"multiprocedure", @"espresso.ane.no_mh_procedures"]);

        A12V4AppendCStringContexts(lines,
            @"AppleNeuralEngine.framework/AppleNeuralEngine", @"AppleNeuralEngine",
            @[@"ANEFModelProcedures", @"kANEFModelProcedureNameToIDMapKey",
              @"kANEFModelInstanceParameters"]);

        A12V4AppendKnownGoodModelDescription(lines);
        [lines addObject:@"RESULT=PASS targeted-runtime-v4"];
        return [lines componentsJoinedByString:@"\n"];
    }
}

#endif
