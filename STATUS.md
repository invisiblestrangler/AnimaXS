# STATUS — AnimaXS (keep short & current)

- **Current milestone:** Phase 3 — animapk parser + quant decoders validated against real packs
- **Current task:** Commit parser + tests; wire CI green; then Metal context (E001)
- **Last green commit:** 5b1ec67 (ci: require committed xcodeproj) + bootstrap committed xcodeproj (c38a7af)
- **Current CI run:** (push CI; last one failed on missing committed xcodeproj — now committed)
- **What currently works:**
  - Repo `invisiblestrangler/AnimaXS` populated, xcodeproj generated+committed (XcodeGen)
  - Swift ANMA v1 parser: header/JSON/table, JSON-authoritative names+shapes, blobOffset index, CRC-32, alignment, range checks
  - CPU W4/W8 decoders (fp16 scale/zero, group 64, nibble order)
  - Validated against real packs: all 1,189 tensors CRC-clean, all blobOffsets 16 KB aligned, W4/W8 known vectors byte-exact, DiT block + TE layer physical ordering confirmed
  - 3 test suites: AnimapkParsingTests (synthetic), QuantDecoderTests (mechanics), RealPackDecoderTests (env-gated real packs)
- **What currently fails:** Nothing known. CI not yet re-run after parser commit.
- **Known device-only unknowns:** MPS fp16 accuracy on Apple5; A12 memory/jetsam/perf/watchdog/thermal. (PENDING — no physical device.)
- **Next three tasks:**
  1. Commit parser+decoders+tests; push; get CI green
  2. E001 — MetalContext + probe
  3. E002 — Metal dequant kernels (w4/w8 to half)
