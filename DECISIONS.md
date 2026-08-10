# DECISIONS — AnimaXS

Record non-obvious choices and why. Append only; newest at bottom.

## Preflight / project
- **D001: Deployment target = iOS 18.0** (runbook §5). Build SDK under Xcode 26.3 = iOS 26.2; minimum OS = 18.0. That is what installs on the user's iOS 18.6 phone.
- **D002: Swift language mode 5** (runbook §9). async/await available; less strict-Sendable friction for Metal/MPS; supported by Xcode 26.3. No Swift 6 migration as part of inference.
- **D003: XcodeGen = project-generation tool only** (runbook §9). Commit both `project.yml` and generated `AnimaXS.xcodeproj` so the Mac user needs no XcodeGen. CI regenerates + `git diff --exit-code`.
- **D004: Pin swift-transformers to release 1.3.3** (latest stable, 2026-05-16; iOS 16+). Track verified release, not `main`. Will re-verify tokenizer parity before accepting.
- **D005: CI runner = macos-15, explicitly select Xcode 26.3** via `DEVELOPER_DIR=/Applications/Xcode_26.3.app/Contents/Developer`. Xcode 16.4 is default on that image, so selection is mandatory. Verified 26.3 (17C529) exists at that path.
- **D006: XcodeGen installed via Homebrew in CI** — not preinstalled on macos-15 runner (verified against runner-images README).
- **D007: Model packs via GitHub Releases, not git / LFS** (runbook §14). Release `model-assets-v1`. Never commit .animapk to git.

## Parsing / format (validated against real packs 2026-08-10)
- **D008: Real packs have `json_offset=256` and `table_offset=327936` — NOT 16 KB aligned** (only `payload_offset` and every `blob_offset` are). ANIMAPK_SPEC.md's "all sections 16 KB aligned" claim is inaccurate. Parser enforces blob alignment (the real invariant) but NOT section alignment.
- **D009: Architecture JSON `component` is a string** (`"dit"`/`"te"`/`"vae"`), not an integer code. The integer code 1/2/3 lives only in the 256-byte binary header.
- **D010: TE layers are ALSO physically string-sorted** (`0,1,10..19,2,20..27,3..9`), exactly like DiT blocks — HANDOFF.md's "layers 0-27 in order" is misleading. Each TE layer spans a contiguous 16,777,216 B range, but layer N's location MUST come from metadata lookup, never arithmetic. Same rule as DiT block ranges.
- **D011: Every tensor blob (all 1,189 across 3 packs) verified 16 KB aligned and CRC-32 clean** (0 mismatches) by the Swift parser. W4/W8 known vectors byte-exact vs HANDOFF.md §12–§13.
- **D012: DiT physical block order confirmed** = `0,1,10..19,2,20..27,3..9` (string sort). Each block's 20 tensors are contiguous and span exactly 38,993,920 B; adjacent blocks tile without gaps.
- **D013: `TestPackFactory` (synthetic ANMA v1 writer) uses a FIXED region layout** mirroring real packs: JSON at 256, table at 16 KB, payload after. This avoids JSON-size/offset circular dependency and keeps synthetic tests free of multi-GB assets. (Iteration note: `Data._Representation.init(count:)` aborts on negative counts — guard all padding arithmetic.)
