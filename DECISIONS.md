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
