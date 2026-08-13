# STATUS — AnimaXS (keep short & current)

- **Current milestone:** The runbook refinement cycle is complete. The project
  now has bounded ANMA-v1 W4/W8 packing, one shared W4/W8 DiT runtime graph,
  independent pack verification, HF provenance, and exact full-inference image
  evidence.
- **Final HEAD:** `2055c66` (`main`, `origin/main`). The only untracked path is
  the pre-existing user artifact `scripts/oracle_out/block0/`.
- **Normal CI:** main run `31668078050` is **SUCCESS** — project consistency,
  iPhone build, and simulator tests all pass; 153 tests, 13 expected skips,
  0 failures. The simulator is shut down and erased before pack-free tests.
- **Source pin:** `circlestone-labs/Anima` revision
  `f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b`, source SHA-256
  `c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174`.
- **Published DiT derivatives:** W4-v2 is
  `ScalingBiz/AnimaXS-DiT-W4`, revision
  `06353ece23ae02bf71e89e3d7d68a814714e6216`, 1,187,479,552 bytes, SHA-256
  `9109e502a236822d168a3e0b5dc39beb3c03d213b4bf459865815af934c8b5cf`.
  W8-v2 is `ScalingBiz/AnimaXS-DiT-W8`, revision
  `589d028122f872e66ee20cdd12cb55eb3b816add`, 2,232,975,360 bytes, SHA-256
  `8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130`.
- **Inference decision:** Full matrix run `31669515816` completed both
  canonical 8-step/224-block inferences and uploaded artifacts. W8-v2 passes
  with latent cosine `0.7522` and RGB cosine `0.7171`; W4-v2 is rejected by
  the same documented 0.65 floor with latent cosine `0.6488` and RGB cosine
  `0.6111`. Visual inspection selects W8-v2: it is substantially cleaner and
  more detailed than the legacy W4 and W4-v2, while retaining a visible fine
  grid texture that is recorded as the remaining quality limitation.
- **Evidence artifacts:** the run uploads
  `anima-xs-refine-w8-v2-images` and `anima-xs-refine-w4-v2-images`, each with
  generated/reference/comparison PNGs, metrics, and exact pack metadata. The
  comparison attachment is upright and directly inspectable.
- **License scope:** non-commercial redistribution remains permitted with the
  committed license copies/notices; commercial use remains blocked without
  separate authorization.
- **Device-only unknowns:** A12 speed, memory/jetsam, Apple5 behavior,
  watchdog limits, thermal stability, and real-device install/launch remain
  physical-device acceptance items and were not inferred from hosted Metal.
