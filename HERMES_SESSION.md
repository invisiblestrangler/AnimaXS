# HERMES_SESSION.md

# Current state

Branch: investigate/animapk-cuda-parity
HEAD: (various ladder commits — see git log)
origin/main: 45c28f44697253e20385edeab662c8db8098c55f
Last known green CI: 31671071004 (main)
Clore status: ACTIVE, order 2024354, RTX 3060 12GB, torch 2.7.1+cu126 venv at /workspace/venv
Clore GPU: RTX 3060 12GB (O-2024354, $0.70/h)
Current hypothesis: grid/carrier artifact — separating storage precision vs pack container vs quantization vs backend via ladder A-E
Control matrix status:
  A BF16 source (actual upstream): RUNNING (comfy stub w/ real predict2.py)
  B FP16 source: RUNNING
  C FP16-all animapk CUDA: RUNNING (pack rebuilt on Clore: SHA 07adee8e… ≠ recorded c17d6272… — metadata-only diff: packer env; 685 blobs verified vs source SHA c0b90503…)
  D W8 animapk CUDA: RUNNING (W8-v2 8b63c7fd… verified)
  E W4 animapk CUDA: RUNNING (W4-v2 9109e502… verified)
  C/D/E Metal: pending GitHub macOS workflow (reuse FullInferenceTests step-0 capture)
Direct animapk CUDA status: decoded_reference + streaming_animapk implemented, running in ladder
Last experiment: BF16->FP16 storage audit — 100.0% bit-exact on all 2.09B elements (all 685 tensors). A->B weight storage delta = ZERO.
Last result: fixture prep OK (context512 via source adapter); comfy stub MiniTrainDIT instantiates; ladder launched
Proven: BF16 values all exactly representable in FP16; W8-v2/W4-v2 HF packs SHA-match D077
Rejected: (none this session)
Blockers: none
Next 3 actions:
  1. Collect ladder results (A vs A2 vs B vs C/D/E, decoded vs streaming)
  2. Publish fp16-all pack to HF (ScalingBiz) for macOS runner + upload evidence
  3. Build quality-step0-backend-matrix.yml (manual dispatch, C/D/E Metal step-0 capture)
Important GitHub run IDs: 31671071004 (main CI green), 31671198927 (final matrix rerun)
Important pack SHAs: FP16-all 07adee8e… (rebuilt; recorded c17d6272…), W8-v2 8b63c7fd…, W4-v2 9109e502…
Important source SHAs: circlestone-labs/Anima@f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b → c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
Important Clore paths: /workspace/{repo,source,packs,out/fixture,out/storage,out/ladder,comfy-ref}
