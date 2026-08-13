# HERMES_SESSION.md

# Current state

Branch: investigate/animapk-cuda-parity (HEAD ~2f8ef28; never touch main for experiments — one infra-only workflow file was pushed to main: 686e5a8, required for workflow_dispatch registration)
origin/main: 686e5a8 (was 45c28f4 + one infra commit registering quality-step0-backend-matrix.yml)
Last known green CI: 31671071004 (main)
Clore status: ACTIVE order 2024354 (RTX 3060 12GB, $0.70/h, spend ~39 CLORE ≈ $0.037). SSH flaky (edge rate-limits our IP intermittently; Jupyter edge stays up). tmux session "ladder" used for long runs.
Clore GPU: RTX 3060 12GB
Current hypothesis: grid/carrier artifact — separating storage precision vs pack container vs quantization vs backend via ladder A-E
Control matrix status (CUDA, step-0, all captured):
  A BF16 source actual upstream (comfy stub w/ real predict2.py): DONE (caps_A_upstream_bf16)
  A2 oracle bf16: DONE  |  B fp16 source: DONE
  C FP16-all pack decoded+streaming: DONE  |  D W8 decoded+streaming: DONE  |  E W4 decoded+streaming: DONE
  Postprocess (comparisons/plots/decoder parity): RUNNING (resume mode)
  C/D/E Metal: GitHub run 31752911430 in progress (quality-step0-backend-matrix.yml, 3 macOS jobs)
Direct animapk CUDA status: decoded_reference + streaming_animapk implemented; all 8 capture sets saved; E was rerun (VM rebooted mid-run)
Last experiment: resume ladder postprocess
Last result: BF16→FP16 storage 100% bit-exact (all 2.09B elements); ladder captures complete for A/A2/B/C/D/E
Proven: BF16 values all exactly representable in FP16; W8-v2/W4-v2 HF packs SHA-match D077; FP16-all rebuilt (07adee8e…) published to HF ScalingBiz/AnimaXS-DiT-FP16-ALL rev 54160ee9
Rejected: (none this session)
Blockers: Clore SSH intermittently blocked (rate limit); ladder VM rebooted once mid-run (resume mode built)
Next 3 actions:
  1. Collect ladder postprocess artifacts (CSV/JSON/MD/PNG) from Clore → VPS → HF evidence repo
  2. Collect Metal step-0 captures from run 31752911430 → backend_compare.py → backend_parity artifacts
  3. Write DECISIONS.md D082+ entries, final report, push, close Clore
Important GitHub run IDs: 31752911430 (backend matrix, running), 31750949327 (failed: adapterSeconds scope — fixed), 31671071004 (main CI)
Important pack SHAs: FP16-all 07adee8e… (rebuilt; recorded c17d6272…), W8-v2 8b63c7fd…, W4-v2 9109e502…
Important source SHAs: circlestone-labs/Anima@f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b → c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
Important Clore paths: /workspace/{repo,source,packs,out/fixture,out/storage,out/ladder,comfy-ref}; tmux session "ladder"; /root/HUGGINGFACE_TOKEN
