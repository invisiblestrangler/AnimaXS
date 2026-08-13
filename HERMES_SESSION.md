# HERMES_SESSION.md

# Current state

Branch: investigate/animapk-cuda-parity
HEAD: df02810 (chore: add HERMES_ANIMAPK_CUDA_PARITY.md)
origin/main: 45c28f44697253e20385edeab662c8db8098c55f
Last known green CI: 31671071004 (main, 153 tests / 13 skips / 0 failures)
Clore status: connected via /root/key
Clore GPU: RTX 3060 12GB (O-2024354, order 2024354, $0.70/h, jupyter:ubuntu24.04-v2 image, fresh: no torch yet)
Current hypothesis: grid/carrier artifact root cause unknown; controlled precision ladder A-E + direct .animapk CUDA vs Metal needed to localize earliest divergence
Control matrix status:
  A BF16 source: pending (need official weights on Clore)
  B FP16 source: pending
  C FP16-all animapk CUDA: pending (pack SHA c17d62727beb114590febbe9dd019e5c9d523863d6e2b32b17c5872c6b0635ca, 4,193,255,424 B)
  D W8 animapk CUDA: pending (W8-v2 SHA 8b63c7fd9b5872805e5a2ba799ab6d79989c54a6a89a4f34edf022c59c9ed130)
  E W4 animapk CUDA: pending (W4-v2 SHA 9109e502a236822d168a3e0b5dc39beb3c03d213b4bf459865815af934c8b5cf)
  C/D/E Metal: pending (GitHub macOS workflow)
Direct animapk CUDA status: not started (scripts/animapk_cuda/ to build)
Last experiment: (session start) repo audit + Clore connect
Last result: HEAD==origin/main 45c28f4; Clore reachable, GPU RTX 3060
Proven: W8-v2 full inference PASS latent 0.7522/RGB 0.7171; W4-v2 FAIL 0.6488/0.6111; grid texture remains in W8 (D079-D081)
Rejected: (none this session)
Blockers: none yet
Next 3 actions:
  1. Clore: venv + torch cu126 install, download official BF16 source (f7382c4…), verify SHA c0b90503…
  2. Copy packs (FP16-all/W8/W4) + canonical fixture to Clore; verify SHAs
  3. BF16->FP16 storage audit + oracle-vs-upstream validation
Important GitHub run IDs: 31671071004 (main CI green), 31671198927 (final matrix rerun)
Important pack SHAs: FP16-all c17d6272…, W8-v2 8b63c7fd…, W4-v2 9109e502…
Important source SHAs: circlestone-labs/Anima@f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b → c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174
Important Clore paths: /root (fresh), no /workspace yet
