# AnimaXS Refinement Test Matrix

| Area | Test/evidence | Acceptance |
|---|---|---|
| ANMA parser | Real Swift parser opens each v2 pack | PASS; W4/W8 independent reports have zero errors and 685 CRC entries |
| Quantization | Synthetic W4 `[2,68]`, W8 `[2,65]` | PASS; row groups reset and nibble/byte decode exact |
| Packer | Dry plan, finite source, bounded writer, verifier | PASS; both source-pinned packs written and independently verified |
| HF publication | Dedicated W4/W8 model repositories | PASS; immutable revisions, sizes, SHA-256 recorded in D077 |
| Metal | W4/W8 dequant and direct matvec | PASS; W8 K=65 direct matvec parity and full pack inference |
| Runtime | Shared W4/W8 DiT graph | PASS; normal CI is green with 153 tests, 13 skips, 0 failures |
| Inference | Canonical W4-v2 and W8-v2 matrix jobs | PASS; both completed 8 steps/224 blocks and captured PNGs; W4 rejected by quality floor |
| Quality | Generated vs canonical reference images | W8-v2 selected: RGB cosine 0.7171 and visibly cleaner than legacy/W4-v2 |

The historical legacy W4 RGB cosine floor is a structural regression check, not
the image-quality acceptance criterion for this refinement cycle.
