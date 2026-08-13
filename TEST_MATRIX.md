# AnimaXS Refinement Test Matrix

| Area | Test/evidence | Acceptance |
|---|---|---|
| ANMA parser | Real Swift parser opens each v2 pack | PASS; all ranges and metadata valid |
| Quantization | Synthetic W4 `[2,68]`, W8 `[2,65]` | Row groups reset; nibble/byte decode exact |
| Packer | Dry plan, finite source, bounded writer, verifier | PASS; no malformed pack |
| Metal | W4/W8 dequant and direct matvec | Tight CPU/Metal parity; no OOB writes |
| Runtime | Shared W4/W8 DiT graph | Legacy W4 tests unchanged; W8 accepted |
| CI | Normal workflow with selected simulator reset | Green and deterministic |
| Inference | Canonical W4-v2 and W8-v2 matrix jobs | 8 steps, 224 blocks, finite output, PNGs |
| Quality | Generated vs canonical reference images | Actual visual comparison plus latent/RGB metrics |

The historical legacy W4 RGB cosine floor is a structural regression check, not
the image-quality acceptance criterion for this refinement cycle.
