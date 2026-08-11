# VAE T=1 causal-convolution fold report

Pinned ComfyUI commit: `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`  
VAE pack SHA-256: `10171af0b826927b75fecf4482aaa0e268254874e694a0788ebdd8c4254fc447`  
Validated tensors: **34**; last-slice failures: **0**.

The decoder is Wan `vae.py`, not the similarly named Cosmos tokenizer. For uncached T=1, `CausalConv3d` uses causal zero padding, so a temporal kernel folds to its final temporal slice. Summing temporal slices models replication padding and is wrong here. The two `time_conv` tensors are not executed because T=1 decoding creates no feature cache.

| Tensor | Shape | T=1 runtime | last-slice maxAbs | sum-fold maxAbs |
|---|---:|---|---:|---:|
| `conv2.weight` | `[16, 16, 1, 1, 1]` | executed | 0 | 0 |
| `decoder.conv1.weight` | `[384, 16, 3, 3, 3]` | executed | 5.55e-17 | 0.0236111 |
| `decoder.head.2.weight` | `[3, 96, 3, 3, 3]` | executed | 2.78e-17 | 0.0125853 |
| `decoder.middle.0.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0938193 |
| `decoder.middle.0.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.036611 |
| `decoder.middle.2.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0284316 |
| `decoder.middle.2.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0311548 |
| `decoder.upsamples.0.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0206835 |
| `decoder.upsamples.0.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 5.55e-17 | 0.0290571 |
| `decoder.upsamples.1.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0318867 |
| `decoder.upsamples.1.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 5.55e-17 | 0.036044 |
| `decoder.upsamples.10.residual.2.weight` | `[192, 192, 3, 3, 3]` | executed | 1.39e-17 | 0.0259098 |
| `decoder.upsamples.10.residual.6.weight` | `[192, 192, 3, 3, 3]` | executed | 2.78e-17 | 0.0345254 |
| `decoder.upsamples.12.residual.2.weight` | `[96, 96, 3, 3, 3]` | executed | 2.78e-17 | 0.0245257 |
| `decoder.upsamples.12.residual.6.weight` | `[96, 96, 3, 3, 3]` | executed | 1.39e-17 | 0.0403012 |
| `decoder.upsamples.13.residual.2.weight` | `[96, 96, 3, 3, 3]` | executed | 1.39e-17 | 0.0538396 |
| `decoder.upsamples.13.residual.6.weight` | `[96, 96, 3, 3, 3]` | executed | 5.55e-17 | 0.081971 |
| `decoder.upsamples.14.residual.2.weight` | `[96, 96, 3, 3, 3]` | executed | 2.78e-17 | 0.060411 |
| `decoder.upsamples.14.residual.6.weight` | `[96, 96, 3, 3, 3]` | executed | 1.11e-16 | 0.0582963 |
| `decoder.upsamples.2.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0345082 |
| `decoder.upsamples.2.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 5.55e-17 | 0.023654 |
| `decoder.upsamples.3.time_conv.weight` | `[768, 384, 3, 1, 1]` | skipped by decoder | 0 | 0.0669671 |
| `decoder.upsamples.4.residual.2.weight` | `[384, 192, 3, 3, 3]` | executed | 2.78e-17 | 0.0334749 |
| `decoder.upsamples.4.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 5.55e-17 | 0.0340784 |
| `decoder.upsamples.4.shortcut.weight` | `[384, 192, 1, 1, 1]` | executed | 0 | 0 |
| `decoder.upsamples.5.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0556823 |
| `decoder.upsamples.5.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 1.39e-17 | 0.0410294 |
| `decoder.upsamples.6.residual.2.weight` | `[384, 384, 3, 3, 3]` | executed | 2.78e-17 | 0.0395326 |
| `decoder.upsamples.6.residual.6.weight` | `[384, 384, 3, 3, 3]` | executed | 1.39e-17 | 0.0206458 |
| `decoder.upsamples.7.time_conv.weight` | `[768, 384, 3, 1, 1]` | skipped by decoder | 0 | 0.0654595 |
| `decoder.upsamples.8.residual.2.weight` | `[192, 192, 3, 3, 3]` | executed | 2.78e-17 | 0.0467168 |
| `decoder.upsamples.8.residual.6.weight` | `[192, 192, 3, 3, 3]` | executed | 5.55e-17 | 0.0726623 |
| `decoder.upsamples.9.residual.2.weight` | `[192, 192, 3, 3, 3]` | executed | 2.78e-17 | 0.0376701 |
| `decoder.upsamples.9.residual.6.weight` | `[192, 192, 3, 3, 3]` | executed | 2.78e-17 | 0.048771 |

`VAE_2D_FOLD_VALIDATED=LAST_TEMPORAL_SLICE_CAUSAL_ZERO`
