# AnimaXS Final Quality Test Matrix

| Class | Test/evidence | Runner | Cost | Status / key metric |
|---|---|---|---|---|
| Historical baseline | W8-v2 canonical full inference, run `31671198927` | macOS | full image | Verified: latent `0.7522`, RGB `0.7171`, RGB RMSE `0.4850`; visible fine grid remains |
| Historical baseline | W4-v2 canonical full inference, run `31671198927` | macOS | full image | Verified rejected: latent `0.6488`, RGB `0.6111`, RGB RMSE `0.5476` |
| Pack fidelity | Exact W8 reconstruction, run `31666871623` | Linux | pack | Verified: cosine `0.999987`, RMSE `0.004517` |
| Pack fidelity | Exact W4 reconstruction, run `31666871623` | Linux | pack | Verified: cosine `0.996461`, RMSE `0.070897` |
| Cheap source/reference | Same-W8 block-0 precision modes | Linux | cheap diagnostic | Pending |
| Attention numerics | Legacy vs FP32 score/softmax vs independent reference, run `31675825761` | macOS | focused | PASS: RMSE `2.82e-4` → `5.41e-5`; maxAbs `1.19e-3` → `2.40e-4` |
| Block parity | Same-input W8 block 0 | Linux + macOS | focused | Pending |
| Multi-block parity | W8 step-0 block drift CSV | Linux + macOS | moderate | Pending |
| Latent inference | Eight-step W8 FP32-attention candidate, run `31676322657` | macOS | expensive latent | Rejected as insufficient: cosine `0.75267`, RMSE `0.92902`, maxAbs `4.03151`; only `+0.00047` cosine |
| Full image | Canonical generated/reference/comparison PNG | macOS | full image | Pending; must be visibly reference-comparable |
| Production regression | Generic iPhone build + simulator tests | macOS | normal CI | Historical main green; rerun required for final candidate |

Metrics from different SHAs, packs, source revisions, or numerical modes must
not be compared without recording those differences. Artifacts must upload even
when assertions fail.
