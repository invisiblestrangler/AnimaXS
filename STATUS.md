# Current Project Status

- **Base main:** `45c28f44697253e20385edeab662c8db8098c55f`.
- **Investigation branch/worktree:** `investigate/dit-quality-runtime` at
  `/root/AnimaXS-quality`.
- **Current phase:** Phase 4, first-divergence localization beyond block 0.
- **Current best image:** W8-v2 from Actions run `31671198927`; latent cosine
  `0.7522`, RGB cosine `0.7171`, RGB RMSE `0.4850`. It has substantially better
  structure than W4 but retains an unacceptable fine grid/etched texture.
- **Attention result:** isolated FP32 score/softmax/PV reduced attention RMSE
  from `2.82e-4` to `5.41e-5`, but W8 final latent cosine only moved from
  `0.7522` to `0.75267` in run `31676322657`; attention precision is therefore
  insufficient as the primary explanation and does not earn full RGB inference.
- **Block-0 result:** exact same-input S/Q/E/M branch-delta run `31678571617`
  found tight Metal parity: M/Q relative L2 `0.00019` self, `0.00088` cross,
  `0.00023` MLP. Q/S is larger but still small (`0.00047`, `0.00231`,
  `0.00089`). Block-0 backend math is not the primary defect.
- **Current leading hypothesis:** quantization/precision error accumulates across
  later blocks or sampler steps; localize its growth before changing runtime.
- **Current blockers:** none. Telegram visibility is unavailable because this
  environment exposes no `tg_send` command or connector; work continues with
  persistent on-disk evidence.
- **Current running CI:** none; run `31678571617` completed successfully.
- **Next action:** capture sparse later blocks at step 0 with exact same-input
  S/Q/E comparisons, then narrow the first material growth interval.
- **Completion truth:** old 0.65 regression floors are not the acceptance gate;
  no current image is yet reference-comparable.
