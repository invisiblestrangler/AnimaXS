# Current Project Status

- **Base main:** `45c28f44697253e20385edeab662c8db8098c55f`.
- **Investigation branch/worktree:** `investigate/dit-quality-runtime` at
  `/root/AnimaXS-quality`.
- **Current phase:** Phase 2/4 same-W8 reference and first-divergence localization.
- **Current best image:** W8-v2 from Actions run `31671198927`; latent cosine
  `0.7522`, RGB cosine `0.7171`, RGB RMSE `0.4850`. It has substantially better
  structure than W4 but retains an unacceptable fine grid/etched texture.
- **Attention result:** isolated FP32 score/softmax/PV reduced attention RMSE
  from `2.82e-4` to `5.41e-5`, but W8 final latent cosine only moved from
  `0.7522` to `0.75267` in run `31676322657`; attention precision is therefore
  insufficient as the primary explanation and does not earn full RGB inference.
- **Current leading hypothesis:** either another repeated DiT boundary/runtime
  mismatch or genuine trajectory sensitivity to the small W8 weight error.
- **Current blockers:** none. Telegram visibility is unavailable because this
  environment exposes no `tg_send` command or connector; work continues with
  persistent on-disk evidence.
- **Current running CI:** none.
- **Next action:** build the cheapest decisive exact-W8 same-input block/source
  comparison and step-0 localization experiment. One permitted Sol 5.6 Extra
  High advisor session is active to review this experiment design.
- **Completion truth:** old 0.65 regression floors are not the acceptance gate;
  no current image is yet reference-comparable.
