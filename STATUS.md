# Current Project Status

- **Base main:** `45c28f44697253e20385edeab662c8db8098c55f`.
- **Investigation branch/worktree:** `investigate/dit-quality-runtime` at
  `/root/AnimaXS-quality`.
- **Current phase:** Phase 1 baseline and runtime/oracle audit.
- **Current best image:** W8-v2 from Actions run `31671198927`; latent cosine
  `0.7522`, RGB cosine `0.7171`, RGB RMSE `0.4850`. It has substantially better
  structure than W4 but retains an unacceptable fine grid/etched texture.
- **Current leading hypothesis:** repeated DiT numerical divergence, with the
  FP16 attention score/probability path the first suspect. This is unconfirmed.
- **Current blockers:** none. Telegram visibility is unavailable because this
  environment exposes no `tg_send` command or connector; work continues with
  persistent on-disk evidence.
- **Current running CI:** none.
- **Next action:** audit the attention/DiT implementation and existing oracle
  infrastructure, then add branch-only same-W8 and focused attention diagnostics.
- **Completion truth:** old 0.65 regression floors are not the acceptance gate;
  no current image is yet reference-comparable.
