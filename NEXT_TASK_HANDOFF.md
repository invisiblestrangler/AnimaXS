# AnimaXS Refinement Handoff

The runbook-defined refinement cycle is complete on `main` at commit
`2055c66`. The authoritative execution instructions remain in `RUNBOOK.md`.

Source provenance is pinned to `circlestone-labs/Anima` revision
`f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b`, file
`split_files/diffusion_models/anima-turbo-v1.0.safetensors`, LFS SHA
`c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174`.

Completed evidence:

- Pack workflow `31666871623` packed W4-v2 and W8-v2 in separate Ubuntu jobs,
  independently verified all 685 tensor blobs, and published both packs to
  dedicated HF repositories. Exact revisions, byte sizes, and hashes are in
  `DECISIONS.md` D077 and the workflow artifacts.
- Normal CI `31668078050` is green: 153 tests, 13 expected skips, 0 failures.
- Full inference workflow `31669515816` ran both exact canonical candidates on
  macOS simulator Metal with the same Qwen/VAE assets, prompt, seed/noise,
  sigmas, and graph. Both ran all 8 Euler steps and 224 block callbacks and
  produced generated/reference/comparison/metrics/pack-metadata artifacts.
- W8-v2 is the selected candidate: latent cosine `0.7522`, RGB cosine
  `0.7171`, and the cleanest generated image of the tested DiT variants.
  W4-v2 is rejected: latent cosine `0.6488`, RGB cosine `0.6111`, plus visibly
  stronger washout and texture. The W8 image still has a fine grid texture;
  that limitation is explicit rather than hidden by changing the gate.

If work resumes, the next task is physical iPhone XS Max acceptance or a
separate quality investigation aimed at removing the residual W8 texture. Do
not change the 0.65 regression floor or claim the legacy/W4 candidate is
acceptable based on metrics alone. Do not repack or republish without keeping
the pinned source revision and license notices.
