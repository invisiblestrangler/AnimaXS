# AnimaXS Refinement Handoff

The repository is on `refine/packing-v2-w4-w8`, based on the verified current
project plus the temporary PNG-capture plumbing. The authoritative execution
instructions are in `RUNBOOK.md`.

Source provenance is pinned to `circlestone-labs/Anima` revision
`f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b`, file
`split_files/diffusion_models/anima-turbo-v1.0.safetensors`, LFS SHA
`c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174`.

The remaining critical path is: finish v2 packer and verifier, generalize the
shared DiT/adapter runtime to W8, run the two HF-backed pack jobs, run exact
canonical W4-v2/W8-v2 inference on macOS Metal, inspect the actual PNGs, and
record the evidence-backed winner. Do not call the legacy 0.65 regression floor
an image-quality result.
