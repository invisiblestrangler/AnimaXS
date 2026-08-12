# MODEL_LICENSE — AnimaXS redistributed model weights

**A005 (2026-08-12) — source-backed audit of every redistributed weight source.**

This document records the applicable license for each model component the app
runs, whether redistribution of the converted/quantized derivative is
permitted for this project's intended public **non-commercial** release, and
the required attribution/notices. It is a provenance record, not legal advice.
Full license texts are archived under `docs/model-licenses/`.

## Components and provenance

| Pack | Filename | Source project | License | Redistribution of converted derivative? |
|------|----------|----------------|---------|----------------------------------------|
| DiT + adapter (Anima Turbo) | `anima-turbo-v1.0-xsmax-w4.animapk` | Anima (CircleStone Labs), itself a derivative of NVIDIA Cosmos-Predict2-2B-Text2Image | **CircleStone Non-Commercial License v1.2** (controlling) + **NVIDIA Open Model License** | **Yes, non-commercial only**, with license copy + attribution + modification disclosure |
| Qwen3 text encoder | `qwen3-0.6b-xsmax-w8.animapk` | Qwen (Alibaba) Qwen3-0.6B | **Apache License 2.0** | Yes, with Apache 2.0 notice |
| Qwen-Image VAE | `qwen-image-vae-xsmax-fp16.animapk` | Qwen (Alibaba) Qwen-Image | **Apache License 2.0** | Yes, with Apache 2.0 notice |

## DiT/adapter pack — the controlling restriction (CircleStone NC v1.2)

Anima's model card (retrieved 2026-08-12) states the model is licensed under
the **CircleStone Labs Non-Commercial License** and is a **Derivative Model** of
Cosmos-Predict2-2B-Text2Image, subject to the **NVIDIA Open Model License**
insofar as it applies to Derivative Models.

CircleStone NC v1.2 (§2a, §3) grants a **non-exclusive, non-transferable,
royalty-free, revocable license to use, create Derivatives of, and Distribute
the Models/Derivatives solely for Non-Commercial Purposes**, subject to:

- **Non-commercial use only** (§2b, §4a): no direct/indirect payment, no
  revenue-generating activity, no production use, no training/fine-tuning for
  commercial use.
- **Distribution conditions** (§3): (a) provide a copy of the license to
  recipients; (b) prominently display the required Attribution Notice; (d) for
  Derivatives: note the modification, and impose no terms conflicting with the
  license.
- **Restrictions** (§4): no commercial/production use; don't remove copyright
  notices; don't offer terms inconsistent with the license.

The NVIDIA Open Model License (version 2025-10-24) is **permissive** —
"Models are commercially usable", "You are free to create and distribute
Derivative Models" — and requires (Redistribution section): a copy of the
Agreement, the notice "Licensed by NVIDIA Corporation under the NVIDIA Open
Model License", and for Cosmos derivatives "Built on NVIDIA Cosmos" in product
documentation.

## Assessment for the intended release

The intended use is a **public GitHub release** (`model-assets-v1`) of the
converted/quantized `.animapk` packs so users can run AnimaXS **for personal,
non-commercial research and experimentation on an iPhone XS Max**. Under both
the CircleStone NC and NVIDIA OML:

- **Non-commercial redistribution of the converted/quantized derivatives is
  permitted** — this is the explicit §2a/§3 grant of CircleStone NC and the
  Redistribution section of NVIDIA OML.
- The following must accompany the release and the app:
  1. **CircleStone NC v1.2** license copy (`docs/model-licenses/CircleStone-NC-1.2.md`);
  2. **CircleStone Attribution Notice** verbatim (see `MODEL_NOTICE.txt`);
  3. **NVIDIA Open Model License** copy + "Licensed by NVIDIA Corporation under
     the NVIDIA Open Model License";
  4. **"Built on NVIDIA Cosmos"** (the DiT pack is a Cosmos derivative);
  5. **Apache 2.0** for the Qwen3 text encoder and Qwen-Image VAE;
  6. **Modification disclosure**: the `.animapk` packs are converted/quantized
     (W4/W8) derivatives of the source models — noted in `MODEL_NOTICE.txt`.
- **Commercial use is NOT authorized** by this license chain. The app must not
  be marketed or used for commercial/production purposes without separate
  licenses from CircleStone (and NVIDIA where applicable). AnimaXS's README and
  NOTICE must state this explicitly.

## Verdict

**A005: RESOLVED for non-commercial redistribution.** The three packs may be
released publicly for non-commercial use provided the license copies and
notices above are included. **Commercial distribution/use remains blocked**
without separate licenses; this is recorded as a precise constraint, not an
unresolved blocker, because the intended use (personal, non-commercial, on a
user's own device) is within the granted scope.

## Required notice text (reproduced in MODEL_NOTICE.txt)

CircleStone attribution (verbatim, required by §3b):

> "The CircleStone Model is licensed by CircleStone Labs LLC under the
> CircleStone Non-Commercial License. Copyright CircleStone Labs LLC.
> IN NO EVENT SHALL CIRCLESTONE LABS LLC BE LIABLE FOR ANY CLAIM, DAMAGES OR
> OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL."

NVIDIA notice (required by NVIDIA OML Redistribution):

> "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"

Cosmos attribution (required by NVIDIA OML Redistribution for Cosmos derivatives):

> "Built on NVIDIA Cosmos"

## Source references (retrieved 2026-08-12)

- Anima model card + license: `https://huggingface.co/circlestone-labs/Anima`
  (`README.md` §145-164, `LICENSE.md`)
- NVIDIA Open Model License (2025-10-24):
  `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/`
- Qwen3-0.6B: `https://huggingface.co/Qwen/Qwen3-0.6B` (Apache 2.0)
- Qwen-Image: `https://huggingface.co/Qwen/Qwen-Image` (Apache 2.0)

> Note: the Cosmos-Predict2 model card on HuggingFace is access-restricted
> (gated); the applicable terms are those of the NVIDIA Open Model License as
> referenced by the Anima model card and NVIDIA's public license page.
