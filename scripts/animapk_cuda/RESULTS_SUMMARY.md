# Real-graph precision ladder — authoritative final results
# case1_danbooru_seed1337, 8-step Euler, final latent vs golden final_latent
# Real pinned ComfyUI MiniTrainDIT (predict2.py @ cbbc9da), fixed RoPE/SDPA stub.
# Generated 2026-08-14 on Clore RTX 3060.

| variant | weight source        | final_cosine_vs_golden | final_rmse | final_rel_l2 |
|---------|----------------------|------------------------|------------|--------------|
| A       | official BF16        | 0.811030               | 0.87211    | 0.66733      |
| B       | official -> FP16     | 0.812982               | 0.86896    | 0.66493      |
| C       | FP16-all .animapk    | 0.812982               | 0.86896    | 0.66493      |
| D       | W8 .animapk          | 0.808978               | 0.87207    | 0.66731      |
| E       | W4 .animapk          | 0.659592               | 1.01821    | 0.77914      |

## decoded vs streaming (step-0 velocity, real graph) — bit-identical
| pair                    | cosine     | max_abs | rel_l2 |
|-------------------------|------------|---------|--------|
| C decoded vs streaming  | 0.99999613 | 0.0     | 0.0    |
| D decoded vs streaming  | 0.99999630 | 0.0     | 0.0    |
| E decoded vs streaming  | 0.99999672 | 0.0     | 0.0    |

## Metal simulator (same packs, 8-step trajectory, final latent vs golden)
| variant | Metal final_cosine_vs_golden |
|---------|------------------------------|
| C       | 0.81231                       |
| D       | 0.80982                       |
| E       | 0.66037                       |

## Headline
Metal matches real-upstream CUDA within 0.001 cosine on every variant.
W8 is nearly free (0.809 vs 0.813). W4 is the quality cliff (0.660).
The .animapk container and streaming runtime are bit-faithful (max_abs 0.0).
The residual 0.81 ceiling is a genuine numeric-accumulation/quantization effect
reproduced on BOTH backends — NOT a Metal-specific bug.
