#!/usr/bin/env python3
"""Regression tests for the Anima/Wan21 sampler<->VAE latent-format boundary.

Covers instruction file section 18:
  18.1 process_out matches the pinned ComfyUI math for known 16-channel values
  18.2 process_in(process_out(x)) ~= x within dtype tolerance
  18.3 channel mapping: deliberately distinct channel values so a transpose/
       order mistake cannot accidentally pass
  18.4 (golden relationship) lives in scripts/animapk_cuda/wan21_boundary_proof.py
       A2 — it requires the golden npz; run it on Clore.

Run anywhere with numpy:  python3 scripts/test_wan21_latent_format.py
"""
from __future__ import annotations

import numpy as np

from animapk_cuda.wan21_latent_format import (
    LATENTS_MEAN, LATENTS_STD, SCALE_FACTOR,
    anima_sampler_to_vae, anima_vae_to_sampler,
)


def _assert_close(a: np.ndarray, b: np.ndarray, tol: float, msg: str) -> None:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
    assert d <= tol, f"{msg}: max abs diff {d} > {tol}"


def test_known_values() -> None:
    """18.1: process_out on a known 16-channel input matches the formula."""
    rng = np.random.default_rng(1337)
    x = rng.standard_normal((16, 8, 8)).astype(np.float32)
    got = anima_sampler_to_vae(x)
    exp = x.astype(np.float64) * LATENTS_STD[:, None, None] / SCALE_FACTOR \
        + LATENTS_MEAN[:, None, None]
    _assert_close(got, exp.astype(np.float32), 1e-6, "process_out formula")
    print("18.1 process_out formula: OK")


def test_zero_input_is_mean() -> None:
    """Sanity: process_out(0) == latents_mean exactly."""
    x = np.zeros((16, 4, 4), dtype=np.float32)
    got = anima_sampler_to_vae(x)
    _assert_close(got, LATENTS_MEAN[:, None, None].astype(np.float32),
                  1e-7, "process_out(0) == mean")
    print("18.1 process_out(0)==mean: OK")


def test_inverse() -> None:
    """18.2: process_in(process_out(x)) ~= x."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((16, 8, 8)).astype(np.float32) * 3.0
    roundtrip = anima_vae_to_sampler(anima_sampler_to_vae(x))
    _assert_close(roundtrip, x, 2e-5, "process_in(process_out(x)) ~= x")
    # also check both 5D and 4D layouts broadcast on the C axis
    x5 = x.reshape(1, 16, 1, 8, 8)
    rt5 = anima_vae_to_sampler(anima_sampler_to_vae(x5))
    _assert_close(rt5.reshape(16, 8, 8), x, 2e-5, "5D roundtrip")
    print("18.2 inverse (3D/5D): OK")


def test_channel_mapping() -> None:
    """18.3: distinct per-channel values — a transpose/order bug cannot pass.

    Each channel c gets a DIFFERENT constant value c+1. After process_out,
    channel c must equal (c+1)*std[c] + mean[c]. If channels were permuted or
    the affine broadcast along the wrong axis, this fails loudly.
    """
    x = np.zeros((16, 6, 6), dtype=np.float32)
    for c in range(16):
        x[c] = float(c + 1)
    got = anima_sampler_to_vae(x)
    for c in range(16):
        exp = (c + 1) * float(LATENTS_STD[c]) / SCALE_FACTOR + float(LATENTS_MEAN[c])
        assert abs(float(got[c, 0, 0]) - exp) < 1e-5, f"channel {c} mismatch"
    # sanity: all 16 channels differ from each other (affine is not degenerate)
    vals = {round(float(got[c, 0, 0]), 4) for c in range(16)}
    assert len(vals) == 16, "channels must be distinguishable after transform"
    print("18.3 channel mapping: OK (16 distinct channels, correct order)")


def test_rank5_rank3_agree() -> None:
    """Same math for [16,64,64] and [1,16,1,64,64] (the two fixture layouts)."""
    rng = np.random.default_rng(7)
    x3 = rng.standard_normal((16, 64, 64)).astype(np.float32)
    x5 = x3.reshape(1, 16, 1, 64, 64)
    _assert_close(anima_sampler_to_vae(x3),
                  anima_sampler_to_vae(x5).reshape(16, 64, 64),
                  1e-6, "3D vs 5D agree")
    print("layout consistency 3D/5D: OK")


if __name__ == "__main__":
    test_known_values()
    test_zero_input_is_mean()
    test_inverse()
    test_channel_mapping()
    test_rank5_rank3_agree()
    print("ALL wan21_latent_format tests PASSED")
