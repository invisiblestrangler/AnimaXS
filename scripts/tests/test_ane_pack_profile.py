from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pack_anima import (
    ANE_HYBRID_PROFILE,
    ANE_TENSOR_FORMAT,
    GROUP64_TENSOR_FORMAT,
    _quantize_ane_row_chunk,
    is_ane_projection,
    make_plan,
)


class ANEPackProfileTests(unittest.TestCase):
    def test_projection_selector_exact(self):
        self.assertTrue(is_ane_projection("model.diffusion_model.blocks.0.self_attn.q_proj.weight"))
        self.assertTrue(is_ane_projection("model.diffusion_model.blocks.27.mlp.layer2.weight"))
        self.assertFalse(is_ane_projection("model.diffusion_model.blocks.28.self_attn.q_proj.weight"))
        self.assertFalse(is_ane_projection("model.diffusion_model.blocks.0.self_attn.q_norm.weight"))

    def test_per_row_quantization_general_and_constant(self):
        values = np.array([
            [-2.0, -1.0, 0.0, 2.0],
            [3.25, 3.25, 3.25, 3.25],
        ], dtype=np.float32)
        q_bytes, scale_bytes, bias_bytes, stats = _quantize_ane_row_chunk(values)
        q = np.frombuffer(q_bytes, dtype=np.uint8).reshape(values.shape)
        scale = np.frombuffer(scale_bytes, dtype="<f4")
        bias = np.frombuffer(bias_bytes, dtype="<f4")
        self.assertEqual(q.shape, values.shape)
        self.assertAlmostEqual(float(scale[0]), 4.0 / 255.0, places=7)
        self.assertAlmostEqual(float(bias[0]), -2.0, places=7)
        self.assertEqual(float(scale[1]), 1.0)
        self.assertEqual(float(bias[1]), 3.25)
        self.assertTrue(np.all(q[1] == 0))
        recon = q.astype(np.float32) * scale[:, None] + bias[:, None]
        self.assertTrue(np.isfinite(recon).all())
        self.assertEqual(stats["element_count"], 8.0)

    def test_per_row_quantization_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            _quantize_ane_row_chunk(np.array([[0.0, np.inf]], dtype=np.float32))

    def test_hybrid_plan_selects_exact_280_and_sizes(self):
        header = {}
        suffixes = [
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.output_proj.weight", "cross_attn.q_proj.weight",
            "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
            "cross_attn.output_proj.weight", "mlp.layer1.weight", "mlp.layer2.weight",
        ]
        # Tiny synthetic shapes: selection/count/layout is what matters here.
        for block in range(28):
            for suffix in suffixes:
                header[f"model.diffusion_model.blocks.{block}.{suffix}"] = {
                    "shape": [4, 3], "dtype": "BF16", "data_offsets": [0, 24]
                }
            header[f"model.diffusion_model.blocks.{block}.adaln_modulation_mlp.1.weight"] = {
                "shape": [2, 3], "dtype": "BF16", "data_offsets": [0, 12]
            }
            header[f"model.diffusion_model.blocks.{block}.self_attn.q_norm.weight"] = {
                "shape": [2], "dtype": "BF16", "data_offsets": [0, 4]
            }
        plans = make_plan(header, "w8", [], 64, 8192, ANE_HYBRID_PROFILE)
        native = [p for p in plans if p["quantization_format"] == ANE_TENSOR_FORMAT]
        self.assertEqual(len(native), 280)
        for p in native:
            rows, cols = p["shape"]
            self.assertEqual(p["data_size"], rows * cols)
            self.assertEqual(p["scale_size"], rows * 4)
            self.assertEqual(p["zero_size"], rows * 4)
        ordinary = [p for p in plans if p["storage_dtype"] == "w8" and p not in native]
        self.assertTrue(ordinary)
        self.assertTrue(all(p["quantization_format"] == GROUP64_TENSOR_FORMAT for p in ordinary))
        for block in range(28):
            bp = [p for p in plans if p["block_index"] == block]
            metal = [p for p in bp if p["quantization_format"] != ANE_TENSOR_FORMAT]
            ane = [p for p in bp if p["quantization_format"] == ANE_TENSOR_FORMAT]
            self.assertLessEqual(
                max(p["blob_offset"] + p["blob_size"] for p in metal),
                min(p["blob_offset"] for p in ane),
            )

class ANEVerifierIntegrationTests(unittest.TestCase):
    def _synthetic_header(self):
        suffixes = [
            "adaln_modulation_self_attn.1.weight", "adaln_modulation_self_attn.2.weight",
            "adaln_modulation_cross_attn.1.weight", "adaln_modulation_cross_attn.2.weight",
            "adaln_modulation_mlp.1.weight", "adaln_modulation_mlp.2.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.output_proj.weight", "cross_attn.q_proj.weight",
            "cross_attn.k_proj.weight", "cross_attn.v_proj.weight",
            "cross_attn.output_proj.weight", "mlp.layer1.weight", "mlp.layer2.weight",
        ]
        return {
            f"model.diffusion_model.blocks.{block}.{suffix}": {
                "shape": [2, 2], "dtype": "BF16", "data_offsets": [0, 8]
            }
            for block in range(28) for suffix in suffixes
        }

    def _write_pack(self, path):
        import hashlib, os, struct, zlib
        from scripts.pack_anima import (
            build_metadata, build_table, write_header_and_metadata,
            HEADER_SIZE, RECORD_SIZE, align_up,
        )
        header = self._synthetic_header()
        plans = make_plan(header, "w8", [], 64, 1 << 20, ANE_HYBRID_PROFILE)
        payload_offset = align_up(HEADER_SIZE + (1 << 20) + len(plans) * RECORD_SIZE)
        file_size = payload_offset + sum(int(p["blob_size"]) for p in plans)
        with open(path, "w+b") as dst:
            dst.truncate(file_size)
            for plan in plans:
                if plan["quantization_format"] == ANE_TENSOR_FORMAT:
                    data = bytes([0, 64, 128, 255])
                    scale = struct.pack("<2f", 1.0, 1.0)
                    zero = struct.pack("<2f", 0.0, 0.0)
                else:
                    data = bytes([0, 0, 0, 0])
                    scale = struct.pack("<2e", 1.0, 1.0)
                    zero = struct.pack("<2e", 0.0, 0.0)
                self.assertEqual(len(data), plan["data_size"])
                self.assertEqual(len(scale), plan["scale_size"])
                self.assertEqual(len(zero), plan["zero_size"])
                dst.seek(plan["blob_offset"])
                dst.write(data + scale + zero)
                plan["crc32"] = zlib.crc32(data) & 0xFFFFFFFF
                plan["blob_sha256"] = hashlib.sha256(data + scale + zero).hexdigest()
            metadata = build_metadata(
                plans, "dit", "w8", 64, "synthetic.safetensors", "0" * 64,
                "test/repo", "test-revision", None, "1" * 64, "test",
                np.__version__, "test", "test", "mseclip", ANE_HYBRID_PROFILE)
            table = build_table(plans)
            write_header_and_metadata(
                dst, metadata, table, "dit", len(plans), 1 << 20,
                payload_offset, file_size)
        return plans

    def test_independent_verifier_accepts_synthetic_hybrid_pack(self):
        import tempfile
        from scripts.verify_animapk import verify_file
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "synthetic.animapk")
            self._write_pack(path)
            result = verify_file(path)
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["ane_native_tensor_count"], 280)
            self.assertTrue(result["metal_only_block_layout"])

    def test_standard_plan_has_no_hybrid_metadata(self):
        header = {
            "model.diffusion_model.blocks.0.self_attn.q_proj.weight": {
                "shape": [2, 2], "dtype": "BF16", "data_offsets": [0, 8]
            }
        }
        plan = make_plan(header, "w8", [], 64, 8192)[0]
        self.assertIsNone(plan["quantization_format"])

if __name__ == "__main__":
    unittest.main()
