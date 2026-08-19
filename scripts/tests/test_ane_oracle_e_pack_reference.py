from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from scripts.ane_oracle_e_pack_reference import (
    ANE_QUANT_SCHEME,
    ANE_TENSOR_FORMAT,
    AnimapkReference,
    checkpoint_sample_numpy,
    checkpoint_stats_numpy,
    compare_samples,
    native_linear_numpy,
)


class OracleEPackReferenceTests(unittest.TestCase):
    def _write_tiny_pack(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([[0, 10, 255], [4, 5, 6]], dtype=np.uint8)
        scale = np.array([0.25, 2.0], dtype="<f4")
        bias = np.array([-1.0, 3.0], dtype="<f4")
        blob_offset = 16_384
        data = q.tobytes(order="C")
        scale_bytes = scale.tobytes()
        bias_bytes = bias.tobytes()
        item = {
            "name": "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            "shape": [2, 3],
            "storage_dtype": "w8",
            "quantization_format": ANE_TENSOR_FORMAT,
            "blob_offset": blob_offset,
            "blob_size": len(data) + len(scale_bytes) + len(bias_bytes),
            "data_offset": 0,
            "data_size": len(data),
            "scale_offset": len(data),
            "scale_size": len(scale_bytes),
            "zero_offset": len(data) + len(scale_bytes),
            "zero_size": len(bias_bytes),
        }
        metadata = {
            "component": "dit",
            "profile": "ane-hybrid-w8-v1",
            "quant": {"scheme": ANE_QUANT_SCHEME, "group": 64},
            "tensor_meta": [item],
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        json_offset = 256
        with path.open("w+b") as f:
            f.truncate(blob_offset + item["blob_size"])
            f.seek(0)
            header = bytearray(256)
            header[:4] = b"ANMA"
            struct.pack_into("<H", header, 4, 1)
            struct.pack_into("<Q", header, 20, json_offset)
            struct.pack_into("<Q", header, 28, len(encoded))
            f.write(header)
            f.seek(json_offset)
            f.write(encoded)
            f.seek(blob_offset)
            f.write(data + scale_bytes + bias_bytes)
        return q, scale, bias

    def test_native_reader_and_linear_are_exact_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.animapk"
            q, scale, bias = self._write_tiny_pack(path)
            name = "model.diffusion_model.blocks.0.self_attn.q_proj.weight"
            with AnimapkReference(path) as pack:
                got_q, got_scale, got_bias = pack.native_rows(name)
                np.testing.assert_array_equal(got_q, q)
                np.testing.assert_array_equal(got_scale, scale)
                np.testing.assert_array_equal(got_bias, bias)
                expected_w = q.astype(np.float32) * scale[:, None] + bias[:, None]
                np.testing.assert_array_equal(pack.native_reconstructed_rows(name), expected_w)

                x = np.array([[1.1, -2.2, 0.3]], dtype=np.float32)
                expected_x = x.astype(np.float16).astype(np.float32)
                expected = expected_x @ expected_w.T
                expected = expected.astype(np.float16).astype(np.float32)
                got = native_linear_numpy(pack, name, x, output_row_chunk=1)
                np.testing.assert_array_equal(got, expected)

    def test_checkpoint_sampling_and_stats_are_deterministic(self):
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        sample, stride = checkpoint_sample_numpy(values, sample_count=10)
        self.assertEqual(stride, 10)
        np.testing.assert_array_equal(sample, np.arange(0, 100, 10, dtype=np.float32))
        stats = checkpoint_stats_numpy(values)
        self.assertEqual(stats["element_count"], 100)
        self.assertEqual(stats["finite_count"], 100)
        self.assertEqual(stats["nan_count"], 0)
        self.assertEqual(stats["max_abs"], 99.0)

    def test_compare_samples_reports_zero_for_identity(self):
        values = np.linspace(-4, 4, 101, dtype=np.float32)
        metrics = compare_samples(values, values.copy())
        self.assertEqual(metrics["relative_rmse"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0, places=12)
        self.assertEqual(metrics["max_abs_error"], 0.0)

    def test_compare_samples_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            compare_samples(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
