from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from scripts.ane_oracle_e_pack_reference import (
    ANE_QUANT_SCHEME,
    ANE_TENSOR_FORMAT,
    GROUP64_TENSOR_FORMAT,
    AnimapkReference,
    checkpoint_sample_numpy,
    checkpoint_stats_numpy,
    compare_samples,
    group64_matvec_numpy,
    native_linear_numpy,
)


class OracleEPackReferenceTests(unittest.TestCase):
    def _write_tiny_pack(self, path: Path) -> dict[str, object]:
        native_q = np.array([[0, 10, 255], [4, 5, 6]], dtype=np.uint8)
        native_scale = np.array([0.25, 2.0], dtype="<f4")
        native_bias = np.array([-1.0, 3.0], dtype="<f4")
        native_blob = (
            native_q.tobytes(order="C")
            + native_scale.tobytes()
            + native_bias.tobytes()
        )

        group_q = np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], dtype=np.uint8)
        # Values deliberately need real FP16 representation, not exact FP32
        # decimals, so the test locks promotion of stored FP16 metadata.
        group_scale = np.array([[0.1], [0.3]], dtype="<f2")
        group_zero = np.array([[-1.25], [2.5]], dtype="<f2")
        group_blob = (
            group_q.tobytes(order="C")
            + group_scale.tobytes()
            + group_zero.tobytes()
        )

        norm = np.array([1.0, 1.0009765625, -0.333251953125], dtype="<f2")
        norm_blob = norm.tobytes()

        blob_offset = 16_384
        native_offset = blob_offset
        group_offset = native_offset + len(native_blob)
        norm_offset = group_offset + len(group_blob)

        native_item = {
            "name": "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
            "shape": [2, 3],
            "storage_dtype": "w8",
            "quantization_format": ANE_TENSOR_FORMAT,
            "blob_offset": native_offset,
            "blob_size": len(native_blob),
            "data_offset": 0,
            "data_size": native_q.nbytes,
            "scale_offset": native_q.nbytes,
            "scale_size": native_scale.nbytes,
            "zero_offset": native_q.nbytes + native_scale.nbytes,
            "zero_size": native_bias.nbytes,
        }
        group_item = {
            "name": "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
            "shape": [2, 5],
            "storage_dtype": "w8",
            "quantization_format": GROUP64_TENSOR_FORMAT,
            "blob_offset": group_offset,
            "blob_size": len(group_blob),
            "data_offset": 0,
            "data_size": group_q.nbytes,
            "scale_offset": group_q.nbytes,
            "scale_size": group_scale.nbytes,
            "zero_offset": group_q.nbytes + group_scale.nbytes,
            "zero_size": group_zero.nbytes,
        }
        norm_item = {
            "name": "model.diffusion_model.blocks.0.self_attn.q_norm.weight",
            "shape": [3],
            "storage_dtype": "fp16",
            "quantization_format": None,
            "blob_offset": norm_offset,
            "blob_size": len(norm_blob),
            "data_offset": 0,
            "data_size": len(norm_blob),
            "scale_offset": 0,
            "scale_size": 0,
            "zero_offset": 0,
            "zero_size": 0,
        }
        metadata = {
            "component": "dit",
            "profile": "ane-hybrid-w8-v1",
            "quant": {"scheme": ANE_QUANT_SCHEME, "group": 64},
            "tensor_meta": [native_item, group_item, norm_item],
        }
        encoded = json.dumps(metadata, separators=(",", ":")).encode()
        json_offset = 256
        with path.open("w+b") as f:
            f.truncate(norm_offset + len(norm_blob))
            header = bytearray(256)
            header[:4] = b"ANMA"
            struct.pack_into("<H", header, 4, 1)
            struct.pack_into("<Q", header, 20, json_offset)
            struct.pack_into("<Q", header, 28, len(encoded))
            f.seek(0); f.write(header)
            f.seek(json_offset); f.write(encoded)
            f.seek(native_offset); f.write(native_blob)
            f.seek(group_offset); f.write(group_blob)
            f.seek(norm_offset); f.write(norm_blob)
        return {
            "native_q": native_q,
            "native_scale": native_scale,
            "native_bias": native_bias,
            "group_q": group_q,
            "group_scale": group_scale,
            "group_zero": group_zero,
            "norm": norm,
        }

    def test_native_reader_and_linear_are_exact_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.animapk"
            expected = self._write_tiny_pack(path)
            name = "model.diffusion_model.blocks.0.self_attn.q_proj.weight"
            with AnimapkReference(path) as pack:
                q, scale, bias = pack.native_rows(name)
                np.testing.assert_array_equal(q, expected["native_q"])
                np.testing.assert_array_equal(scale, expected["native_scale"])
                np.testing.assert_array_equal(bias, expected["native_bias"])
                expected_w = (
                    expected["native_q"].astype(np.float32)
                    * expected["native_scale"][:, None]
                    + expected["native_bias"][:, None]
                )
                np.testing.assert_array_equal(pack.native_reconstructed_rows(name), expected_w)

                x = np.array([[1.1, -2.2, 0.3]], dtype=np.float32)
                expected_x = x.astype(np.float16).astype(np.float32)
                expected_out = expected_x @ expected_w.T
                expected_out = expected_out.astype(np.float16).astype(np.float32)
                got = native_linear_numpy(pack, name, x, output_row_chunk=1)
                np.testing.assert_array_equal(got, expected_out)

    def test_group64_reader_promotes_exact_fp16_metadata_before_math(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.animapk"
            expected = self._write_tiny_pack(path)
            name = "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight"
            with AnimapkReference(path) as pack:
                q, scale, zero = pack.group64_rows(name)
                np.testing.assert_array_equal(q, expected["group_q"])
                np.testing.assert_array_equal(scale.view(np.uint16), expected["group_scale"].view(np.uint16))
                np.testing.assert_array_equal(zero.view(np.uint16), expected["group_zero"].view(np.uint16))

                expected_w = (
                    expected["group_q"].astype(np.float32)
                    * expected["group_scale"].astype(np.float32)
                    + expected["group_zero"].astype(np.float32)
                )
                np.testing.assert_array_equal(pack.group64_reconstructed_rows(name), expected_w)

                x = np.array([[0.5, -1.0, 2.0, 0.25, -0.75]], dtype=np.float32)
                got = group64_matvec_numpy(pack, name, x, output_row_chunk=1)
                np.testing.assert_allclose(got, x @ expected_w.T, rtol=0, atol=0)

    def test_fp16_tensor_preserves_exact_payload_bits(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.animapk"
            expected = self._write_tiny_pack(path)
            name = "model.diffusion_model.blocks.0.self_attn.q_norm.weight"
            with AnimapkReference(path) as pack:
                got = pack.fp16_tensor(name)
                self.assertEqual(got.dtype, np.dtype("<f2"))
                np.testing.assert_array_equal(got.view(np.uint16), expected["norm"].view(np.uint16))

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
