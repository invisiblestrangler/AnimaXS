from pathlib import Path
import json
import struct
import tempfile
import unittest

import numpy as np

from scripts.ane_oracle_e_compare import compare
from scripts.ane_oracle_e_unpack_device_capture import MAGIC, HEADER_BYTES, unpack_capture


class OracleECaptureAndCompareTests(unittest.TestCase):
    def _write_capture(self, path: Path, *, omit_payload: str | None = None) -> None:
        payloads = []
        checkpoints = []
        raw = bytearray(b"\x00" * HEADER_BYTES)
        raw[0:8] = MAGIC
        struct.pack_into("<I", raw, 8, 1)

        def append(name: str, values: np.ndarray, shape: list[int]):
            if name == omit_payload:
                return
            offset = len(raw)
            blob = np.asarray(values, dtype="<f4").tobytes()
            raw.extend(blob)
            payloads.append({
                "name": name,
                "offset": offset,
                "byteCount": len(blob),
                "elementCount": int(values.size),
                "dtype": "float32-le",
                "shape": shape,
            })

        append("initial_latent", np.zeros(16 * 64 * 64, dtype=np.float32), [1, 16, 64, 64])
        append("cross_context", np.zeros(512 * 1024, dtype=np.float32), [1, 512, 1024])
        append("prepared_residual", np.zeros(1024 * 2048, dtype=np.float32), [1024, 2048])
        append("prepared_embedding", np.zeros(2048, dtype=np.float32), [2048])
        append("prepared_adaln_lora", np.zeros(6144, dtype=np.float32), [6144])

        for block in range(28):
            for branch_index, branch in enumerate(("self", "cross", "mlp")):
                sample = np.arange(8, dtype=np.float32) + block * 0.01 + branch_index * 0.001
                offset = len(raw)
                blob = sample.astype("<f4").tobytes()
                raw.extend(blob)
                checkpoints.append({
                    "step": 0,
                    "block": block,
                    "branch": branch,
                    "offset": offset,
                    "byteCount": len(blob),
                    "sampleCount": int(sample.size),
                    "sampleStride": 32,
                    "sampleDtype": "float32-le",
                    "residualElements": 256,
                    "sampleStats": {},
                })

        manifest = {
            "schema": 1,
            "producer": "synthetic",
            "status": "completed",
            "error": None,
            "seed": 123,
            "ditVariantID": "w8-ane-v1",
            "linearBackend": "aneHybridW8",
            "pingPongWeightStreaming": False,
            "conditioningSource": "synthetic",
            "initialLatentSource": "synthetic",
            "preparedStateSource": "synthetic",
            "completedStep0Checkpoints": 84,
            "payloads": payloads,
            "checkpoints": checkpoints,
        }
        manifest_blob = json.dumps(manifest).encode()
        manifest_offset = len(raw)
        raw.extend(manifest_blob)
        struct.pack_into("<Q", raw, 16, manifest_offset)
        struct.pack_into("<Q", raw, 24, len(manifest_blob))
        path.write_bytes(raw)

    def _write_reference_manifest(self, root: Path, perturb: tuple[int, str, float] | None = None) -> Path:
        records = []
        for block in range(28):
            for branch_index, branch in enumerate(("self", "cross", "mlp")):
                sample = np.arange(8, dtype=np.float32) + block * 0.01 + branch_index * 0.001
                if perturb and (block, branch) == perturb[:2]:
                    sample = sample + perturb[2]
                filename = f"step00_block{block:02d}_{branch}.f32"
                sample.astype("<f4").tofile(root / filename)
                records.append({
                    "step": 0,
                    "block": block,
                    "branch": branch,
                    "sample_file": filename,
                    "sample_count": int(sample.size),
                    "sample_stride": 32,
                })
        path = root / "manifest.json"
        path.write_text(json.dumps({"schema": 1, "mode": "test", "records": records}))
        return path

    def test_unpack_validates_and_extracts_complete_capture(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            capture = root / "test.oraclee"
            self._write_capture(capture)
            out = root / "out"
            manifest = unpack_capture(capture, out)
            self.assertEqual(manifest["completed_step0_checkpoints"], 84)
            self.assertEqual((out / "initial_latent.f32").stat().st_size, 16 * 64 * 64 * 4)
            self.assertEqual((out / "cross_context.f32").stat().st_size, 512 * 1024 * 4)
            self.assertEqual((out / "prepared_residual.f32").stat().st_size, 1024 * 2048 * 4)
            self.assertEqual((out / "prepared_embedding.f32").stat().st_size, 2048 * 4)
            self.assertEqual((out / "prepared_adaln_lora.f32").stat().st_size, 6144 * 4)
            self.assertEqual(manifest["prepared_state_source"], "synthetic")
            self.assertTrue((out / "step00_block27_mlp.f32").is_file())

    def test_unpack_rejects_missing_prepared_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            capture = root / "missing.oraclee"
            self._write_capture(capture, omit_payload="prepared_embedding")
            with self.assertRaisesRegex(ValueError, "missing device payloads"):
                unpack_capture(capture, root / "out")

    def test_unpack_rejects_bad_manifest_range(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.oraclee"
            raw = bytearray(b"\x00" * HEADER_BYTES)
            raw[:8] = MAGIC
            struct.pack_into("<I", raw, 8, 1)
            struct.pack_into("<Q", raw, 16, HEADER_BYTES + 100)
            struct.pack_into("<Q", raw, 24, 1000)
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                unpack_capture(path, Path(td) / "out")

    def test_compare_identity_is_zero_everywhere(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_dir = root / "ref"
            cand_dir = root / "cand"
            ref_dir.mkdir(); cand_dir.mkdir()
            ref = self._write_reference_manifest(ref_dir)
            cand = self._write_reference_manifest(cand_dir)
            result = compare(ref, cand, rel_threshold=0.1, cosine_threshold=0.99)
            self.assertIsNone(result["first_threshold_crossing"])
            self.assertEqual(result["checkpoint_count"], 84)
            for row in result["rows"]:
                self.assertEqual(row["relative_rmse"], 0.0)
                self.assertAlmostEqual(row["cosine"], 1.0, places=12)

    def test_compare_localizes_known_first_perturbation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_dir = root / "ref"
            cand_dir = root / "cand"
            ref_dir.mkdir(); cand_dir.mkdir()
            ref = self._write_reference_manifest(ref_dir)
            cand = self._write_reference_manifest(cand_dir, perturb=(5, "cross", 10.0))
            result = compare(ref, cand, rel_threshold=0.1, cosine_threshold=0.99)
            crossing = result["first_threshold_crossing"]
            self.assertIsNotNone(crossing)
            self.assertEqual(crossing["checkpoint"], "b05.cross")
            self.assertGreater(crossing["relative_rmse"], 0.1)


if __name__ == "__main__":
    unittest.main()
