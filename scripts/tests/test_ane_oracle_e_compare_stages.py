from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from scripts.ane_oracle_e_compare_stages import STAGE_ORDER, compare


class OracleEStageCompareTests(unittest.TestCase):
    def _write_manifest(
        self,
        root: Path,
        *,
        perturb_stage: str | None = None,
        perturb: float = 0.0,
    ) -> Path:
        records = []
        for ordinal, name in enumerate(STAGE_ORDER):
            dtype = "float32-le" if name == "stage_b00_self_modulation" else "float16-le"
            extension = "f32" if dtype == "float32-le" else "f16"
            values = np.linspace(-1.0, 1.0, 16, dtype=np.float32) + ordinal * 0.1
            if name == perturb_stage:
                values = values + perturb
            filename = f"{name}.{extension}"
            if dtype == "float16-le":
                np.asarray(values, dtype="<f2").tofile(root / filename)
            else:
                np.asarray(values, dtype="<f4").tofile(root / filename)
            records.append({
                "name": name,
                "dtype": dtype,
                "element_count": 16,
                "file": filename,
            })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": 1,
            "expected_block0_stage_payloads": 9,
            "completed_block0_stage_payloads": 9,
            "block0_stage_records": records,
        }))
        return manifest

    def test_identity_has_no_stage_crossing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self._write_manifest(root)
            result = compare(manifest, manifest)
            self.assertEqual(result["stage_count"], 9)
            self.assertIsNone(result["first_stage_threshold_crossing"])
            for row in result["rows"]:
                self.assertEqual(row["relative_rmse"], 0.0)
                self.assertAlmostEqual(row["cosine"], 1.0, places=12)

    def test_first_perturbed_stage_is_localized(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ref_dir = root / "reference"
            cand_dir = root / "candidate"
            ref_dir.mkdir(); cand_dir.mkdir()
            reference = self._write_manifest(ref_dir)
            candidate = self._write_manifest(
                cand_dir,
                perturb_stage="stage_b00_self_q_raw",
                perturb=0.5,
            )
            result = compare(reference, candidate)
            crossing = result["first_stage_threshold_crossing"]
            self.assertIsNotNone(crossing)
            self.assertEqual(crossing["name"], "stage_b00_self_q_raw")
            self.assertGreater(crossing["relative_rmse"], 0.01)

    def test_incomplete_stage_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = self._write_manifest(root)
            data = json.loads(manifest.read_text())
            data["block0_stage_records"] = data["block0_stage_records"][:-1]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "incomplete stage manifests"):
                compare(manifest, manifest)


if __name__ == "__main__":
    unittest.main()
