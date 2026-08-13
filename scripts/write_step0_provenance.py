"""Write provenance.json for the step-0 backend matrix Metal job (called from
quality-step0-backend-matrix.yml)."""

import json
import os
import sys


def main():
    out_dir = os.environ.get("FIXTURE_DIR", ".")
    prov = {
        "commit": os.environ.get("GITHUB_SHA", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "ref": os.environ.get("GITHUB_REF_NAME", ""),
        "variant": os.environ.get("MATRIX_VARIANT", ""),
        "case": "case1_danbooru_seed1337",
        "golden_qwen_context": "1",
        "golden_dit_context": "1",
        "dit": {
            "filename": os.environ.get("DIT_FILE", ""),
            "sha256": sys.argv[1] if len(sys.argv) > 1 else "",
            "source_revision": os.environ.get("SOURCE_REVISION", ""),
            "source_sha256": os.environ.get("SOURCE_SHA256", ""),
        },
        "qwen": {
            "filename": os.environ.get("QWEN_FILE", ""),
            "sha256": os.environ.get("QWEN_SHA256", ""),
        },
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    print(f"provenance variant={prov['variant']} dit_sha={prov['dit']['sha256']}")


if __name__ == "__main__":
    main()
