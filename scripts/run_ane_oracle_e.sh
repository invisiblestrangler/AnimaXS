#!/usr/bin/env bash
set -euo pipefail

# Run Oracle E on an already-prepared Oracle-V2 workspace.
# Usage:
#   run_ane_oracle_e.sh WORK_ROOT E1_native_ane /path/to/exact.animapk [run_id]
#   run_ane_oracle_e.sh WORK_ROOT E2_device_residual /path/to/exact.animapk [run_id]
#
# WORK_ROOT must contain the successful Oracle-V2 02_run_source.py, ComfyUI,
# reconstructed checkpoint, and A1 initial_noise.safetensors.  This script does
# not redownload/rebuild those assets and never edits the original V2 runner.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 WORK_ROOT MODE PACK_PATH [RUN_ID]" >&2
  exit 2
fi

WORK_ROOT="$(cd "$1" && pwd)"
MODE="$2"
PACK_PATH="$(realpath "$3")"
RUN_ID="${4:-${MODE}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$MODE" in
  E1_native_ane|E2_device_residual) ;;
  *) echo "invalid mode: $MODE" >&2; exit 2 ;;
esac

V2_RUNNER="$WORK_ROOT/02_run_source.py"
COMFY="$WORK_ROOT/ComfyUI"
NOISE="$WORK_ROOT/runs/A1_gold_capture/initial_noise.safetensors"
[[ -f "$V2_RUNNER" ]] || { echo "missing V2 runner: $V2_RUNNER" >&2; exit 1; }
[[ -d "$COMFY/custom_nodes" ]] || { echo "missing ComfyUI custom_nodes: $COMFY/custom_nodes" >&2; exit 1; }
[[ -f "$NOISE" ]] || { echo "missing exact A1 noise: $NOISE" >&2; exit 1; }
[[ -f "$PACK_PATH" ]] || { echo "missing ANIMAPK: $PACK_PATH" >&2; exit 1; }

EXPECTED_PACK_SHA="f5c80a25114b62a6807996180d439c5d12828d7392c604e1eee15acb28977dc4"
EXPECTED_PACK_BYTES="2128838656"
ACTUAL_BYTES="$(stat -c %s "$PACK_PATH")"
ACTUAL_SHA="$(sha256sum "$PACK_PATH" | awk '{print $1}')"
[[ "$ACTUAL_BYTES" == "$EXPECTED_PACK_BYTES" ]] || {
  echo "pack size mismatch: $ACTUAL_BYTES != $EXPECTED_PACK_BYTES" >&2; exit 1;
}
[[ "$ACTUAL_SHA" == "$EXPECTED_PACK_SHA" ]] || {
  echo "pack sha256 mismatch: $ACTUAL_SHA != $EXPECTED_PACK_SHA" >&2; exit 1;
}

# Avoid loading the V2 arm node alongside E.  It globally wraps sampler noise
# and VAE hooks at import time even when its graph node is not selected.  Move
# only packages that actually advertise AnimaOracleArm; restore them on exit.
DISABLED="$WORK_ROOT/.oracle_e_disabled_custom_nodes"
mkdir -p "$DISABLED"
RESTORE_LIST="$DISABLED/restore.tsv"
: > "$RESTORE_LIST"
while IFS= read -r hit; do
  [[ -z "$hit" ]] && continue
  node_dir="$(dirname "$hit")"
  [[ "$node_dir" == "$COMFY/custom_nodes/anima_oracle_e" ]] && continue
  base="$(basename "$node_dir")"
  target="$DISABLED/${base}.$(date +%s%N)"
  mv "$node_dir" "$target"
  printf '%s\t%s\n' "$target" "$node_dir" >> "$RESTORE_LIST"
done < <(grep -RIl --include='*.py' 'AnimaOracleArm' "$COMFY/custom_nodes" 2>/dev/null || true)

restore_nodes() {
  if [[ -f "$RESTORE_LIST" ]]; then
    while IFS=$'\t' read -r source target; do
      [[ -z "$source" ]] && continue
      if [[ -e "$source" && ! -e "$target" ]]; then
        mv "$source" "$target"
      fi
    done < "$RESTORE_LIST"
  fi
}
trap restore_nodes EXIT

E_NODE="$COMFY/custom_nodes/anima_oracle_e"
rm -rf "$E_NODE"
mkdir -p "$E_NODE"
cp "$SCRIPT_DIR/ane_oracle_e_custom_node.py" "$E_NODE/__init__.py"
cp "$SCRIPT_DIR/ane_oracle_e_pack_reference.py" "$E_NODE/ane_oracle_e_pack_reference.py"

PATCHED_RUNNER="$WORK_ROOT/02_run_oracle_e.py"
python "$SCRIPT_DIR/ane_oracle_e_patch_v2_runner.py" "$V2_RUNNER" "$PATCHED_RUNNER"

RUN_DIR="$WORK_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export ANIMA_ORACLE_E_RUN_DIR="$RUN_DIR"
export ANIMA_ORACLE_E_PACK="$PACK_PATH"
export ANIMA_ORACLE_E_PACK_SHA256="$EXPECTED_PACK_SHA"
export ANIMA_ORACLE_E_NOISE_MODE="load"
export ANIMA_ORACLE_E_NOISE_FILE="$NOISE"
export ANIMA_ORACLE_E_SAMPLE_COUNT="65536"
export ANIMA_ORACLE_E_OUTPUT_ROW_CHUNK="256"

{
  echo "mode=$MODE"
  echo "run_id=$RUN_ID"
  echo "work_root=$WORK_ROOT"
  echo "v2_runner=$V2_RUNNER"
  echo "patched_runner=$PATCHED_RUNNER"
  echo "pack=$PACK_PATH"
  echo "pack_bytes=$ACTUAL_BYTES"
  echo "pack_sha256=$ACTUAL_SHA"
  echo "noise=$NOISE"
  echo "repo_or_bundle_script_dir=$SCRIPT_DIR"
} > "$RUN_DIR/ORACLE_E_PROVENANCE.txt"

# The V2 runner's reconstructed-model branch is deliberately retained.  Its
# proven graph/prompt/sampler/download logic is unchanged apart from the two
# guarded patch anchors.
python "$PATCHED_RUNNER" \
  --run-id "$RUN_ID" \
  --instrument "$MODE" \
  --model reconstructed \
  --noise load

# Fail closed if the step-0 capture is incomplete.
python - "$RUN_DIR/oracle_e_checkpoints.json" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"missing checkpoint manifest: {p}")
d=json.loads(p.read_text())
if d.get("completed_step0_checkpoints") != 84 or not d.get("complete"):
    raise SystemExit(f"incomplete Oracle E checkpoints: {d.get('completed_step0_checkpoints')} / 84")
print(f"Oracle E checkpoint gate PASS: mode={d['mode']} count=84")
PY
