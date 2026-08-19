#!/usr/bin/env bash
set -euo pipefail

# Run Oracle E on an already-prepared Oracle-V2 workspace.
# Usage:
#   run_ane_oracle_e.sh WORK_ROOT E1_native_ane /path/to/exact.animapk [run_id]
#   run_ane_oracle_e.sh WORK_ROOT E2_device_residual /path/to/exact.animapk [run_id]
#
# Optional physical-device parity override:
#   ANIMA_ORACLE_E_DEVICE_DIR=/path/to/unpacked-device-capture ...
# The directory must contain:
#   cross_context.f32
#   prepared_residual.f32
#   prepared_embedding.f32
#   prepared_adaln_lora.f32
#
# The exact A1 noise remains the V2 runner input even in device mode because
# E2 replaces the complete step-0 block-entry state before block 0. The raw A12
# initial latent remains in the `.oraclee` bundle for provenance/preparation
# analysis, but it is not needed to localize the 28 hybrid blocks.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 WORK_ROOT MODE PACK_PATH [RUN_ID]" >&2
  exit 2
fi

WORK_ROOT="$(cd "$1" && pwd)"
MODE="$2"
PACK_PATH="$(realpath "$3")"
RUN_ID="${4:-${MODE}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_DIR="${ANIMA_ORACLE_E_DEVICE_DIR:-}"

case "$MODE" in
  E1_native_ane|E2_device_residual) ;;
  *) echo "invalid mode: $MODE" >&2; exit 2 ;;
esac

V2_RUNNER="$WORK_ROOT/02_run_source.py"
COMFY="$WORK_ROOT/ComfyUI"
A1_NOISE="$WORK_ROOT/runs/A1_gold_capture/initial_noise.safetensors"
[[ -f "$V2_RUNNER" ]] || { echo "missing V2 runner: $V2_RUNNER" >&2; exit 1; }
[[ -d "$COMFY/custom_nodes" ]] || { echo "missing ComfyUI custom_nodes: $COMFY/custom_nodes" >&2; exit 1; }
[[ -f "$A1_NOISE" ]] || { echo "missing exact A1 noise: $A1_NOISE" >&2; exit 1; }
[[ -f "$PACK_PATH" ]] || { echo "missing ANIMAPK: $PACK_PATH" >&2; exit 1; }

NOISE="$A1_NOISE"
CROSS_CONTEXT=""
PREPARED_RESIDUAL=""
PREPARED_EMBEDDING=""
PREPARED_ADALN=""
INPUT_MODE="oracle_v2_a1"
if [[ -n "$DEVICE_DIR" ]]; then
  DEVICE_DIR="$(cd "$DEVICE_DIR" && pwd)"
  CROSS_CONTEXT="$DEVICE_DIR/cross_context.f32"
  PREPARED_RESIDUAL="$DEVICE_DIR/prepared_residual.f32"
  PREPARED_EMBEDDING="$DEVICE_DIR/prepared_embedding.f32"
  PREPARED_ADALN="$DEVICE_DIR/prepared_adaln_lora.f32"
  declare -A EXPECTED_BYTES=(
    ["$CROSS_CONTEXT"]="2097152"
    ["$PREPARED_RESIDUAL"]="8388608"
    ["$PREPARED_EMBEDDING"]="8192"
    ["$PREPARED_ADALN"]="24576"
  )
  for path in "$CROSS_CONTEXT" "$PREPARED_RESIDUAL" "$PREPARED_EMBEDDING" "$PREPARED_ADALN"; do
    [[ -f "$path" ]] || { echo "missing Oracle E device payload: $path" >&2; exit 1; }
    actual="$(stat -c %s "$path")"
    expected="${EXPECTED_BYTES[$path]}"
    [[ "$actual" == "$expected" ]] || {
      echo "device payload size mismatch: $path has $actual bytes, expected $expected" >&2
      exit 1
    }
  done
  INPUT_MODE="device_step0_override"
fi

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

# Avoid loading the V2 arm node alongside E. It installs global wrappers at
# import time. Move only packages that actually advertise AnimaOracleArm and
# restore them on exit.
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
cp "$SCRIPT_DIR/ane_oracle_e_package_init.py" "$E_NODE/__init__.py"
cp "$SCRIPT_DIR/ane_oracle_e_custom_node.py" "$E_NODE/ane_oracle_e_custom_node.py"
cp "$SCRIPT_DIR/ane_oracle_e_pack_reference.py" "$E_NODE/ane_oracle_e_pack_reference.py"

PATCHED_RUNNER="$WORK_ROOT/02_run_oracle_e.py"
python "$SCRIPT_DIR/ane_oracle_e_patch_v2_runner.py" "$V2_RUNNER" "$PATCHED_RUNNER"

RUN_DIR="$WORK_ROOT/runs/$RUN_ID"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
export ANIMA_ORACLE_E_RUN_DIR="$RUN_DIR"
export ANIMA_ORACLE_E_PACK="$PACK_PATH"
export ANIMA_ORACLE_E_PACK_SHA256="$EXPECTED_PACK_SHA"
export ANIMA_ORACLE_E_NOISE_MODE="load"
export ANIMA_ORACLE_E_NOISE_FILE="$NOISE"
export ANIMA_ORACLE_E_SAMPLE_COUNT="65536"
export ANIMA_ORACLE_E_OUTPUT_ROW_CHUNK="256"
export ANIMA_ORACLE_E_STOP_AFTER_STEP0="1"
if [[ -n "$CROSS_CONTEXT" ]]; then
  export ANIMA_ORACLE_E_CROSS_CONTEXT="$CROSS_CONTEXT"
  export ANIMA_ORACLE_E_PREPARED_RESIDUAL="$PREPARED_RESIDUAL"
  export ANIMA_ORACLE_E_PREPARED_EMBEDDING="$PREPARED_EMBEDDING"
  export ANIMA_ORACLE_E_PREPARED_ADALN_LORA="$PREPARED_ADALN"
else
  unset ANIMA_ORACLE_E_CROSS_CONTEXT || true
  unset ANIMA_ORACLE_E_PREPARED_RESIDUAL || true
  unset ANIMA_ORACLE_E_PREPARED_EMBEDDING || true
  unset ANIMA_ORACLE_E_PREPARED_ADALN_LORA || true
fi

{
  echo "mode=$MODE"
  echo "run_id=$RUN_ID"
  echo "input_mode=$INPUT_MODE"
  echo "work_root=$WORK_ROOT"
  echo "v2_runner=$V2_RUNNER"
  echo "patched_runner=$PATCHED_RUNNER"
  echo "pack=$PACK_PATH"
  echo "pack_bytes=$ACTUAL_BYTES"
  echo "pack_sha256=$ACTUAL_SHA"
  echo "noise=$NOISE"
  echo "cross_context=${CROSS_CONTEXT:-oracle_v2_pipeline}"
  echo "prepared_residual=${PREPARED_RESIDUAL:-oracle_v2_pipeline}"
  echo "prepared_embedding=${PREPARED_EMBEDDING:-oracle_v2_pipeline}"
  echo "prepared_adaln_lora=${PREPARED_ADALN:-oracle_v2_pipeline}"
  echo "stop_after_step0=1"
  echo "repo_or_bundle_script_dir=$SCRIPT_DIR"
} > "$RUN_DIR/ORACLE_E_PROVENANCE.txt"

# The two-anchor patched V2 runner intentionally receives a model exception
# after block 27 MLP. A nonzero runner code is therefore expected. We accept it
# ONLY when the custom node atomically produced both the complete 84-checkpoint
# manifest and the explicit STEP0_COMPLETE sentinel file.
set +e
python "$PATCHED_RUNNER" \
  --run-id "$RUN_ID" \
  --instrument "$MODE" \
  --model reconstructed \
  --noise load
RUNNER_RC=$?
set -e
printf 'v2_runner_exit_code=%s\n' "$RUNNER_RC" >> "$RUN_DIR/ORACLE_E_PROVENANCE.txt"

python - "$RUN_DIR/oracle_e_checkpoints.json" "$RUN_DIR/STEP0_COMPLETE" <<'PY'
import json, sys
from pathlib import Path
manifest=Path(sys.argv[1]); sentinel=Path(sys.argv[2])
if not manifest.is_file(): raise SystemExit(f"missing checkpoint manifest: {manifest}
")
d=json.loads(manifest.read_text())
if d.get("completed_step0_checkpoints") != 84 or not d.get("complete"):
    raise SystemExit(f"incomplete Oracle E checkpoints: {d.get('completed_step0_checkpoints')} / 84")
if not sentinel.is_file():
    raise SystemExit("84 checkpoints exist but intentional STEP0_COMPLETE sentinel is missing")
text=sentinel.read_text().strip()
if text != "__ANIMA_ORACLE_E_STEP0_COMPLETE__":
    raise SystemExit(f"unexpected Oracle E sentinel: {text!r}")
print(f"Oracle E step-0 gate PASS: mode={d['mode']} count=84 intentional_stop=yes")
PY
