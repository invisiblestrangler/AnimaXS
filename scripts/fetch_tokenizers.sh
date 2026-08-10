#!/usr/bin/env bash
# Regenerate tokenizer data for the app bundle from the pinned reference assets.
# Produces flat unique-named tokenizer.json files (no Xcode resource collisions).
# Requires a Python env with `tokenizers` + `transformers` (or run the pip install below).
set -euo pipefail
cd "$(dirname "$0")/.."

REF="${ANIMAXS_REF_TEXT_ENCODERS:-/root/comfy-ref/comfy/text_encoders}"
DEST="AnimaXS/Resources/Tokenizers"
mkdir -p "$DEST"

PY="${ANIMAXS_PYTHON:-.venv/bin/python}"
command -v "$PY" >/dev/null || { echo "python not found: $PY (set ANIMAXS_PYTHON)"; exit 1; }

# 1) Qwen: build tokenizer.json with the EXACT Qwen2Tokenizer pre-tokenization regex
#    (Split(PRETOKENIZE_REGEX) + ByteLevel(use_regex=False)) so IDs match the reference.
"$PY" - "$REF" "$DEST/qwen_tokenizer.json" <<'PYEOF'
import sys, json
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split, ByteLevel, Sequence
from tokenizers.decoders import ByteLevel as BLDecoder
from tokenizers.normalizers import NFC

ref, out = sys.argv[1], sys.argv[2]
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
vocab = json.load(open(ref + "/qwen25_tokenizer/vocab.json"))
merges = [tuple(l.split()) for l in open(ref + "/qwen25_tokenizer/merges.txt") if l.strip() and not l.startswith("#")]
tok = Tokenizer(BPE(vocab=vocab, merges=merges))
tok.decoder = BLDecoder()
tok.normalizer = NFC()
tok.pre_tokenizer = Sequence([
    Split(Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
    ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.save(out)
print("wrote", out, "bytes:", json.load(open(out)) and __import__("os").path.getsize(out))
PYEOF

# 2) T5: copy tokenizer.json + config as flat unique names.
cp "$REF/t5_tokenizer/tokenizer.json" "$DEST/t5_tokenizer.json"
cp "$REF/t5_tokenizer/tokenizer_config.json" "$DEST/t5_tokenizer_config.json"
cp "$REF/t5_tokenizer/special_tokens_map.json" "$DEST/t5_special_tokens_map.json"

echo "Tokenizers staged under $DEST. Verify parity with TokenizerParityTests (F004)."
