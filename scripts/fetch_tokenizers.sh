#!/usr/bin/env bash
# Fetch Qwen + T5 tokenizer assets matching the pinned reference into the app resources.
# Idempotent; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="AnimaXS/Resources/Tokenizers"
mkdir -p "$DEST/qwen" "$DEST/t5"

# Qwen tokenizer for Qwen3-0.6B (Qwen2Tokenizer, vocab 151936).
# The reference ComfyUI repo pins the tokenizer files used to produce the goldens.
QWEN_SRC="${QWEN_TOKENIZER_SRC:-}"
if [ -n "$QWEN_SRC" ]; then
  cp "$QWEN_SRC"/tokenizer.json "$DEST/qwen/" 2>/dev/null || true
  cp "$QWEN_SRC"/config.json "$DEST/qwen/" 2>/dev/null || true
else
  echo "QWEN_TOKENIZER_SRC not set; place tokenizer.json + config.json into $DEST/qwen/ manually."
fi

# T5 tokenizer for the LLM adapter target (T5TokenizerFast, SentencePiece).
T5_SRC="${T5_TOKENIZER_SRC:-}"
if [ -n "$T5_SRC" ]; then
  cp "$T5_SRC"/spiece.model "$DEST/t5/" 2>/dev/null || true
  cp "$T5_SRC"/tokenizer_config.json "$DEST/t5/" 2>/dev/null || true
else
  echo "T5_TOKENIZER_SRC not set; place spiece.model + tokenizer_config.json into $DEST/t5/ manually."
fi

echo "Tokenizers staged under $DEST. Verify with the F004 parity tests before proceeding."
