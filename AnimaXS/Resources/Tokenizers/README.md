# Tokenizer assets

Local tokenizer data used by the app — no Hugging Face network access at runtime.
Files are flat (unique names) to avoid Xcode resource-copy filename collisions.

- `qwen_tokenizer.json` — Qwen2Tokenizer-equivalent BPE tokenizer, serialized from the
  pinned reference `comfy/text_encoders/qwen25_tokenizer/` (vocab.json + merges.txt) with the
  EXACT Qwen2Tokenizer pre-tokenization regex baked in. Verified byte-exact against the
  golden reference IDs for all canonical prompts (see DECISIONS D014).
- `t5_tokenizer.json` + `t5_tokenizer_config.json` + `t5_special_tokens_map.json` —
  T5TokenizerFast (Unigram/SentencePiece) from `comfy/text_encoders/t5_tokenizer/`.

Regenerate with `scripts/fetch_tokenizers.sh` (from a machine with the reference assets).
