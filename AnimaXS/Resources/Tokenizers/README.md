# Tokenizer assets

This directory is populated by `scripts/fetch_tokenizers.sh` with the Qwen and T5 tokenizer
folders that match the pinned reference (see TODO F002 and DECISIONS.md).

Each folder is loaded by `swift-transformers` `Tokenizers` from local files — no HF network
access at runtime.

Required subdirectories (created by the fetch script):
- `qwen/`  — Qwen2Tokenizer files (tokenizer.json + config)
- `t5/`    — T5TokenizerFast files (spiece.model + tokenizer_config.json)
