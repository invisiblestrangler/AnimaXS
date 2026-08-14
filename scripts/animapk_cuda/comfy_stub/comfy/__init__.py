"""Minimal plumbing stubs for running the REAL pinned comfy predict2.py
(MiniTrainDIT) standalone on CUDA.  Only the pieces that are pure machinery
are stubbed: patcher-extension wrapper, cli args, SDPA attention adapter,
plain rms_rope fallback, manual_cast op classes.  predict2.py and
position_embedding.py are the pinned upstream files verbatim.
"""
