"""Instruction text → language tokens for the VTLA policy (see ``language/README.md``).

The dependency-free :class:`HashingTextEncoder` is the default; HuggingFace text towers
(CLIP / SigLIP / T5) are imported lazily by :class:`HFTextEncoder`.
"""
from .text_encoders import (
    BOS_ID, N_SPECIAL, PAD_ID, HashingTextEncoder, HashingTokenizer, HFTextEncoder, HFTokenizerFn,
    InstructionCache, TextEncoder, build_text_encoder, hash_token, pool_tokens, simple_word_tokenize,
)

__all__ = [
    "TextEncoder", "HashingTokenizer", "HashingTextEncoder", "HFTokenizerFn", "HFTextEncoder",
    "InstructionCache", "build_text_encoder",
    "simple_word_tokenize", "hash_token", "pool_tokens", "PAD_ID", "BOS_ID", "N_SPECIAL",
]
