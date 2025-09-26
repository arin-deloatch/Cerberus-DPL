"""Token counting utilities"""

import re
from typing import Callable, Sequence, List
from cerberus_dpl.config import settings

TokenCounter = Callable[[Sequence[str]], List[int]]


def wordpiece_counter() -> TokenCounter:
    """Return a regex-based counter (counts word-like tokens)."""
    regex_parser = re.compile(r"\w+")
    return lambda texts: [len(regex_parser.findall(t or "")) for t in texts]


def hf_token_counter(model_name: str = settings.EMBEDDING_MODEL) -> TokenCounter:
    """
    Return a Hugging Face tokenizer-based counter for a given model.

    Parameters
    ----------
    model_name : str
        Hugging Face model repo ID (e.g. Granite embedding model).

    Notes
    -----
    Imports `transformers` locally to avoid hard dependency unless used.
    """
    from transformers import AutoTokenizer

    token = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _count(texts: Sequence[str]) -> List[int]:
        enc = token(
            list(texts), add_special_tokens=False, padding=False, truncation=False
        )
        return [len(x) for x in enc["input_ids"]]

    return _count