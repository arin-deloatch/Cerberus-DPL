"""General utility functions for Cerberus"""

from __future__ import annotations

from typing import Sequence, List, Iterable, Tuple, Optional
from datetime import datetime, timezone
from pathlib import Path
from lxml import html as lhtml
from urllib.parse import urljoin

from cerberus_dpl.models import TokenCounter
from cerberus_dpl.logging import logger
from cerberus_dpl.config import settings


##################
# Token Counting #
##################
def wordpiece_counter() -> TokenCounter:
    """Return a regex-based counter (counts word-like tokens)."""
    import re

    logger.info("Using regex-based token counter...")
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


############
# Datetime #
############


def _utc_now() -> datetime:
    """Returns the current time in the UTC timezone."""
    return datetime.now(timezone.utc)


###########
# Helpers #
###########


SUPPORTED_TEXT_EXT = {".txt", ".md", ".markdown"}
HTML_EXT = {".html", ".htm"}


def _extract_text_content(path: Path) -> Optional[str]:
    """Extract text content from a text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_html_content(path: Path) -> Optional[str]:
    """Extract text content from an HTML file."""
    try:
        tree = lhtml.fromstring(path.read_bytes())
        for element in tree.xpath("//script|//style"):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
        text = (tree.text_content() or "").strip()
        return text if text else None
    except Exception:
        return None


def _process_file(path: Path) -> Optional[Tuple[str, str]]:
    """Process a single file and extract text content."""
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT_EXT:
        text = _extract_text_content(path)
        return (path.stem, text) if text else None
    elif suffix in HTML_EXT:
        text = _extract_html_content(path)
        return (path.stem, text) if text else None

    return None


def _iter_seed_texts(input_path: Path) -> Iterable[Tuple[str, str]]:
    """Iterate through seed texts from input path."""
    if not input_path.is_dir():
        raise ValueError(f"""Unsupported seed input: {input_path};
                         Provide a directory of .txt, .md or .html files.""")

    for path in sorted(input_path.rglob("*")):
        if not path.is_file():
            continue

        result = _process_file(path)
        if result:
            yield result


def _strip_noise(tree: lhtml.HtmlElement) -> None:
    # Remove script and style elements
    for element in tree.xpath("//script | //style | //header | //nav | //footer"):
        element.getparent().remove(element)


def _extract_outlinks(tree: lhtml.HtmlElement, base_url: str) -> list[str]:
    hrefs = tree.xpath("//a[@href]/@href")
    out: list[str] = []
    seen: set[str] = set()
    for h in hrefs:
        if not h or h.startswith(("#", "mailto:", "javascript:")):
            continue
        u = urljoin(base_url, h)
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out
