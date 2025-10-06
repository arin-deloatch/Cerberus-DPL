"""

Cerberus: Data Quality Gate for RAG ingestion.

This package provides:
- NormalizedDoc: a standard schema for all documents.
"""

from importlib.metadata import version

from .models import NormalizedDoc


__all__ = [
    "NormalizedDoc",
]

try:
    __version__ = version("cerberus-dpl")
except Exception:
    __version__ = "0.0.1"

__author__ = "Arin DeLoatch"
