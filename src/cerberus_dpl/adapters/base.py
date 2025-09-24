"""
Minimal adapter base (MVP).

An Adapter converts a source-specific handle (e.g., URL) into a NormalizedDoc.
Keep this tiny for now: one method, a few exceptions.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from ..models import NormalizedDoc


class AdapterError(Exception):
    """Generic adapter failure (parsing, unsupported content, etc.)."""


class NotFoundError(AdapterError):
    """(e.g., 404/410)."""


class TransientError(AdapterError):
    """Retryable issues (timeouts, 5xx)."""


@runtime_checkable
class Adapter(Protocol):
    name: str

    def fetch_and_normalize(self, handle: str) -> NormalizedDoc:
        """
        Fetch from the source and return a NormalizedDoc.

        Raises:
            NotFoundError   -> when the resource doesn't exist
            TransientError  -> for temporary/network/server issues
            AdapterError    -> for permanent, non-retryable issues
        """
        ...
