"""
Adapters for inp
"""

from .base import Adapter, AdapterError, NotFoundError, TransientError
from .web import WebAdapter

__all__ = [
    "Adapter",
    "AdapterError",
    "NotFoundError",
    "TransientError",
    "WebAdapter",
]
