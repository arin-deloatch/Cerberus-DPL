"""
Seed corpus preparation:

- Build a seed model from a directory of text/HTML/JSONL documents
- Produce one or more centroids (semantic anchors) via embeddings + clustering
- Persist artifacts (manifest.json, centroid(s).npy, optional topics.json)
- Load the seed model later for use in quality metrics (e.g., SeedAlign)

"""

from .models import SeedModel
from .build import build_seed_model

__all__ = ["build_seed_model", "SeedModel"]
