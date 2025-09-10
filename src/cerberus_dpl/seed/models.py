"""
Seed model loader & manifest helpers.

Artifacts layout (SEED_MODEL_PATH):
  - manifest.json
  - centroid.npy
  - centroids/0.npy,1.npy,...
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import json
import numpy as np


@dataclass(frozen=True)
class SeedModel:
    path: Path
    embedding_model: str
    dims: int
    doc_count: int
    centroids: List[np.ndarray]
    manifest: dict

    @property
    def num_centroids(self) -> int:
        return len(self.centroids)

    @classmethod
    def load(cls, path: str | Path) -> "SeedModel":
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

        centroids: List[np.ndarray] = []
        centroids_dir = path / "centroids"
        if centroids_dir.exists():
            for fp in sorted(centroids_dir.glob("*.npy"), key=lambda x: x.stem):
                v = np.load(fp).astype(np.float32)
                centroids.append(v)
        elif (path / "centroid.npy").exists():  # legacy single-centroid
            centroids.append(np.load(path / "centroid.npy").astype(np.float32))
        else:
            raise FileNotFoundError(
                "No seed centroids found (centroids/*.npy or centroid.npy)."
            )

        dims = int(centroids[0].shape[0])
        return cls(
            path=path,
            embedding_model=manifest["embedding_model"],
            dims=dims,
            doc_count=int(manifest.get("doc_count", 0)),
            centroids=centroids,
            manifest=manifest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "embedding_model": self.embedding_model,
            "dims": self.dims,
            "doc_count": self.doc_count,
            "num_centroids": self.num_centroids,
        }
