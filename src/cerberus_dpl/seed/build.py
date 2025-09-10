"""
Build a seed model with 1 or N centroids.

Inputs:
  - Folder of *.txt/*.md/*.html

Outputs in SEED_MODEL_PATH:
  - manifest.json
  - centroid.npy - [n_centroids == 1]
  - centroids/0.npy - [n_centroids > 1]
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional
import json
import numpy as np
from lxml import html as lhtml
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from cerberus_dpl.logging import logger
from cerberus_dpl.config import settings

SUPPORTED_TEXT_EXT = {".txt", ".md", ".markdown"}
HTML_EXT = {".html", ".htm"}


def _iter_seed_texts(input_path: Path) -> Iterable[Tuple[str, str]]:
    """ """
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_TEXT_EXT:
                yield path.stem, path.read_text(encoding="utf-8", errors="ignore")
            elif suffix in HTML_EXT:
                try:
                    tree = lhtml.fromstring(path.read_bytes())
                    for element in tree.xpath("//script|//style"):
                        parent = element.getparent()
                        if parent is not None:
                            parent.remove(element)
                    text = (tree.text_content() or "").strip()
                    if text:
                        yield path.stem, text
                except Exception:
                    continue
        return

    raise ValueError(f"""Unsupported seed input: {input_path};
                     Provide a directory of .txt, .md or .html files.""")


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return (x / norms).astype(np.float32)


def build_seed_model(
    seed_input: str | Path,
    output_dir: Optional[str | Path] = None,
    embedding_model_name: Optional[str] = None,
    n_centroids: int = 1,
    random_state: int = 42,
) -> Path:
    input_path = Path(seed_input)
    output_path = Path(output_dir or settings.SEED_MODEL_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = embedding_model_name or settings.EMBEDDING_MODEL
    
    log = logger.bind(
        stage="build_seed",
        input=str(input_path),
        output=str(output_path),
        model=model_name,
        n_centroids=n_centroids,
    )
    log.info("seed_build_start")

    # Load texts
    ids, texts = [], []
    for doc_id, text in _iter_seed_texts(input_path):
        if len(text) >= 20:
            ids.append(doc_id)
            texts.append(text)
    if not texts:
        raise RuntimeError("No seed texts found ≥ 20 chars. Aborting.")
    if n_centroids > len(texts):
        log.warning(
            "seed_n_centroids_clamped", requested=n_centroids, available=len(texts)
        )
        n_centroids = max(1, len(texts))

    # Embed + normalize
    st_model = SentenceTransformer(model_name)
    emb = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = _normalize_rows(emb)

    centroids: list[np.ndarray] = []
    cluster_sizes: list[int] = []

    if n_centroids == 1:
        c = emb.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroids = [c.astype(np.float32)]
        # save legacy single file
        np.save(output_path / "centroid.npy", centroids[0])
    else:
        # KMeans on unit vectors; cluster centers then re-normalized
        km = KMeans(n_clusters=n_centroids, n_init="auto", random_state=random_state)
        labels = km.fit_predict(emb)
        centers = km.cluster_centers_
        # normalize centers to unit length
        centers = _normalize_rows(centers)
        # persist each center to centroids/<i>.npy
        cdir = output_path / "centroids"
        cdir.mkdir(parents=True, exist_ok=True)
        centroids = []
        for i, c in enumerate(centers):
            c = c / (np.linalg.norm(c) + 1e-9)
            c = c.astype(np.float32)
            np.save(cdir / f"{i}.npy", c)
            centroids.append(c)
        # sizes
        for i in range(n_centroids):
            cluster_sizes.append(int((labels == i).sum()))

    # Manifest
    manifest = {
        "version": "mvp-0.2",
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "embedding_model": model_name,
        "dims": int(centroids[0].shape[0]),
        "doc_count": len(texts),
        "source": str(input_path),
        "num_centroids": len(centroids),
        "cluster_sizes": cluster_sizes if cluster_sizes else None,
        "stats": {
            "mean_len": float(np.mean([len(t) for t in texts])),
            "min_len": int(min(len(t) for t in texts)),
            "max_len": int(max(len(t) for t in texts)),
        },
        "random_state": random_state,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log.info(
        "seed_build_done",
        docs=len(texts),
        dims=int(centroids[0].shape[0]),
        num_centroids=len(centroids),
        output=str(output_path),
    )
    return output_path

if __name__ == '__main__':
    build_seed_model(seed_input="../../../data/seed_corpus/",
                     )