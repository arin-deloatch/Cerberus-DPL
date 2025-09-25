"""BERTopic Model Creation"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Optional
import json
import shutil
from lxml import html as lhtml
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from cerberus_dpl.logging import logger
from cerberus_dpl.config import settings
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer


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


def create_bertopic_model(
    min_topic_size: int = 5,
    n_neighbors: int = 5,
    embedding_model_name: str = settings.EMBEDDING_MODEL,
    n_components: int = 5,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> BERTopic:
    """Create a BERTopic model using sentence transformer embedding model."""

    logger.info(
        f"Creating BERTopic model with embedding model: {settings.EMBEDDING_MODEL}"
    )

    # Initialize the embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    # UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    # HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Vectorizer for topic representation
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2)
    )

    # c-TF-IDF for topic representation
    ctfidf_model = ClassTfidfTransformer()

    # KeyBERT-inspired representation
    representation_model = KeyBERTInspired()

    # Create BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        verbose=True,
    )

    return topic_model


def save_bertopic_model(
    topic_model: BERTopic, output_path: Path, overwrite: bool = False
) -> None:
    """Save a trained BERTopic model and its topic information.

    Args:
        topic_model: The trained BERTopic model to save
        output_path: Path where to save the model
        overwrite: Whether to overwrite if the directory already exists
    """
    output_path = Path(output_path)

    # Handle existing directory
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}. Use overwrite=True to replace it.")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    # Create the output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the model (BERTopic saves as a file, not directory)
    model_file_path = output_path / f"bertopic_model_{datetime.now(timezone.utc)}"
    topic_model.save(str(model_file_path),serialization="safetensors", save_ctfidf=True)
    logger.info(f"Model saved to: {model_file_path}")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    info_path = output_path / f"topic_info_{datetime.now(timezone.utc)}.json"
    with open(info_path, "w") as f:
        json.dump(topic_info.to_dict(), f, indent=2, default=str)
    logger.info(f"Topic info saved to: {info_path}")


def train_bertopic_model(
    input_path: Path, output_path: Optional[Path] = None, overwrite: bool = False, **model_kwargs
) -> BERTopic:
    """Train a BERTopic model on seed texts."""

    logger.info(f"Training BERTopic model on texts from: {input_path}")

    # Collect documents
    documents = []
    document_names = []

    for name, text in _iter_seed_texts(input_path):
        if text.strip():
            documents.append(text)
            document_names.append(name)

    if not documents:
        raise ValueError(f"No documents found in {input_path}")

    logger.info(f"Found {len(documents)} documents for training")

    # Create and train model
    topic_model = create_bertopic_model(**model_kwargs)
    _, _ = topic_model.fit_transform(documents)

    logger.info(f"Training completed. Found {len(topic_model.get_topic_info())} topics")

    # Save model if output path provided
    if output_path:
        save_bertopic_model(topic_model, output_path, overwrite=overwrite)

    return topic_model
