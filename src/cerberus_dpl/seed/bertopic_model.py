"""BERTopic Model Creation"""

import json
import shutil
from typing import Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from cerberus_dpl.logging import logger
from cerberus_dpl.config import settings
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

from cerberus_dpl.utils.utils import _utc_now, _iter_seed_texts


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
            raise FileExistsError(
                f"Output path already exists: {output_path}. Use overwrite=True to replace it."
            )
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    # Create the output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the model (BERTopic saves as a file, not directory)
    model_file_path = output_path / f"bertopic_model_{_utc_now()}"
    topic_model.save(
        str(model_file_path), serialization="safetensors", save_ctfidf=True
    )
    logger.info(f"Model saved to: {model_file_path}")

    # Save topic info
    topic_info = topic_model.get_topic_info()
    info_path = output_path / f"topic_info_{_utc_now()}.json"
    with open(info_path, "w") as f:
        json.dump(topic_info.to_dict(), f, indent=2, default=str)
    logger.info(f"Topic info saved to: {info_path}")


def train_bertopic_model(
    input_path: Path,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    **model_kwargs,
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


def load_bertopic_model(model_path: Path) -> BERTopic:
    """
    Load a BERTopic model from a directory or file saved with `topic_model.save(...)`.
    This will also restore the embedding model (often saved with safetensors).
    """
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        raise FileNotFoundError

    logger.info(f"Loaded topic model from: {model_path}")
    topic_model = BERTopic.load(
        str(model_path), embedding_model=settings.EMBEDDING_MODEL
    )
    return topic_model
