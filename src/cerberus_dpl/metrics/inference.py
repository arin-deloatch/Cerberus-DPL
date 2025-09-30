"""Topic inference and classification utilities for new documents"""

from pathlib import Path
from typing import List, Dict, Union
import json

from cerberus_dpl.logging import logger
from cerberus_dpl.models import DocRecord, TopicStats
from cerberus_dpl.seed.bertopic_model import load_bertopic_model
from cerberus_dpl.utils.utils import hf_token_counter, wordpiece_counter
from cerberus_dpl.config import settings
from cerberus_dpl.utils.utils import _utc_now


def infer_topics(
    documents: List[str],
    model_path: Path,
    use_hf_tokenizer: bool = True,
) -> List[DocRecord]:
    """
    Infer topic assignments for new documents using a trained BERTopic model.

    Parameters
    ----------
    documents : List[str]
        List of document texts to classify
    model_path : Path
        Path to the saved BERTopic model
    use_hf_tokenizer : bool, default=True
        Whether to use HuggingFace tokenizer for token counting

    Returns
    -------
    List[DocRecord]
        List of document records with topic assignments and token counts
    """
    topic_model = load_bertopic_model(model_path)

    logger.info(f"Inferring topics for {len(documents)} documents")
    topics, _ = topic_model.transform(documents)

    # Count tokens for each document
    if use_hf_tokenizer:
        try:
            counter = hf_token_counter(settings.EMBEDDING_MODEL)
            token_counts = counter(documents)
            logger.info("Using HuggingFace tokenizer for token counting")
        except Exception as e:
            logger.warning(
                f"Failed to use HF tokenizer: {e}, falling back to wordpiece counter"
            )
            counter = wordpiece_counter()
            token_counts = counter(documents)
    else:
        counter = wordpiece_counter()
        token_counts = counter(documents)
        logger.info("Using wordpiece counter for token counting")

    # Create DocRecord objects
    doc_records = []
    for idx, (topic, token_count) in enumerate(zip(topics, token_counts)):
        doc_records.append(
            DocRecord(doc_index=idx, topic=int(topic), token_len=token_count)
        )

    return doc_records


def save_inference_results(
    doc_records: List[DocRecord],
    topic_stats: List[TopicStats],
    output_path: Path,
) -> None:
    """
    Save inference results to JSON files.

    Parameters
    ----------
    doc_records : List[DocRecord]
        Document records with topic assignments
    topic_stats : List[TopicStats]
        Topic statistics
    output_path : Path
        Directory to save results
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save document records
    doc_records_path = output_path / f"doc_records_{_utc_now()}.json"
    with open(doc_records_path, "w") as f:
        json.dump([record.model_dump() for record in doc_records], f, indent=2)
    logger.info(f"Document records saved to: {doc_records_path}")

    # Save topic statistics
    topic_stats_path = output_path / f"topic_stats_{_utc_now()}.json"
    with open(topic_stats_path, "w") as f:
        json.dump([stat.model_dump() for stat in topic_stats], f, indent=2)
    logger.info(f"Topic statistics saved to: {topic_stats_path}")


def get_topic_info(model_path: Path) -> Dict:
    """
    Get topic information from a trained model.

    Parameters
    ----------
    model_path : Path
        Path to the saved BERTopic model

    Returns
    -------
    Dict
        Topic information including keywords and descriptions
    """
    topic_model = load_bertopic_model(model_path)
    topic_info = topic_model.get_topic_info()
    return topic_info.to_dict()


def predict_single_document(
    text: str,
    model_path: Path,
    use_hf_tokenizer: bool = True,
) -> Dict[str, Union[int, float, str]]:
    """
    Predict topic for a single document and return detailed information.

    Parameters
    ----------
    text : str
        Document text to classify
    model_path : Path
        Path to the saved BERTopic model
    use_hf_tokenizer : bool, default=True
        Whether to use HuggingFace tokenizer for token counting

    Returns
    -------
    Dict[str, Union[int, float, str]]
        Dictionary with topic assignment, probability, and metadata
    """
    topic_model = load_bertopic_model(model_path)

    # Get topic assignment and probability
    topics, probabilities = topic_model.transform([text])
    topic = topics[0]
    probability = probabilities[0] if len(probabilities) > 0 else 0.0

    # Count tokens
    if use_hf_tokenizer:
        try:
            counter = hf_token_counter(settings.EMBEDDING_MODEL)
            token_count = counter([text])[0]
        except Exception:
            counter = wordpiece_counter()
            token_count = counter([text])[0]
    else:
        counter = wordpiece_counter()
        token_count = counter([text])[0]

    return {
        "topic": int(topic),
        "probability": float(probability),
        "token_count": token_count,
        "text_length": len(text),
        "is_outlier": topic == -1,
    }


if __name__ == "__main__":
    predict_single_document(
        text="""The newest major release of Red Hat Enterprise Linux introduces the next era of the 
                            operating system (OS), with AI readiness and advanced 
                            capabilities to help you address the Linux skills gap, 
                            contain drift with container technologies, 
                            get started with post-quantum security, and more.""",
        model_path=Path(
            "../../../artifacts/bertopic_model_<function _utc_now at 0x7f092b682c00>"
        ),
    )
