import typer
from pathlib import Path
from typing import Optional
import json

from cerberus_dpl.logging import logger
from cerberus_dpl.seed.bertopic_model import train_bertopic_model, load_bertopic_model
from cerberus_dpl.config import settings
from cerberus_dpl.adapters.web import WebAdapter
from cerberus_dpl.metrics.inference import predict_single_document

app = typer.Typer(help="Cerberus")


@app.command()
def build_seed(
    input_path: Path = typer.Argument(..., help="Path to seed folder (txt/md/html)"),
    output: Path = typer.Option(
        settings.SEED_MODEL_PATH,
        "--output",
        "-o",
        help="Output folder for seed model (defaults to settings.SEED_MODEL_PATH)",
    ),
    model: str = typer.Option(
        settings.EMBEDDING_MODEL,
        "--model",
        "-m",
        help="Embedding model name (defaults to settings.EMBEDDING_MODEL)",
    ),
    min_cluster_size: int = typer.Option(
        5,
        "--min-cluster-size",
        "-k",
        min=1,
        help="Minimum cluster size for HDBSCAN to build.",
    ),
    random_state: int = typer.Option(42, "--random-state", help="Clustering RNG seed"),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "-f",
        help="Overwrite contents in SEED_MODEL_PATH directory if it already exists",
    ),
):
    out = train_bertopic_model(
        input_path=input_path,
        output_path=output,
        embedding_model_name=model,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
        overwrite=overwrite,
    )
    logger.info("cli_build_seed_done", output=str(out))


@app.command()
def infer(
    model_path: Path = typer.Option(
        ..., "--model-path", "-m", help="Path to saved BERTopic model directory/file."
    ),
    url: Optional[str] = typer.Option(
        None, "--url", "-u", help="Webpage URL to infer."
    ),
    text: Optional[str] = typer.Option(
        None, "--text", "-t", help="Text content to classify."
    ),
    min_words: int = typer.Option(
        40, "--min-words", help="Warn if extracted text has fewer words."
    ),
    show_terms: int = typer.Option(
        10, "--show-terms", help="Include top-N c-TF-IDF terms for the predicted topic."
    ),
    use_hf_tokenizer: bool = typer.Option(
        True,
        "--use-hf-tokenizer/--no-hf-tokenizer",
        help="Use HuggingFace tokenizer for token counting.",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json-out", help="Write full JSON payload to this file."
    ),
):
    """Infer topic for a document from URL or text using a trained BERTopic model."""

    # Validate input
    if not url and not text:
        typer.echo("Error: Either --url or --text must be provided.", err=True)
        raise typer.Exit(1)

    if url and text:
        typer.echo("Error: Provide either --url or --text, not both.", err=True)
        raise typer.Exit(1)

    if url:
        logger.info(f"Fetching content from URL: {url}")
        web_adapter = WebAdapter()
        try:
            normalized_doc = web_adapter.fetch_and_normalize(url)
            document_text: str = normalized_doc.text
            source: str = url
        except Exception as e:
            typer.echo(f"Error fetching URL: {e}", err=True)
            raise typer.Exit(1)
    else:
        document_text = text if text else ""
        source = "user-provided text"

    # Check minimum word count
    word_count = len(document_text.split())
    if word_count < min_words:
        logger.warning(
            f"Document has only {word_count} words (minimum recommended: {min_words})"
        )

    # Predict topic
    logger.info(f"Inferring topic for document ({word_count} words)")
    try:
        result = predict_single_document(
            text=document_text,
            model_path=model_path,
            use_hf_tokenizer=use_hf_tokenizer,
        )
    except Exception as e:
        typer.echo(f"Error during inference: {e}", err=True)
        raise typer.Exit(1)

    # Get topic terms if requested
    topic_terms = None
    if show_terms > 0:
        try:
            topic_model = load_bertopic_model(model_path)
            topic_id = int(result["topic"])
            if topic_id != -1:
                topic_words = topic_model.get_topic(topic_id)
                if topic_words and isinstance(topic_words, list):
                    topic_terms = topic_words[:show_terms]
        except Exception as e:
            logger.warning(f"Could not retrieve topic terms: {e}")

    # Build output
    output = {
        "source": str(source),
        "topic": int(result["topic"]),
        "probability": float(result["probability"]),
        "is_outlier": bool(result["is_outlier"]),
        "token_count": int(result["token_count"]),
        "word_count": int(word_count),
        "text_length": int(result["text_length"]),
    }

    if topic_terms:
        output["top_terms"] = [
            {"word": word, "score": float(score)} for word, score in topic_terms
        ]

    # Output results
    typer.echo(json.dumps(output, indent=2))

    # Save to file if requested
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to: {json_out}")


def main():
    """
    This function serves as the main entrypoint for Cerberus

    Returns:
        None
    """

    app()


if __name__ == "__main__":
    main()
