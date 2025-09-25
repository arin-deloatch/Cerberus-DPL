import typer
from pathlib import Path

from cerberus_dpl.logging import logger
from cerberus_dpl.seed.bertopic_model import train_bertopic_model
from cerberus_dpl.config import settings

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


def main():
    """
    This function serves as the main entrypoint for Cerberus

    Returns:
        None
    """

    app()


if __name__ == "__main__":
    main()
