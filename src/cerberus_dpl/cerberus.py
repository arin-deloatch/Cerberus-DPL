import typer
from cerberus_dpl.logging import logger
from cerberus_dpl.seed.build import build_seed_model
from cerberus_dpl.config import settings

app = typer.Typer(help="Cerberus")

@app.command()
def build_seed(
    seed_input: str = typer.Argument(..., help="Path to seed folder (txt/md/html)"),
    output: str = typer.Option(None, "--output", "-o", help="Output folder for seed model (defaults to settings.SEED_MODEL_PATH)"),
    model: str = typer.Option(None, "--model", "-m", help="Embedding model name (defaults to settings.EMBEDDING_MODEL)"),
    n_centroids: int = typer.Option(1, "--n-centroids", "-k", min=1, help="Number of centroids (topics) to build"),
    random_state: int = typer.Option(42, "--random-state", help="Clustering RNG seed"),
):
    out = build_seed_model(seed_input, output_dir=output, embedding_model_name=model, n_centroids=n_centroids, random_state=random_state)
    logger.info("cli_build_seed_done", output=str(out), n_centroids=n_centroids)

def main():
    """
    This function serves as the main entrypoint for Cerberus

    Returns:
        None
    """

    app()

if __name__ == "__main__":
    main()