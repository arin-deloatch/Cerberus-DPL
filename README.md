<p align="center">
  <img src="assets/cerberus-ai.png" alt="AI Generated Cerberus" width="400"/>
</p>

<h1 align="center">cerberus</h1>
<p align="center"><i>Your first line of defense against low-quality documents in RAG systems.</i></p>

---

## Overview

cerberus is a lightweight pipeline for filtering, normalizing, and ingesting documents into RAG (Retrieval-Augmented Generation) systems. It focuses on the "gatekeeper" step of the pipeline: ensuring only high-quality, relevant, and deduplicated documents get chunked, embedded, and stored in your vector database.

By using topic modeling (BERTopic) and clustering techniques, Cerberus helps you:
- **Filter** low-quality or irrelevant documents before they pollute your vector store
- **Normalize** content from various sources (web, text files, etc.)
- **Cluster** documents by semantic similarity to identify coherent topics
- **Infer** topic assignments for new documents against a trained seed model

## Features

- **Seed Model Training**: Build a topic model from a corpus of high-quality seed documents
- **Document Classification**: Classify new documents against your trained model
- **Web Adapter**: Fetch and normalize content from URLs
- **Flexible Configuration**: YAML-based policy configuration for quality gating thresholds
- **Statistics & Metrics**: Track cluster-level statistics and document quality metrics
- **CLI Interface**: Simple command-line tools for all operations

## Installation

### Prerequisites
- Python >= 3.12, < 3.13

### Install from source

```
# Clone the repository
git clone https://github.com/arin-deloatch/Cerberus-DPL.git

# Install with uv
uv sync
```

## Quick Start

### 1. Build a Seed Model

Train a topic model on a corpus of seed documents:

```
uv run cerberus build-seed /path/to/seed/documents \
  --output artifacts/ \
  --model ibm-granite/granite-embedding-30m-english \
  --min-cluster-size 5
```

**Options:**
- `input_path`: Directory containing seed documents (txt, md, html)
- `--output, -o`: Output directory for the trained model
- `--model, -m`: Embedding model name (HuggingFace compatible)
- `--min-cluster-size, -k`: Minimum cluster size for HDBSCAN (default: 5)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--overwrite, -f`: Overwrite existing model directory

### 2. Classify Documents

Infer topic assignments for new documents:

#### From URL:
```
uv run cerberus infer \
  --model-path artifacts/bertopic_model_2025... \
  --url https://example.com/article \
  --show-terms 10 \
  --json-out results.json
```

#### From text:
```
uv run cerberus infer \
  --model-path artifacts/bertopic_model_2025... \
  --text "Your document text here..." \
  --show-terms 10
```

**Options:**
- `--model-path, -m`: Path to trained BERTopic model
- `--url, -u`: URL to fetch and classify
- `--text, -t`: Direct text input to classify
- `--min-words`: Minimum word count warning threshold (default: 40)
- `--show-terms`: Show top-N c-TF-IDF terms for predicted topic (default: 10)
- `--use-hf-tokenizer/--no-hf-tokenizer`: Use HuggingFace tokenizer for token counting
- `--json-out`: Save full JSON results to file

**Example output:**
```json
{
  "source": "https://example.com/article",
  "topic": 5,
  "probability": 0.87,
  "rejected": false,
  "token_count": 1247,
  "word_count": 982,
  "text_length": 6543,
  "top_terms": [
    {"word": "machine", "score": 0.45},
    {"word": "learning", "score": 0.42},
    ...
  ]
}
```

## Configuration

Cerberus uses environment variables and YAML configuration files:

- **Environment Settings**: Configure via `config.py` or environment variables
- **Policy Configuration**: Define quality gating thresholds in `policy.yaml`

## Architecture

```
cerberus_dpl/
├── adapters/          # Input adapters (web, file, etc.)
├── seed/              # BERTopic model training
├── metrics/           # Statistics and inference
├── utils/             # Utility functions
├── models.py          # Pydantic data models
├── config.py          # Settings management
├── logging.py         # Structured logging
└── cli.py             # Command-line interface
```

### Key Components

- **BERTopic**: Topic modeling with HDBSCAN clustering and UMAP dimensionality reduction
- **Sentence Transformers**: Document embeddings
- **Pydantic**: Data validation and serialization
- **Typer**: CLI framework

## Development

### Dependencies

Core dependencies:
- `bertopic` - Topic modeling
- `docling` - Document processing
- `hdbscan` - Density-based clustering
- `sentence-transformers` - Embeddings
- `scikit-learn` - Machine learning utilities
- `pydantic` - Data validation
- `typer` - CLI framework

Dev dependencies:
- `pyright` - Static type checking
- `ruff` - Linting and formatting

## Use Cases

1. **RAG Pipeline Gating**: Filter documents before ingestion to maintain vector store quality
2. **Content Clustering**: Organize large document collections by semantic topics
3. **Quality Control**: Identify and reject low-quality or off-topic documents
4. **Topic Analysis**: Understand the thematic structure of your document corpus


## License

Apache-2.0

---

*This repository is under active development.*
