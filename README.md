# Academic Embedding Model Evaluator

A benchmarking toolkit for comparing embedding models on academic paper similarity tasks using arXiv metadata and abstracts.

The goal is simple: evaluate whether an embedding model can capture meaningful relationships between research papers, not just produce high-dimensional vectors. The project compares models across scientific fields, measures title / abstract alignment, evaluates same-field and cross-field separation, and saves reproducible experiment outputs.

## Why this exists

Embedding models are often selected by popularity, leaderboard reputation, or anecdotal fit. Research retrieval is different: a useful model needs to connect a paper title to its abstract, keep related papers close, separate unrelated fields, and behave consistently across domains.

This repo provides a practical evaluation harness for that kind of decision. It is especially useful when choosing models for literature search, research recommendation, scientific RAG, technical discovery workflows, or domain-specific knowledge systems.

## What it evaluates

| Dimension | What it measures |
|---|---|
| Title ↔ own abstract | Whether a model connects a paper title with its actual content |
| Title ↔ other abstracts in the same field | Whether it distinguishes related but different papers |
| Title ↔ abstracts from other fields | Whether it separates unrelated research areas |
| Abstract ↔ abstract similarity | Whether papers cluster meaningfully by topic |
| Score consistency | Whether the model behaves reliably across fields and comparisons |

## Current benchmark snapshot

The existing experiment results compare a broad set of local, scientific, biomedical, general-purpose, and cloud-hosted embedding models.

Top results from the current leaderboard:

| Rank | Model | Score | Own title / abstract | Same-field separation | Avg std |
|---:|---|---:|---:|---:|---:|
| 1 | Bedrock | 0.449 | 0.710 | 0.103 | 0.118 |
| 2 | MPNet | 0.443 | 0.714 | 0.271 | 0.134 |
| 3 | MiniLM-L12 | 0.439 | 0.688 | 0.246 | 0.130 |
| 4 | MiniLM-L6 | 0.433 | 0.667 | 0.242 | 0.129 |
| 5 | RoBERTa-Large-ST | 0.410 | 0.601 | 0.165 | 0.110 |

The interesting result is not just “which model won.” Different model families show different tradeoffs between high title / abstract similarity and useful separation between related papers. That matters for retrieval systems where over-clustering can be just as damaging as poor recall.

## Features

- Pulls academic papers from arXiv across configured research fields
- Filters abstracts by token length to keep comparisons consistent
- Evaluates Hugging Face embedding models and AWS Bedrock Titan embeddings
- Supports CPU execution with optional GPU acceleration through PyTorch
- Caches embeddings to avoid unnecessary recomputation
- Produces CSV leaderboards, detailed metrics, paper metadata, and experiment snapshots
- Uses Rich progress output for long-running benchmark visibility

## Quick start

```bash
git clone https://github.com/codychampion/embedding_benchmarking_arvix.git
cd embedding_benchmarking_arvix
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create an environment file:

```bash
cp config/.env-example .env
```

Add a Hugging Face token:

```bash
HUGGINGFACE_TOKEN=your_token_here
```

Then run an evaluation:

```bash
python -m src.embedding_benchmarking.cli evaluate
```

Optional flags:

```bash
python -m src.embedding_benchmarking.cli evaluate \
  --cache-dir embedding_cache \
  --max-tokens 512 \
  --min-tokens 50 \
  --config config/config.yaml
```

## Configuration

Models and research fields are configured in YAML. A typical configuration includes a mix of general-purpose, scientific, biomedical, and cloud-hosted models:

```yaml
models:
  Bedrock: Bedrock
  Specter: allenai/specter
  SciBERT: allenai/scibert_scivocab_uncased
  MPNet: sentence-transformers/all-mpnet-base-v2
  MiniLM-L6: sentence-transformers/all-MiniLM-L6-v2

fields:
  - artificial intelligence
  - machine learning
  - computer vision
  - molecular biology
  - neuroscience
  - quantum computing
  - materials science
```

## Outputs

Each run creates a timestamped experiment directory under `experiments/` with files such as:

| File | Purpose |
|---|---|
| `embedding_comparison_results.csv` | Full per-model metric table |
| `model_leaderboard.csv` | Ranked aggregate leaderboard |
| `papers_metadata.csv` | Paper titles, abstracts, fields, and metadata |
| `collection_statistics.yaml` | Corpus statistics and token distribution |
| config snapshot | Reproducibility record for the run |

## Project structure

```text
.
├── src/embedding_benchmarking/
│   ├── cli.py                  # Command line entry point
│   ├── config.py               # Configuration loading
│   ├── data.py                 # arXiv collection and preprocessing
│   ├── embedding_evaluator.py  # Core embedding workflow
│   ├── evaluation.py           # Similarity metrics and leaderboard logic
│   ├── models.py               # Model adapters
│   └── utils.py                # Console and logging utilities
├── config/
│   ├── config-example.yaml
│   └── .env-example
├── experiments/
│   └── experiment_*            # Saved benchmark runs
├── docs/
└── requirements.txt
```

## Notes on interpretation

This benchmark is best used as a decision-support tool, not a universal ranking. A strong retrieval model depends on the corpus, the query style, the task, and the cost envelope. The most useful comparison is often between models that behave differently: high similarity versus better separation, local execution versus managed API, or domain-specific model versus general-purpose model.

## Citation

```bibtex
@software{embedding_benchmarking_arxiv,
  title = {Academic Embedding Model Evaluator},
  author = {Champion, Cody},
  year = {2024},
  description = {A tool for evaluating embedding models on academic paper similarity tasks}
}
```
