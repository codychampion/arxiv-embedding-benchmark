# Academic Embedding Model Evaluator

A benchmarking toolkit for comparing embedding models on academic paper similarity tasks using research paper titles, abstracts, and field labels.

The project asks a practical retrieval question: can an embedding model connect a paper title to its real abstract, keep related papers close, separate unrelated fields, and behave consistently across domains?

## Why this exists

Embedding models are often chosen by popularity or broad benchmark reputation. Research retrieval is more specific. A useful model for literature search, scientific RAG, or technical discovery needs to represent relationships between papers in a way that supports real downstream decisions.

This repo provides a repeatable evaluation harness for comparing model behavior across scientific fields.

## What it evaluates

| Dimension | What it measures |
|---|---|
| Title to own abstract | Whether a model connects a paper title with its real content |
| Title to same-field abstracts | Whether it distinguishes related but different papers |
| Title to other-field abstracts | Whether it separates unrelated research areas |
| Abstract to abstract similarity | Whether papers cluster meaningfully by topic |
| Score consistency | Whether behavior is stable across fields and comparisons |

## Benchmark snapshot

The current experiment compares local, scientific, biomedical, general-purpose, and cloud-hosted embedding models.

| Rank | Model | Score | Own title / abstract | Same-field separation | Avg std |
|---:|---|---:|---:|---:|---:|
| 1 | Bedrock | 0.449 | 0.710 | 0.103 | 0.118 |
| 2 | MPNet | 0.443 | 0.714 | 0.271 | 0.134 |
| 3 | MiniLM-L12 | 0.439 | 0.688 | 0.246 | 0.130 |
| 4 | MiniLM-L6 | 0.433 | 0.667 | 0.242 | 0.129 |
| 5 | RoBERTa-Large-ST | 0.410 | 0.601 | 0.165 | 0.110 |

The useful signal is not only the winner. Different model families trade off high title / abstract similarity against separation between related papers. For retrieval systems, over-clustering can be just as damaging as weak recall.

## Features

- Collects academic papers across configured research fields
- Filters abstracts by token length for more consistent comparisons
- Evaluates Hugging Face models and AWS Bedrock embeddings
- Supports CPU execution with optional GPU acceleration
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
python -m src.embedding_benchmarking.cli evaluate
```

## Configuration

Models and fields are configured in YAML. A typical run includes a mix of general-purpose, scientific, biomedical, and cloud-hosted models.

## Outputs

Each run creates a timestamped experiment directory under `experiments/`.

| File | Purpose |
|---|---|
| `embedding_comparison_results.csv` | Full per-model metric table |
| `model_leaderboard.csv` | Ranked aggregate leaderboard |
| `papers_metadata.csv` | Paper titles, abstracts, fields, and metadata |
| `collection_statistics.yaml` | Corpus statistics and token distribution |

## Project structure

```text
src/embedding_benchmarking/
├── cli.py
├── config.py
├── data.py
├── embedding_evaluator.py
├── evaluation.py
├── models.py
└── utils.py
```

## Notes on interpretation

This benchmark is best used as a decision-support tool, not a universal ranking. The right embedding model depends on the corpus, query style, task, and cost envelope.

## Citation

```bibtex
@software{embedding_benchmarking_arxiv,
  title = {Academic Embedding Model Evaluator},
  author = {Champion, Cody},
  year = {2024},
  description = {A tool for evaluating embedding models on academic paper similarity tasks}
}
```
