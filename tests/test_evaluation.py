import pytest

from src.embedding_benchmarking.evaluation import Evaluator


class DummyConfig:
    papers_per_field = 2
    max_tokens = 512
    min_tokens = 50
    fields = ["field-a", "field-b"]
    models = {"dummy": "dummy"}
    device = None


class DummyModelManager:
    pass


def make_evaluator():
    return Evaluator(DummyConfig(), DummyModelManager())


def test_validate_papers_requires_two_categories():
    papers = [
        {"title": "A", "abstract": "alpha", "category": "one"},
        {"title": "B", "abstract": "beta", "category": "one"},
    ]

    with pytest.raises(ValueError):
        make_evaluator()._validate_papers(papers)


def test_validate_papers_requires_same_field_pair():
    papers = [
        {"title": "A", "abstract": "alpha", "category": "one"},
        {"title": "B", "abstract": "beta", "category": "two"},
    ]

    with pytest.raises(ValueError):
        make_evaluator()._validate_papers(papers)


def test_summarize_scores_rejects_empty_metric_bucket():
    results = {
        "title_abstract_same": [0.9],
        "title_abstract_diff": [],
    }

    with pytest.raises(ValueError):
        make_evaluator()._summarize_scores(results)
