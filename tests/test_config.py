import pytest

from src.embedding_benchmarking.config import Config


def test_config_accepts_papers_per_field():
    config = Config(config_path=None, papers_per_field=3, min_tokens=10, max_tokens=20)

    assert config.papers_per_field == 3
    assert config.min_tokens == 10
    assert config.max_tokens == 20


def test_config_rejects_invalid_papers_per_field():
    with pytest.raises(ValueError):
        Config(config_path=None, papers_per_field=0)


def test_config_rejects_invalid_token_range():
    with pytest.raises(ValueError):
        Config(config_path=None, min_tokens=100, max_tokens=10)
