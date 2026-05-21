import yaml
from pathlib import Path
from typing import Optional
from .utils import console


class Config:
    """Handles configuration loading and default settings."""

    DEFAULT_MODELS = {
        'allenai/specter': 'allenai/specter',
        'allenai/scibert_scivocab_uncased': 'allenai/scibert_scivocab_uncased',
        'sentence-transformers/all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
    }

    DEFAULT_FIELDS = [
        "artificial intelligence",
        "machine learning",
        "computer vision",
    ]

    def __init__(self,
                 config_path: Optional[str] = None,
                 cache_dir: str = 'embedding_cache',
                 papers_per_field: int = 25,
                 max_tokens: int = 512,
                 min_tokens: int = 50,
                 device: Optional[str] = None):
        """Initialize configuration."""
        self.cache_dir = Path(cache_dir)
        self.experiment_dir = None
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.device = device
        self.papers_per_field = papers_per_field

        self._load_config(config_path)
        self._validate()

    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration from file or use defaults."""
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
                self.models = config.get('models') or self.DEFAULT_MODELS
                self.fields = config.get('fields') or self.DEFAULT_FIELDS
                self.papers_per_field = config.get('evaluation', {}).get(
                    'papers_per_field',
                    self.papers_per_field
                )
                console.print(f"📚 Loaded configuration from [cyan]{config_path}[/cyan]")
        else:
            self.models = self.DEFAULT_MODELS
            self.fields = self.DEFAULT_FIELDS
            console.print("ℹ️ Using default configuration")

        console.print(f"🤖 Models to evaluate: [green]{len(self.models)}[/green]")
        console.print(f"🔬 Research fields: [green]{len(self.fields)}[/green]")
        console.print(f"📄 Papers per field: [green]{self.papers_per_field}[/green]")

    def _validate(self) -> None:
        """Validate configuration values before long-running work starts."""
        if not self.models:
            raise ValueError("At least one model must be configured.")
        if not self.fields:
            raise ValueError("At least one research field must be configured.")
        if self.papers_per_field <= 0:
            raise ValueError("papers_per_field must be greater than 0.")
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be greater than 0.")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be greater than or equal to min_tokens.")

    def get_cache_path(self, text: str, model_name: str) -> Path:
        """Generate cache file path for a specific text and model."""
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_name_safe = model_name.replace('/', '_')
        return self.cache_dir / f"{model_name_safe}_{text_hash}.pkl"
