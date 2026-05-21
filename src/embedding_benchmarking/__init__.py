"""Academic Embedding Model Evaluator Package.

This package provides tools for evaluating embedding models on academic paper similarity tasks.

Heavy model dependencies such as PyTorch are intentionally not imported at package import time.
Import `ModelManager`, `DataManager`, or `Evaluator` from their modules when needed.
"""

from .config import Config

__version__ = "0.1.0"
__all__ = ["Config"]
