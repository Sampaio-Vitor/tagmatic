"""
Tagmatic - Generic Text Classification Library

A flexible, user-defined text classification library using Large Language Models.
"""

from .core import (
    Category,
    CategorySet,
    Classifier,
    DefaultLLMProvider,
    StructuredLLMProvider,
)
from .providers import (
    BaseLLMProvider,
    ClassificationResult,
    VotingResult,
    LLMProviderError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthenticationError,
)
from .utils import (
    TagmaticConfig,
    ValidationError,
    get_config,
    set_config,
    load_config_from_file,
)

__version__ = "0.1.0"
__author__ = "Tagmatic Contributors"
__email__ = "tagmatic@example.com"

__all__ = [
    # Core classes
    "Category",
    "CategorySet",
    "Classifier",
    "DefaultLLMProvider",
    "StructuredLLMProvider",
    # Provider classes
    "BaseLLMProvider",
    "ClassificationResult",
    "VotingResult",
    # Exceptions
    "LLMProviderError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "ValidationError",
    # Configuration
    "TagmaticConfig",
    "get_config",
    "set_config",
    "load_config_from_file",
]
