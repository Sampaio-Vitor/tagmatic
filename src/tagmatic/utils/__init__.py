"""Utility functions and helpers for Tagmatic."""

from .config import (
    TagmaticConfig,
    get_config,
    set_config,
    load_config_from_file,
    update_config,
    reset_config,
    auto_load_config,
)
from .validation import (
    ValidationError,
    validate_text_input,
    validate_category_name,
    validate_voting_rounds,
    validate_confidence_score,
    validate_temperature,
    validate_max_tokens,
    validate_batch_texts,
    validate_category_set,
    validate_llm_response,
    validate_api_key,
    validate_timeout,
    sanitize_text_for_prompt,
    validate_classification_result,
)

__all__ = [
    # Config
    "TagmaticConfig",
    "get_config",
    "set_config",
    "load_config_from_file",
    "update_config",
    "reset_config",
    "auto_load_config",
    # Validation
    "ValidationError",
    "validate_text_input",
    "validate_category_name",
    "validate_voting_rounds",
    "validate_confidence_score",
    "validate_temperature",
    "validate_max_tokens",
    "validate_batch_texts",
    "validate_category_set",
    "validate_llm_response",
    "validate_api_key",
    "validate_timeout",
    "sanitize_text_for_prompt",
    "validate_classification_result",
]
