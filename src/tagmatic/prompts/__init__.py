"""Prompt templates for text classification."""

from .templates import (
    PromptTemplate,
    DefaultClassificationPrompt,
    StructuredClassificationPrompt,
    VotingClassificationPrompt,
    FewShotPrompt,
    get_template,
    DEFAULT_TEMPLATE,
    STRUCTURED_TEMPLATE,
    VOTING_TEMPLATE,
)

__all__ = [
    "PromptTemplate",
    "DefaultClassificationPrompt",
    "StructuredClassificationPrompt",
    "VotingClassificationPrompt",
    "FewShotPrompt",
    "get_template",
    "DEFAULT_TEMPLATE",
    "STRUCTURED_TEMPLATE",
    "VOTING_TEMPLATE",
]
