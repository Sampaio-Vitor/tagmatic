"""Pytest configuration and fixtures for Tagmatic tests."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from tagmatic import Category, CategorySet
from tagmatic.providers.base import ClassificationResult


@pytest.fixture
def sample_categories():
    """Sample categories for testing."""
    return [
        Category(name="positive", description="Text expressing positive emotions, satisfaction, or happiness"),
        Category(name="negative", description="Text expressing negative emotions, complaints, or dissatisfaction"),
        Category(name="neutral", description="Text that is factual or doesn't express strong emotions")
    ]


@pytest.fixture
def category_set(sample_categories):
    """CategorySet fixture for testing."""
    return CategorySet(categories=sample_categories)


@pytest.fixture
def sentiment_categories():
    """Sentiment analysis categories."""
    return CategorySet(categories=[
        Category(name="happy", description="Expressions of joy, happiness, or positive emotions"),
        Category(name="sad", description="Expressions of sadness, disappointment, or negative emotions"),
        Category(name="angry", description="Expressions of anger, frustration, or irritation"),
        Category(name="neutral", description="Neutral or factual statements without strong emotion")
    ])


@pytest.fixture
def topic_categories():
    """Topic classification categories."""
    return CategorySet(categories=[
        Category(name="technology", description="Content about technology, software, hardware, or digital topics"),
        Category(name="sports", description="Content about sports, athletics, games, or competitions"),
        Category(name="politics", description="Content about politics, government, elections, or policy"),
        Category(name="entertainment", description="Content about movies, music, celebrities, or entertainment")
    ])


@pytest.fixture
def mock_llm():
    """Mock LangChain LLM for testing."""
    llm = Mock()
    llm.invoke = Mock()
    return llm


@pytest.fixture
def mock_llm_response():
    """Mock LLM response object."""
    response = Mock()
    response.content = "positive"
    return response


@pytest.fixture
def mock_structured_response():
    """Mock structured LLM response."""
    return {
        "category": "positive",
        "confidence": 0.85,
        "reasoning": "The text expresses satisfaction and happiness"
    }


@pytest.fixture
def sample_texts():
    """Sample texts for classification testing."""
    return [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "The weather is 72 degrees today.",
        "I'm so excited about the new features!",
        "This doesn't work at all. Very disappointed.",
        "The meeting is scheduled for 3 PM."
    ]


@pytest.fixture
def classification_result():
    """Sample classification result."""
    return ClassificationResult(
        category="positive",
        confidence=0.85,
        raw_response="positive",
        metadata={"test": True}
    )


@pytest.fixture
def batch_classification_results():
    """Sample batch classification results."""
    return [
        ClassificationResult("positive", 0.9, "positive", {"index": 0}),
        ClassificationResult("negative", 0.8, "negative", {"index": 1}),
        ClassificationResult("neutral", 0.7, "neutral", {"index": 2})
    ]


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: List[str] = None, structured_responses: List[Dict] = None):
        self.responses = responses or ["positive", "negative", "neutral"]
        self.structured_responses = structured_responses or []
        self.call_count = 0
    
    def invoke(self, prompt: str) -> Mock:
        """Mock invoke method."""
        response = Mock()
        if self.call_count < len(self.responses):
            response.content = self.responses[self.call_count]
        else:
            response.content = self.responses[0]
        self.call_count += 1
        return response


@pytest.fixture
def mock_provider():
    """Mock LLM provider fixture."""
    return MockLLMProvider()


@pytest.fixture
def voting_responses():
    """Mock responses for voting classifier testing."""
    return ["positive", "positive", "negative", "positive", "positive"]


@pytest.fixture
def mock_voting_provider(voting_responses):
    """Mock provider that returns specific responses for voting tests."""
    return MockLLMProvider(responses=voting_responses)


# Test data fixtures
@pytest.fixture
def valid_category_data():
    """Valid category data for testing."""
    return {
        "name": "test_category",
        "description": "A test category for unit testing",
        "examples": ["example 1", "example 2"]
    }


@pytest.fixture
def invalid_category_data():
    """Invalid category data for testing."""
    return [
        {"name": "", "description": "Empty name"},
        {"name": "test", "description": ""},
        {"name": None, "description": "None name"},
        {"description": "Missing name"},
        {}
    ]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "default_temperature": 0.7,
        "default_max_tokens": 100,
        "default_timeout": 30.0,
        "include_examples_by_default": True,
        "voting_rounds": 3,
        "enable_logging": True
    }


# Utility functions for tests
def assert_classification_result(result: ClassificationResult, expected_category: str = None):
    """Assert that a classification result is valid."""
    assert isinstance(result, ClassificationResult)
    assert isinstance(result.category, str)
    assert len(result.category) > 0
    
    if expected_category:
        assert result.category == expected_category
    
    if result.confidence is not None:
        assert 0.0 <= result.confidence <= 1.0


def assert_voting_result(result, expected_category: str = None, min_confidence: float = 0.0):
    """Assert that a voting result is valid."""
    from tagmatic.providers.base import VotingResult
    
    assert isinstance(result, VotingResult)
    assert isinstance(result.category, str)
    assert len(result.category) > 0
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert result.confidence >= min_confidence
    assert isinstance(result.votes, list)
    assert len(result.votes) > 0
    assert isinstance(result.vote_counts, dict)
    assert len(result.vote_counts) > 0
    assert isinstance(result.individual_results, list)
    assert len(result.individual_results) == len(result.votes)
    
    if expected_category:
        assert result.category == expected_category
    
    # Verify vote counts add up
    assert sum(result.vote_counts.values()) == len(result.votes)
    
    # Verify winning category has most votes
    max_votes = max(result.vote_counts.values())
    assert result.vote_counts[result.category] == max_votes
