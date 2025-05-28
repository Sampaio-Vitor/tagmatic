"""Tests for the Classifier classes."""

import pytest
from unittest.mock import Mock, patch
from typing import List

from tagmatic import Classifier, Category, CategorySet
from tagmatic.core.classifier import DefaultLLMProvider, StructuredLLMProvider
from tagmatic.providers.base import ClassificationResult, VotingResult, LLMProviderError
from tests.conftest import assert_classification_result, assert_voting_result


class TestDefaultLLMProvider:
    """Test cases for DefaultLLMProvider."""
    
    def test_provider_creation(self, mock_llm):
        """Test basic provider creation."""
        provider = DefaultLLMProvider(llm=mock_llm)
        assert provider.llm == mock_llm
        assert provider.temperature is None
        assert provider.max_tokens is None
        assert provider.timeout is None
    
    def test_provider_creation_with_params(self, mock_llm):
        """Test provider creation with parameters."""
        provider = DefaultLLMProvider(
            llm=mock_llm,
            temperature=0.7,
            max_tokens=100,
            timeout=30.0
        )
        assert provider.llm == mock_llm
        assert provider.temperature == 0.7
        assert provider.max_tokens == 100
        assert provider.timeout == 30.0
    
    def test_classify_success(self, mock_llm, category_set):
        """Test successful classification."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "positive"
        mock_llm.invoke.return_value = mock_response
        
        provider = DefaultLLMProvider(llm=mock_llm)
        result = provider.classify("I love this!", category_set)
        
        assert_classification_result(result, "positive")
        assert result.raw_response == "positive"
        assert "prompt_length" in result.metadata
        mock_llm.invoke.assert_called_once()
    
    def test_classify_invalid_response(self, mock_llm, category_set):
        """Test classification with invalid LLM response."""
        # Mock LLM response with invalid category
        mock_response = Mock()
        mock_response.content = "invalid_category"
        mock_llm.invoke.return_value = mock_response
        
        provider = DefaultLLMProvider(llm=mock_llm)
        
        with pytest.raises(LLMProviderError):
            provider.classify("I love this!", category_set)
    
    def test_classify_llm_error(self, mock_llm, category_set):
        """Test classification when LLM raises error."""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        provider = DefaultLLMProvider(llm=mock_llm)
        
        with pytest.raises(LLMProviderError, match="Classification failed"):
            provider.classify("I love this!", category_set)
    
    def test_classify_batch(self, mock_llm, category_set, sample_texts):
        """Test batch classification."""
        # Mock LLM responses
        responses = ["positive", "negative", "neutral"]
        mock_responses = []
        for resp in responses:
            mock_response = Mock()
            mock_response.content = resp
            mock_responses.append(mock_response)
        
        mock_llm.invoke.side_effect = mock_responses
        
        provider = DefaultLLMProvider(llm=mock_llm)
        results = provider.classify_batch(sample_texts[:3], category_set)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert_classification_result(result, responses[i])
        
        assert mock_llm.invoke.call_count == 3
    
    def test_classify_batch_with_errors(self, mock_llm, category_set, sample_texts):
        """Test batch classification with some errors."""
        # Mock mixed responses (some success, some errors)
        mock_response1 = Mock()
        mock_response1.content = "positive"
        
        mock_llm.invoke.side_effect = [
            mock_response1,
            Exception("LLM error"),
            mock_response1
        ]
        
        provider = DefaultLLMProvider(llm=mock_llm)
        results = provider.classify_batch(sample_texts[:3], category_set)
        
        assert len(results) == 3
        assert_classification_result(results[0], "positive")
        assert results[1].category == "unknown"  # Error case
        assert_classification_result(results[2], "positive")


class TestStructuredLLMProvider:
    """Test cases for StructuredLLMProvider."""
    
    def test_structured_classify_success(self, mock_llm, category_set):
        """Test successful structured classification."""
        provider = StructuredLLMProvider(llm=mock_llm)
        
        # Mock the chain response
        mock_response = {
            "category": "positive",
            "confidence": 0.85
        }
        
        with patch('tagmatic.core.classifier.LangChainPromptTemplate') as mock_prompt_template, \
             patch('tagmatic.core.classifier.JsonOutputParser') as mock_parser:
            
            # Mock the chain components
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_response
            
            # Mock the prompt template
            mock_prompt_instance = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt_instance
            
            # Mock the chain creation (prompt | llm | parser)
            mock_prompt_instance.__or__ = Mock(return_value=mock_chain)
            
            result = provider.classify("I love this!", category_set)
            
            assert_classification_result(result, "positive")
            assert result.confidence == 0.85
            assert "structured_output" in result.metadata
    
    def test_structured_classify_invalid_response(self, mock_llm, category_set):
        """Test structured classification with invalid response."""
        provider = StructuredLLMProvider(llm=mock_llm)
        
        with patch('tagmatic.core.classifier.LangChainPromptTemplate') as mock_prompt_template, \
             patch('tagmatic.core.classifier.JsonOutputParser') as mock_parser:
            
            # Mock the chain with invalid response
            mock_chain = Mock()
            mock_chain.invoke.return_value = "not a dict"
            mock_prompt_template.from_template.return_value.__or__ = lambda self, other: mock_chain
            
            with pytest.raises(LLMProviderError):
                provider.classify("I love this!", category_set)


class TestClassifier:
    """Test cases for the main Classifier class."""
    
    def test_classifier_creation_default(self, mock_llm, category_set):
        """Test basic classifier creation."""
        classifier = Classifier(llm=mock_llm, categories=category_set)
        
        assert classifier.llm == mock_llm
        assert classifier.categories == category_set
        assert isinstance(classifier.provider, DefaultLLMProvider)
    
    def test_classifier_creation_structured(self, mock_llm, category_set):
        """Test classifier creation with structured output."""
        classifier = Classifier(
            llm=mock_llm,
            categories=category_set,
            use_structured_output=True
        )
        
        assert isinstance(classifier.provider, StructuredLLMProvider)
    
    def test_classifier_creation_custom_provider(self, mock_llm, category_set):
        """Test classifier creation with custom provider."""
        custom_provider = DefaultLLMProvider(llm=mock_llm)
        classifier = Classifier(
            llm=mock_llm,
            categories=category_set,
            provider=custom_provider
        )
        
        assert classifier.provider == custom_provider
    
    def test_classify_success(self, mock_llm, category_set):
        """Test successful classification."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "positive"
        mock_llm.invoke.return_value = mock_response
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = classifier.classify("I love this!")
        
        assert_classification_result(result, "positive")
    
    def test_classify_no_categories(self, mock_llm):
        """Test classification without categories."""
        classifier = Classifier(llm=mock_llm)
        
        with pytest.raises(Exception, match="No categories provided"):
            classifier.classify("I love this!")
    
    def test_classify_with_override_categories(self, mock_llm, category_set, sentiment_categories):
        """Test classification with override categories."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "happy"
        mock_llm.invoke.return_value = mock_response
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = classifier.classify("I love this!", categories=sentiment_categories)
        
        assert_classification_result(result, "happy")
    
    def test_classify_batch(self, mock_llm, category_set, sample_texts):
        """Test batch classification."""
        # Mock LLM responses
        responses = ["positive", "negative", "neutral"]
        mock_responses = []
        for resp in responses:
            mock_response = Mock()
            mock_response.content = resp
            mock_responses.append(mock_response)
        
        mock_llm.invoke.side_effect = mock_responses
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        results = classifier.classify_batch(sample_texts[:3])
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert_classification_result(result, responses[i])
    
    def test_voting_classify(self, mock_llm, category_set):
        """Test voting classification."""
        # Mock LLM responses for voting
        responses = ["positive", "positive", "negative", "positive", "positive"]
        mock_responses = []
        for resp in responses:
            mock_response = Mock()
            mock_response.content = resp
            mock_responses.append(mock_response)
        
        mock_llm.invoke.side_effect = mock_responses
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = classifier.voting_classify("I love this!", voting_rounds=5)
        
        assert_voting_result(result, "positive", min_confidence=0.6)
        assert result.total_votes == 5
        assert result.winning_votes == 4
        assert result.vote_counts["positive"] == 4
        assert result.vote_counts["negative"] == 1
    
    def test_voting_classify_unanimous(self, mock_llm, category_set):
        """Test voting classification with unanimous result."""
        # Mock unanimous LLM responses
        responses = ["positive", "positive", "positive"]
        mock_responses = []
        for resp in responses:
            mock_response = Mock()
            mock_response.content = resp
            mock_responses.append(mock_response)
        
        mock_llm.invoke.side_effect = mock_responses
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = classifier.voting_classify("I love this!", voting_rounds=3)
        
        assert_voting_result(result, "positive", min_confidence=1.0)
        assert result.is_unanimous
        assert result.confidence == 1.0
    
    def test_set_categories(self, mock_llm, category_set, sentiment_categories):
        """Test setting categories."""
        classifier = Classifier(llm=mock_llm, categories=category_set)
        
        assert classifier.get_categories() == category_set
        
        classifier.set_categories(sentiment_categories)
        assert classifier.get_categories() == sentiment_categories
    
    def test_get_info(self, mock_llm, category_set):
        """Test getting classifier info."""
        classifier = Classifier(llm=mock_llm, categories=category_set)
        info = classifier.get_info()
        
        assert "classifier_type" in info
        assert "provider_info" in info
        assert "has_categories" in info
        assert info["has_categories"] is True
        assert "categories" in info
        assert info["categories"]["count"] == 3
    
    def test_get_info_no_categories(self, mock_llm):
        """Test getting classifier info without categories."""
        classifier = Classifier(llm=mock_llm)
        info = classifier.get_info()
        
        assert info["has_categories"] is False
        assert "categories" not in info
    
    def test_validate_connection_success(self, mock_llm, category_set):
        """Test successful connection validation."""
        mock_llm.invoke.return_value = "test response"
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        assert classifier.validate_connection() is True
    
    def test_validate_connection_failure(self, mock_llm, category_set):
        """Test failed connection validation."""
        mock_llm.invoke.side_effect = Exception("Connection error")
        
        classifier = Classifier(llm=mock_llm, categories=category_set)
        assert classifier.validate_connection() is False
    
    def test_str_representation(self, mock_llm, category_set):
        """Test string representation."""
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = str(classifier)
        
        assert "Classifier" in result
        assert "DefaultLLMProvider" in result
        assert "3 categories" in result
    
    def test_repr_representation(self, mock_llm, category_set):
        """Test detailed representation."""
        classifier = Classifier(llm=mock_llm, categories=category_set)
        result = repr(classifier)
        
        assert "Classifier" in result
        assert "DefaultLLMProvider" in result
        assert "CategorySet" in result
