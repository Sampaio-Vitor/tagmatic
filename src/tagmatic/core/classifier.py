"""Main classifier implementation for Tagmatic."""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate as LangChainPromptTemplate

from ..core.category import CategorySet
from ..providers.base import (
    BaseLLMProvider,
    ClassificationResult,
    VotingResult,
    LLMProviderError,
)
from ..prompts.templates import (
    PromptTemplate,
    DefaultClassificationPrompt,
    StructuredClassificationPrompt,
    VotingClassificationPrompt,
    get_template,
)
from ..utils.validation import (
    ValidationError,
    validate_text_input,
    validate_category_set,
    validate_voting_rounds,
    validate_llm_response,
    sanitize_text_for_prompt,
)
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class DefaultLLMProvider(BaseLLMProvider):
    """Default LLM provider implementation using LangChain."""
    
    def classify(
        self,
        text: str,
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify a single text using the LLM.
        
        Args:
            text: Text to classify
            categories: Available categories
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            ClassificationResult with the predicted category
        """
        # Validate inputs
        text = validate_text_input(text)
        categories = validate_category_set(categories)
        
        # Sanitize text for prompt
        sanitized_text = sanitize_text_for_prompt(text)
        
        # Get prompt template
        if prompt_template:
            # Use custom template
            prompt = prompt_template.format(text=sanitized_text, categories=categories)
        else:
            # Use default template
            template = get_template("default")
            prompt = template.generate(
                text=sanitized_text,
                categories=categories,
                include_examples=get_config().include_examples_by_default
            )
        
        try:
            # Call LLM
            response = self.llm.invoke(prompt)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Validate and clean response
            valid_categories = categories.get_category_names()
            predicted_category = validate_llm_response(response_text, valid_categories)
            
            return ClassificationResult(
                category=predicted_category,
                raw_response=response_text,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response_text),
                    "valid_categories": valid_categories,
                }
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise LLMProviderError(f"Classification failed: {e}") from e
    
    def classify_batch(
        self,
        texts: List[str],
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            categories: Available categories
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            List of ClassificationResult objects
        """
        results = []
        for text in texts:
            try:
                result = self.classify(
                    text=text,
                    categories=categories,
                    prompt_template=prompt_template,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text '{text[:50]}...': {e}")
                # Add a failed result
                results.append(ClassificationResult(
                    category="unknown",
                    raw_response=str(e),
                    metadata={"error": str(e)}
                ))
        
        return results


class StructuredLLMProvider(BaseLLMProvider):
    """LLM provider that uses structured output parsing."""
    
    def classify(
        self,
        text: str,
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify text using structured output parsing.
        
        Args:
            text: Text to classify
            categories: Available categories
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            ClassificationResult with confidence score
        """
        # Validate inputs
        text = validate_text_input(text)
        categories = validate_category_set(categories)
        
        # Sanitize text for prompt
        sanitized_text = sanitize_text_for_prompt(text)
        
        # Get structured prompt template
        template = get_template("structured")
        prompt = template.generate(
            text=sanitized_text,
            categories=categories,
            include_examples=get_config().include_examples_by_default
        )
        
        try:
            # Set up JSON output parser
            parser = JsonOutputParser()
            
            # Create LangChain prompt template
            langchain_prompt = LangChainPromptTemplate.from_template(prompt)
            
            # Create chain
            chain = langchain_prompt | self.llm | parser
            
            # Call LLM with structured output
            response = chain.invoke({})
            
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Expected dict response, got {type(response)}")
            
            if "category" not in response:
                raise ValueError("Response missing 'category' field")
            
            # Validate category
            valid_categories = categories.get_category_names()
            predicted_category = validate_llm_response(response["category"], valid_categories)
            
            # Extract confidence if available
            confidence = response.get("confidence")
            if confidence is not None:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            return ClassificationResult(
                category=predicted_category,
                confidence=confidence,
                raw_response=json.dumps(response),
                metadata={
                    "structured_output": True,
                    "prompt_length": len(prompt),
                    "valid_categories": valid_categories,
                }
            )
            
        except Exception as e:
            logger.error(f"Structured classification failed: {e}")
            raise LLMProviderError(f"Structured classification failed: {e}") from e
    
    def classify_batch(
        self,
        texts: List[str],
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> List[ClassificationResult]:
        """Classify multiple texts using structured output."""
        results = []
        for text in texts:
            try:
                result = self.classify(
                    text=text,
                    categories=categories,
                    prompt_template=prompt_template,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to classify text '{text[:50]}...': {e}")
                results.append(ClassificationResult(
                    category="unknown",
                    raw_response=str(e),
                    metadata={"error": str(e)}
                ))
        
        return results


class Classifier:
    """
    Main text classifier for Tagmatic.
    
    This is the primary interface for text classification using LLMs.
    It supports single classification, batch classification, and voting
    classification for improved accuracy.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        categories: Optional[CategorySet] = None,
        provider: Optional[BaseLLMProvider] = None,
        use_structured_output: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            llm: LangChain LLM instance
            categories: Default categories for classification
            provider: Custom LLM provider (optional)
            use_structured_output: Whether to use structured output parsing
            temperature: Temperature for LLM calls
            max_tokens: Maximum tokens for responses
            timeout: Timeout for LLM calls
        """
        self.llm = llm
        self.categories = categories
        
        # Set up provider
        if provider:
            self.provider = provider
        elif use_structured_output:
            self.provider = StructuredLLMProvider(
                llm=llm,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        else:
            self.provider = DefaultLLMProvider(
                llm=llm,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        
        # Configuration
        self.config = get_config()
        
        logger.info(f"Initialized Classifier with {type(self.provider).__name__}")
    
    def classify(
        self,
        text: str,
        categories: Optional[CategorySet] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify a single text.
        
        Args:
            text: Text to classify
            categories: Categories to use (defaults to instance categories)
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            ClassificationResult with the predicted category
            
        Raises:
            ValidationError: If input validation fails
            LLMProviderError: If classification fails
        """
        # Use provided categories or default
        categories = categories or self.categories
        if categories is None:
            raise ValidationError("No categories provided for classification")
        
        return self.provider.classify(
            text=text,
            categories=categories,
            prompt_template=prompt_template,
            **kwargs
        )
    
    def classify_batch(
        self,
        texts: List[str],
        categories: Optional[CategorySet] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of texts to classify
            categories: Categories to use (defaults to instance categories)
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            List of ClassificationResult objects
            
        Raises:
            ValidationError: If input validation fails
        """
        # Use provided categories or default
        categories = categories or self.categories
        if categories is None:
            raise ValidationError("No categories provided for classification")
        
        return self.provider.classify_batch(
            texts=texts,
            categories=categories,
            prompt_template=prompt_template,
            **kwargs
        )
    
    def voting_classify(
        self,
        text: str,
        voting_rounds: int = 3,
        categories: Optional[CategorySet] = None,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> VotingResult:
        """
        Classify text using voting mechanism for improved accuracy.
        
        Args:
            text: Text to classify
            voting_rounds: Number of voting rounds (must be odd)
            categories: Categories to use (defaults to instance categories)
            prompt_template: Optional custom prompt template
            **kwargs: Additional parameters
            
        Returns:
            VotingResult with aggregated classification
            
        Raises:
            ValidationError: If input validation fails
            LLMProviderError: If classification fails
        """
        # Use provided categories or default
        categories = categories or self.categories
        if categories is None:
            raise ValidationError("No categories provided for classification")
        
        return self.provider.voting_classify(
            text=text,
            categories=categories,
            voting_rounds=voting_rounds,
            prompt_template=prompt_template,
            **kwargs
        )
    
    def set_categories(self, categories: CategorySet) -> None:
        """
        Set the default categories for this classifier.
        
        Args:
            categories: CategorySet to use as default
        """
        self.categories = validate_category_set(categories)
        logger.info(f"Updated classifier categories: {categories.get_category_names()}")
    
    def get_categories(self) -> Optional[CategorySet]:
        """
        Get the current default categories.
        
        Returns:
            Current CategorySet or None if not set
        """
        return self.categories
    
    def validate_connection(self) -> bool:
        """
        Validate that the LLM connection is working.
        
        Returns:
            True if connection is valid, False otherwise
        """
        return self.provider.validate_connection()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this classifier.
        
        Returns:
            Dictionary with classifier information
        """
        info = {
            "classifier_type": self.__class__.__name__,
            "provider_info": self.provider.get_provider_info(),
            "has_categories": self.categories is not None,
        }
        
        if self.categories:
            info["categories"] = {
                "count": len(self.categories),
                "names": self.categories.get_category_names(),
            }
        
        return info
    
    def __str__(self) -> str:
        """String representation of the classifier."""
        category_info = f"{len(self.categories)} categories" if self.categories else "no categories"
        return f"Classifier({type(self.provider).__name__}, {category_info})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the classifier."""
        return (
            f"Classifier("
            f"provider={type(self.provider).__name__}, "
            f"categories={self.categories}, "
            f"llm={type(self.llm).__name__})"
        )
