"""Base LLM provider interface for Tagmatic."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from ..core.category import CategorySet
from ..utils.validation import ValidationError


class ClassificationResult:
    """Result of a text classification operation."""
    
    def __init__(
        self,
        category: str,
        confidence: Optional[float] = None,
        raw_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize classification result.
        
        Args:
            category: Predicted category name
            confidence: Confidence score (0.0 to 1.0)
            raw_response: Raw LLM response
            metadata: Additional metadata about the classification
        """
        self.category = category
        self.confidence = confidence
        self.raw_response = raw_response
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """String representation of the result."""
        if self.confidence is not None:
            return f"ClassificationResult(category='{self.category}', confidence={self.confidence:.3f})"
        return f"ClassificationResult(category='{self.category}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the result."""
        return (
            f"ClassificationResult(category='{self.category}', "
            f"confidence={self.confidence}, "
            f"raw_response='{self.raw_response}', "
            f"metadata={self.metadata})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {"category": self.category}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.raw_response is not None:
            result["raw_response"] = self.raw_response
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class VotingResult:
    """Result of a voting classification operation."""
    
    def __init__(
        self,
        category: str,
        confidence: float,
        votes: List[str],
        vote_counts: Dict[str, int],
        individual_results: List[ClassificationResult]
    ):
        """
        Initialize voting result.
        
        Args:
            category: Final predicted category (winner)
            confidence: Confidence based on vote distribution
            votes: List of all votes
            vote_counts: Count of votes per category
            individual_results: Individual classification results
        """
        self.category = category
        self.confidence = confidence
        self.votes = votes
        self.vote_counts = vote_counts
        self.individual_results = individual_results
    
    @property
    def total_votes(self) -> int:
        """Total number of votes."""
        return len(self.votes)
    
    @property
    def winning_votes(self) -> int:
        """Number of votes for the winning category."""
        return self.vote_counts.get(self.category, 0)
    
    @property
    def is_unanimous(self) -> bool:
        """Whether all votes were for the same category."""
        return len(self.vote_counts) == 1
    
    def __str__(self) -> str:
        """String representation of the voting result."""
        return (
            f"VotingResult(category='{self.category}', "
            f"confidence={self.confidence:.3f}, "
            f"votes={self.winning_votes}/{self.total_votes})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of the voting result."""
        return (
            f"VotingResult(category='{self.category}', "
            f"confidence={self.confidence}, "
            f"votes={self.votes}, "
            f"vote_counts={self.vote_counts})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voting result to dictionary."""
        return {
            "category": self.category,
            "confidence": self.confidence,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
            "total_votes": self.total_votes,
            "winning_votes": self.winning_votes,
            "is_unanimous": self.is_unanimous,
            "individual_results": [result.to_dict() for result in self.individual_results]
        }


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement
    to work with Tagmatic's classification system.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize the LLM provider.
        
        Args:
            llm: LangChain LLM instance
            temperature: Temperature for LLM calls
            max_tokens: Maximum tokens for responses
            timeout: Timeout for LLM calls in seconds
        """
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    @abstractmethod
    def classify(
        self,
        text: str,
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> ClassificationResult:
        """
        Classify a single text into one of the provided categories.
        
        Args:
            text: Text to classify
            categories: Available categories
            prompt_template: Optional custom prompt template
            **kwargs: Additional provider-specific parameters
            
        Returns:
            ClassificationResult with the predicted category
            
        Raises:
            ValidationError: If input validation fails
            Exception: If classification fails
        """
        pass
    
    @abstractmethod
    def classify_batch(
        self,
        texts: List[str],
        categories: CategorySet,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts into categories.
        
        Args:
            texts: List of texts to classify
            categories: Available categories
            prompt_template: Optional custom prompt template
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of ClassificationResult objects
            
        Raises:
            ValidationError: If input validation fails
            Exception: If classification fails
        """
        pass
    
    def voting_classify(
        self,
        text: str,
        categories: CategorySet,
        voting_rounds: int = 3,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> VotingResult:
        """
        Classify text using voting mechanism for improved accuracy.
        
        Args:
            text: Text to classify
            categories: Available categories
            voting_rounds: Number of voting rounds (must be odd)
            prompt_template: Optional custom prompt template
            **kwargs: Additional provider-specific parameters
            
        Returns:
            VotingResult with aggregated classification
            
        Raises:
            ValidationError: If input validation fails
            Exception: If classification fails
        """
        from ..utils.validation import validate_voting_rounds
        
        # Validate voting rounds
        voting_rounds = validate_voting_rounds(voting_rounds)
        
        # Perform multiple classifications
        individual_results = []
        votes = []
        
        for round_num in range(voting_rounds):
            try:
                result = self.classify(
                    text=text,
                    categories=categories,
                    prompt_template=prompt_template,
                    round_number=round_num + 1,
                    **kwargs
                )
                individual_results.append(result)
                votes.append(result.category)
            except Exception as e:
                # If a single vote fails, we can still continue with other votes
                # as long as we have at least one successful vote
                if len(votes) == 0 and round_num == voting_rounds - 1:
                    # If this is the last round and we have no votes, re-raise
                    raise e
                continue
        
        if not votes:
            raise Exception("All voting rounds failed")
        
        # Count votes
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Determine winner (category with most votes)
        winning_category = max(vote_counts, key=vote_counts.get)
        winning_votes = vote_counts[winning_category]
        total_votes = len(votes)
        
        # Calculate confidence based on vote distribution
        confidence = winning_votes / total_votes
        
        return VotingResult(
            category=winning_category,
            confidence=confidence,
            votes=votes,
            vote_counts=vote_counts,
            individual_results=individual_results
        )
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider_type": self.__class__.__name__,
            "llm_type": type(self.llm).__name__,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
    
    def validate_connection(self) -> bool:
        """
        Validate that the LLM connection is working.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try a simple test call
            test_response = self.llm.invoke("Test")
            return isinstance(test_response, str) and len(test_response) > 0
        except Exception:
            return False
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(llm={type(self.llm).__name__})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"llm={type(self.llm).__name__}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, "
            f"timeout={self.timeout})"
        )


class LLMProviderError(Exception):
    """Exception raised by LLM providers."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Exception raised when LLM call times out."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Exception raised when LLM rate limit is exceeded."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Exception raised when LLM authentication fails."""
    pass
