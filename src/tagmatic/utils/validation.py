"""Input validation utilities for Tagmatic."""

import re
from typing import Any, List, Optional, Union
from ..core.category import Category, CategorySet


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_text_input(text: str, min_length: int = 1, max_length: int = 10000) -> str:
    """
    Validate text input for classification.
    
    Args:
        text: Text to validate
        min_length: Minimum allowed text length
        max_length: Maximum allowed text length
        
    Returns:
        Cleaned text
        
    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError(f"Text must be a string, got {type(text)}")
    
    # Strip whitespace
    cleaned_text = text.strip()
    
    if len(cleaned_text) < min_length:
        raise ValidationError(f"Text must be at least {min_length} characters long")
    
    if len(cleaned_text) > max_length:
        raise ValidationError(f"Text must be no more than {max_length} characters long")
    
    # Check for empty or whitespace-only text
    if not cleaned_text:
        raise ValidationError("Text cannot be empty or contain only whitespace")
    
    return cleaned_text


def validate_category_name(name: str) -> str:
    """
    Validate category name format.
    
    Args:
        name: Category name to validate
        
    Returns:
        Cleaned category name
        
    Raises:
        ValidationError: If name is invalid
    """
    if not isinstance(name, str):
        raise ValidationError(f"Category name must be a string, got {type(name)}")
    
    # Strip and normalize
    cleaned_name = name.strip().lower()
    
    if not cleaned_name:
        raise ValidationError("Category name cannot be empty")
    
    if len(cleaned_name) > 100:
        raise ValidationError("Category name must be 100 characters or less")
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9\s\-_]+$', cleaned_name):
        raise ValidationError(
            "Category name can only contain letters, numbers, spaces, hyphens, and underscores"
        )
    
    return cleaned_name


def validate_voting_rounds(rounds: int) -> int:
    """
    Validate voting rounds parameter.
    
    Args:
        rounds: Number of voting rounds
        
    Returns:
        Validated rounds number
        
    Raises:
        ValidationError: If rounds is invalid
    """
    if not isinstance(rounds, int):
        raise ValidationError(f"Voting rounds must be an integer, got {type(rounds)}")
    
    if rounds < 1:
        raise ValidationError("Voting rounds must be at least 1")
    
    if rounds > 21:  # Reasonable upper limit
        raise ValidationError("Voting rounds must be 21 or less")
    
    if rounds % 2 == 0:
        raise ValidationError("Voting rounds must be an odd number to avoid ties")
    
    return rounds


def validate_confidence_score(confidence: float) -> float:
    """
    Validate confidence score.
    
    Args:
        confidence: Confidence score to validate
        
    Returns:
        Validated confidence score
        
    Raises:
        ValidationError: If confidence is invalid
    """
    if not isinstance(confidence, (int, float)):
        raise ValidationError(f"Confidence must be a number, got {type(confidence)}")
    
    confidence = float(confidence)
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValidationError("Confidence must be between 0.0 and 1.0")
    
    return confidence


def validate_temperature(temperature: float) -> float:
    """
    Validate LLM temperature parameter.
    
    Args:
        temperature: Temperature value to validate
        
    Returns:
        Validated temperature
        
    Raises:
        ValidationError: If temperature is invalid
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"Temperature must be a number, got {type(temperature)}")
    
    temperature = float(temperature)
    
    if temperature < 0.0 or temperature > 2.0:
        raise ValidationError("Temperature must be between 0.0 and 2.0")
    
    return temperature


def validate_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Validate max tokens parameter.
    
    Args:
        max_tokens: Maximum tokens value to validate
        
    Returns:
        Validated max tokens
        
    Raises:
        ValidationError: If max_tokens is invalid
    """
    if max_tokens is None:
        return None
    
    if not isinstance(max_tokens, int):
        raise ValidationError(f"Max tokens must be an integer, got {type(max_tokens)}")
    
    if max_tokens < 1:
        raise ValidationError("Max tokens must be at least 1")
    
    if max_tokens > 100000:  # Reasonable upper limit
        raise ValidationError("Max tokens must be 100,000 or less")
    
    return max_tokens


def validate_batch_texts(texts: List[str]) -> List[str]:
    """
    Validate a batch of texts for classification.
    
    Args:
        texts: List of texts to validate
        
    Returns:
        List of validated texts
        
    Raises:
        ValidationError: If any text is invalid
    """
    if not isinstance(texts, list):
        raise ValidationError(f"Texts must be a list, got {type(texts)}")
    
    if not texts:
        raise ValidationError("Text list cannot be empty")
    
    if len(texts) > 1000:  # Reasonable batch size limit
        raise ValidationError("Batch size cannot exceed 1000 texts")
    
    validated_texts = []
    for i, text in enumerate(texts):
        try:
            validated_text = validate_text_input(text)
            validated_texts.append(validated_text)
        except ValidationError as e:
            raise ValidationError(f"Text at index {i} is invalid: {e}")
    
    return validated_texts


def validate_category_set(categories: CategorySet) -> CategorySet:
    """
    Validate a CategorySet for classification.
    
    Args:
        categories: CategorySet to validate
        
    Returns:
        Validated CategorySet
        
    Raises:
        ValidationError: If CategorySet is invalid
    """
    if not isinstance(categories, CategorySet):
        raise ValidationError(f"Categories must be a CategorySet, got {type(categories)}")
    
    if len(categories.categories) < 2:
        raise ValidationError("CategorySet must contain at least 2 categories")
    
    if len(categories.categories) > 50:
        raise ValidationError("CategorySet cannot contain more than 50 categories")
    
    # Validate individual categories
    for category in categories.categories:
        if not isinstance(category, Category):
            raise ValidationError(f"All items must be Category objects, got {type(category)}")
    
    return categories


def validate_llm_response(response: str, valid_categories: List[str]) -> str:
    """
    Validate LLM response for classification.
    
    Args:
        response: LLM response to validate
        valid_categories: List of valid category names
        
    Returns:
        Validated category name
        
    Raises:
        ValidationError: If response is invalid
    """
    if not isinstance(response, str):
        raise ValidationError(f"LLM response must be a string, got {type(response)}")
    
    # Clean the response
    cleaned_response = response.strip().lower()
    
    if not cleaned_response:
        raise ValidationError("LLM response cannot be empty")
    
    # Normalize valid categories for comparison
    normalized_categories = {cat.lower(): cat for cat in valid_categories}
    
    # Check if response matches any valid category
    if cleaned_response in normalized_categories:
        return normalized_categories[cleaned_response]
    
    # Try to extract category from response if it contains extra text
    for normalized_cat, original_cat in normalized_categories.items():
        if normalized_cat in cleaned_response:
            return original_cat
    
    raise ValidationError(
        f"LLM response '{response}' does not match any valid category: {valid_categories}"
    )


def validate_api_key(api_key: Optional[str], provider: str) -> Optional[str]:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        provider: Provider name for context
        
    Returns:
        Validated API key
        
    Raises:
        ValidationError: If API key format is invalid
    """
    if api_key is None:
        return None
    
    if not isinstance(api_key, str):
        raise ValidationError(f"API key must be a string, got {type(api_key)}")
    
    api_key = api_key.strip()
    
    if not api_key:
        raise ValidationError("API key cannot be empty")
    
    # Basic format validation based on provider
    provider = provider.lower()
    
    if provider == "openai":
        if not api_key.startswith("sk-"):
            raise ValidationError("OpenAI API key must start with 'sk-'")
        if len(api_key) < 20:
            raise ValidationError("OpenAI API key appears to be too short")
    
    elif provider == "anthropic":
        if not api_key.startswith("sk-ant-"):
            raise ValidationError("Anthropic API key must start with 'sk-ant-'")
        if len(api_key) < 20:
            raise ValidationError("Anthropic API key appears to be too short")
    
    return api_key


def validate_timeout(timeout: Union[int, float]) -> float:
    """
    Validate timeout parameter.
    
    Args:
        timeout: Timeout value in seconds
        
    Returns:
        Validated timeout
        
    Raises:
        ValidationError: If timeout is invalid
    """
    if not isinstance(timeout, (int, float)):
        raise ValidationError(f"Timeout must be a number, got {type(timeout)}")
    
    timeout = float(timeout)
    
    if timeout <= 0:
        raise ValidationError("Timeout must be greater than 0")
    
    if timeout > 300:  # 5 minutes max
        raise ValidationError("Timeout cannot exceed 300 seconds")
    
    return timeout


def sanitize_text_for_prompt(text: str) -> str:
    """
    Sanitize text for safe inclusion in prompts.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove or escape potentially problematic characters
    sanitized = text.strip()
    
    # Remove excessive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Remove control characters except common ones
    sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
    
    return sanitized


def validate_classification_result(result: dict) -> dict:
    """
    Validate classification result structure.
    
    Args:
        result: Classification result to validate
        
    Returns:
        Validated result
        
    Raises:
        ValidationError: If result structure is invalid
    """
    if not isinstance(result, dict):
        raise ValidationError(f"Classification result must be a dict, got {type(result)}")
    
    required_fields = ['category']
    for field in required_fields:
        if field not in result:
            raise ValidationError(f"Classification result missing required field: {field}")
    
    # Validate category
    if not isinstance(result['category'], str):
        raise ValidationError("Category must be a string")
    
    # Validate confidence if present
    if 'confidence' in result:
        result['confidence'] = validate_confidence_score(result['confidence'])
    
    return result
