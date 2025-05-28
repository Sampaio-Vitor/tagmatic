"""Configuration management for Tagmatic."""

import os
from typing import Any, Dict, Optional, Union
from pathlib import Path
import json
import yaml
from pydantic import BaseModel, Field


class TagmaticConfig(BaseModel):
    """Configuration settings for Tagmatic."""
    
    # LLM Settings
    default_temperature: float = Field(
        default=0.1,
        description="Default temperature for LLM calls",
        ge=0.0,
        le=2.0
    )
    default_max_tokens: Optional[int] = Field(
        default=100,
        description="Default max tokens for LLM responses",
        ge=1
    )
    default_timeout: int = Field(
        default=30,
        description="Default timeout for LLM calls in seconds",
        ge=1
    )
    
    # Voting Classifier Settings
    default_voting_rounds: int = Field(
        default=3,
        description="Default number of voting rounds",
        ge=1
    )
    voting_confidence_threshold: float = Field(
        default=0.6,
        description="Minimum confidence threshold for voting results",
        ge=0.0,
        le=1.0
    )
    
    # Prompt Settings
    include_examples_by_default: bool = Field(
        default=True,
        description="Whether to include category examples in prompts by default"
    )
    max_examples_per_category: int = Field(
        default=3,
        description="Maximum number of examples to include per category",
        ge=1,
        le=10
    )
    
    # Performance Settings
    enable_caching: bool = Field(
        default=True,
        description="Whether to enable response caching"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
        ge=0
    )
    
    # Logging Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_llm_calls: bool = Field(
        default=False,
        description="Whether to log LLM calls for debugging"
    )
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    def __init__(self, **data):
        """Initialize config with environment variable loading."""
        # Load API keys from environment if not provided
        if 'openai_api_key' not in data:
            data['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if 'anthropic_api_key' not in data:
            data['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
        
        super().__init__(**data)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'TagmaticConfig':
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to JSON or YAML configuration file
            
        Returns:
            TagmaticConfig instance
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_path.suffix.lower() == '.json':
            data = json.loads(content)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return cls(**data)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path where to save the configuration
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.model_dump(exclude_none=True)
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def update(self, **kwargs) -> 'TagmaticConfig':
        """
        Create a new config instance with updated values.
        
        Args:
            **kwargs: Configuration values to update
            
        Returns:
            New TagmaticConfig instance with updated values
        """
        current_data = self.model_dump()
        current_data.update(kwargs)
        return TagmaticConfig(**current_data)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider: Provider name ('openai', 'anthropic')
            
        Returns:
            API key if available, None otherwise
        """
        provider = provider.lower()
        if provider == 'openai':
            return self.openai_api_key
        elif provider == 'anthropic':
            return self.anthropic_api_key
        else:
            return None


# Global configuration instance
_global_config: Optional[TagmaticConfig] = None


def get_config() -> TagmaticConfig:
    """
    Get the global configuration instance.
    
    Returns:
        TagmaticConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = TagmaticConfig()
    return _global_config


def set_config(config: TagmaticConfig) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: TagmaticConfig instance to set as global
    """
    global _global_config
    _global_config = config


def load_config_from_file(file_path: Union[str, Path]) -> TagmaticConfig:
    """
    Load configuration from file and set as global.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Loaded TagmaticConfig instance
    """
    config = TagmaticConfig.from_file(file_path)
    set_config(config)
    return config


def update_config(**kwargs) -> TagmaticConfig:
    """
    Update the global configuration with new values.
    
    Args:
        **kwargs: Configuration values to update
        
    Returns:
        Updated TagmaticConfig instance
    """
    current_config = get_config()
    new_config = current_config.update(**kwargs)
    set_config(new_config)
    return new_config


def reset_config() -> TagmaticConfig:
    """
    Reset configuration to default values.
    
    Returns:
        New default TagmaticConfig instance
    """
    global _global_config
    _global_config = TagmaticConfig()
    return _global_config


# Configuration file search paths
DEFAULT_CONFIG_PATHS = [
    Path.cwd() / "tagmatic.json",
    Path.cwd() / "tagmatic.yaml",
    Path.cwd() / "tagmatic.yml",
    Path.cwd() / ".tagmatic.json",
    Path.cwd() / ".tagmatic.yaml",
    Path.cwd() / ".tagmatic.yml",
    Path.home() / ".tagmatic" / "config.json",
    Path.home() / ".tagmatic" / "config.yaml",
    Path.home() / ".tagmatic" / "config.yml",
]


def find_config_file() -> Optional[Path]:
    """
    Find configuration file in default locations.
    
    Returns:
        Path to configuration file if found, None otherwise
    """
    for config_path in DEFAULT_CONFIG_PATHS:
        if config_path.exists():
            return config_path
    return None


def auto_load_config() -> TagmaticConfig:
    """
    Automatically load configuration from default locations.
    
    Returns:
        TagmaticConfig instance (default if no config file found)
    """
    config_file = find_config_file()
    if config_file:
        return load_config_from_file(config_file)
    else:
        return get_config()
