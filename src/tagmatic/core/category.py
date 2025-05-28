"""Category system for text classification."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Category(BaseModel):
    """
    Represents a classification category with name, description, and optional examples.
    
    A Category defines a classification target that the LLM will use to classify text.
    The description should be clear and specific to help the LLM understand what
    types of text belong to this category.
    
    Attributes:
        name: Unique identifier for the category
        description: Clear description of what text belongs in this category
        examples: Optional list of example texts that belong to this category
    """
    
    name: str = Field(
        ...,
        description="Unique name for the category",
        min_length=1,
        max_length=100
    )
    description: str = Field(
        ...,
        description="Clear description of what text belongs in this category",
        min_length=10,
        max_length=1000
    )
    examples: Optional[List[str]] = Field(
        default=None,
        description="Optional list of example texts for this category",
        max_length=10
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate category name format."""
        if not v.strip():
            raise ValueError("Category name cannot be empty or whitespace")
        
        # Remove extra whitespace and convert to lowercase for consistency
        cleaned_name = v.strip().lower()
        
        # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
        if not all(c.isalnum() or c in ' -_' for c in cleaned_name):
            raise ValueError(
                "Category name can only contain letters, numbers, spaces, hyphens, and underscores"
            )
        
        return cleaned_name
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate category description."""
        if not v.strip():
            raise ValueError("Category description cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('examples')
    @classmethod
    def validate_examples(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate category examples."""
        if v is None:
            return v
        
        # Remove empty examples and strip whitespace
        cleaned_examples = [ex.strip() for ex in v if ex.strip()]
        
        if not cleaned_examples:
            return None
        
        # Check for duplicates
        if len(cleaned_examples) != len(set(cleaned_examples)):
            raise ValueError("Category examples must be unique")
        
        return cleaned_examples
    
    def __str__(self) -> str:
        """String representation of the category."""
        return f"Category(name='{self.name}', description='{self.description[:50]}...')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the category."""
        return (
            f"Category(name='{self.name}', "
            f"description='{self.description}', "
            f"examples={self.examples})"
        )


class CategorySet(BaseModel):
    """
    Manages a collection of categories for text classification.
    
    A CategorySet ensures that all categories have unique names and provides
    utilities for serialization, validation, and category management.
    
    Attributes:
        categories: List of Category objects
        name: Optional name for this category set
        description: Optional description of this category set
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    categories: List[Category] = Field(
        ...,
        description="List of categories in this set",
        min_length=2,
        max_length=50
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional name for this category set",
        max_length=100
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of this category set",
        max_length=500
    )
    
    @field_validator('categories')
    @classmethod
    def validate_categories(cls, v: List[Category]) -> List[Category]:
        """Validate that all category names are unique."""
        if len(v) < 2:
            raise ValueError("CategorySet must contain at least 2 categories")
        
        names = [cat.name for cat in v]
        if len(names) != len(set(names)):
            raise ValueError("All category names must be unique")
        
        return v
    
    def get_category(self, name: str) -> Optional[Category]:
        """
        Get a category by name.
        
        Args:
            name: Name of the category to retrieve
            
        Returns:
            Category object if found, None otherwise
        """
        normalized_name = name.strip().lower()
        for category in self.categories:
            if category.name == normalized_name:
                return category
        return None
    
    def add_category(self, category: Category) -> None:
        """
        Add a new category to the set.
        
        Args:
            category: Category to add
            
        Raises:
            ValueError: If a category with the same name already exists
        """
        if self.get_category(category.name) is not None:
            raise ValueError(f"Category with name '{category.name}' already exists")
        
        # Create a new list with the added category
        new_categories = list(self.categories)
        new_categories.append(category)
        self.categories = new_categories
    
    def remove_category(self, name: str) -> bool:
        """
        Remove a category by name.
        
        Args:
            name: Name of the category to remove
            
        Returns:
            True if category was removed, False if not found
            
        Raises:
            ValueError: If removing would leave less than 2 categories
        """
        normalized_name = name.strip().lower()
        
        # Check if category exists
        category_to_remove = None
        for category in self.categories:
            if category.name == normalized_name:
                category_to_remove = category
                break
        
        if category_to_remove is None:
            return False
        
        # Check if removing would leave less than 2 categories
        if len(self.categories) <= 2:
            raise ValueError("Cannot remove category: CategorySet must have at least 2 categories")
        
        # Remove the category
        new_categories = [cat for cat in self.categories if cat.name != normalized_name]
        self.categories = new_categories
        return True
    
    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return [cat.name for cat in self.categories]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CategorySet to dictionary."""
        return self.model_dump()
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert CategorySet to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_yaml(self) -> str:
        """
        Convert CategorySet to YAML string.
        
        Returns:
            YAML string representation
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CategorySet:
        """
        Create CategorySet from dictionary.
        
        Args:
            data: Dictionary containing category set data
            
        Returns:
            CategorySet instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> CategorySet:
        """
        Create CategorySet from JSON string.
        
        Args:
            json_str: JSON string containing category set data
            
        Returns:
            CategorySet instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> CategorySet:
        """
        Create CategorySet from YAML string.
        
        Args:
            yaml_str: YAML string containing category set data
            
        Returns:
            CategorySet instance
        """
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: str) -> CategorySet:
        """
        Create CategorySet from JSON or YAML file.
        
        Args:
            file_path: Path to JSON or YAML file
            
        Returns:
            CategorySet instance
            
        Raises:
            ValueError: If file format is not supported
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_path.lower().endswith(('.json',)):
            return cls.from_json(content)
        elif file_path.lower().endswith(('.yaml', '.yml')):
            return cls.from_yaml(content)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save CategorySet to JSON or YAML file.
        
        Args:
            file_path: Path where to save the file
            
        Raises:
            ValueError: If file format is not supported
        """
        if file_path.lower().endswith(('.json',)):
            content = self.to_json()
        elif file_path.lower().endswith(('.yaml', '.yml')):
            content = self.to_yaml()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def __len__(self) -> int:
        """Return number of categories."""
        return len(self.categories)
    
    def __iter__(self):
        """Iterate over categories."""
        return iter(self.categories)
    
    def __getitem__(self, index: Union[int, str]) -> Category:
        """Get category by index or name."""
        if isinstance(index, int):
            return self.categories[index]
        elif isinstance(index, str):
            category = self.get_category(index)
            if category is None:
                raise KeyError(f"Category '{index}' not found")
            return category
        else:
            raise TypeError("Index must be int or str")
    
    def __str__(self) -> str:
        """String representation of the category set."""
        return f"CategorySet(categories={len(self.categories)}, names={self.get_category_names()})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the category set."""
        return (
            f"CategorySet(name='{self.name}', "
            f"description='{self.description}', "
            f"categories={self.categories})"
        )
