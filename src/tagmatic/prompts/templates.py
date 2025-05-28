"""Prompt templates for text classification."""

from typing import List, Optional
from ..core.category import Category, CategorySet


class PromptTemplate:
    """Base class for prompt templates."""
    
    def generate(
        self,
        text: str,
        categories: CategorySet,
        include_examples: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for text classification.
        
        Args:
            text: Text to classify
            categories: CategorySet containing available categories
            include_examples: Whether to include category examples in prompt
            custom_instructions: Optional custom instructions to add
            
        Returns:
            Generated prompt string
        """
        raise NotImplementedError


class DefaultClassificationPrompt(PromptTemplate):
    """Default prompt template for text classification."""
    
    def generate(
        self,
        text: str,
        categories: CategorySet,
        include_examples: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate a classification prompt using the default template.
        
        Args:
            text: Text to classify
            categories: CategorySet containing available categories
            include_examples: Whether to include category examples in prompt
            custom_instructions: Optional custom instructions to add
            
        Returns:
            Generated prompt string
        """
        # Build category descriptions
        category_descriptions = []
        for category in categories.categories:
            desc = f"- **{category.name}**: {category.description}"
            
            if include_examples and category.examples:
                examples_text = ", ".join([f'"{ex}"' for ex in category.examples[:3]])  # Limit to 3 examples
                desc += f"\n  Examples: {examples_text}"
            
            category_descriptions.append(desc)
        
        categories_text = "\n".join(category_descriptions)
        
        # Build the main prompt
        prompt_parts = [
            "You are an expert text classifier. Your task is to classify the given text into one of the predefined categories.",
            "",
            "**Available Categories:**",
            categories_text,
            "",
            "**Instructions:**",
            "1. Read the text carefully",
            "2. Consider the meaning, context, and tone of the text",
            "3. Select the most appropriate category from the list above",
            "4. Respond with ONLY the category name (exactly as listed above)",
            "5. Do not provide explanations or additional text",
        ]
        
        if custom_instructions:
            prompt_parts.extend([
                "",
                "**Additional Instructions:**",
                custom_instructions
            ])
        
        prompt_parts.extend([
            "",
            "**Text to classify:**",
            f'"{text}"',
            "",
            "**Category:**"
        ])
        
        return "\n".join(prompt_parts)


class StructuredClassificationPrompt(PromptTemplate):
    """Prompt template for structured output classification."""
    
    def generate(
        self,
        text: str,
        categories: CategorySet,
        include_examples: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate a classification prompt for structured output.
        
        Args:
            text: Text to classify
            categories: CategorySet containing available categories
            include_examples: Whether to include category examples in prompt
            custom_instructions: Optional custom instructions to add
            
        Returns:
            Generated prompt string
        """
        # Build category descriptions
        category_descriptions = []
        category_names = []
        
        for category in categories.categories:
            category_names.append(category.name)
            desc = f"- **{category.name}**: {category.description}"
            
            if include_examples and category.examples:
                examples_text = ", ".join([f'"{ex}"' for ex in category.examples[:3]])
                desc += f"\n  Examples: {examples_text}"
            
            category_descriptions.append(desc)
        
        categories_text = "\n".join(category_descriptions)
        valid_categories = ", ".join([f'"{name}"' for name in category_names])
        
        # Build the main prompt
        prompt_parts = [
            "You are an expert text classifier. Classify the given text into one of the predefined categories.",
            "",
            "**Available Categories:**",
            categories_text,
            "",
            "**Instructions:**",
            "1. Analyze the text carefully considering its meaning, context, and tone",
            "2. Select the most appropriate category from the available options",
            "3. Provide your response in the exact JSON format specified below",
            "4. The category must be one of the valid options",
        ]
        
        if custom_instructions:
            prompt_parts.extend([
                "",
                "**Additional Instructions:**",
                custom_instructions
            ])
        
        prompt_parts.extend([
            "",
            "**Text to classify:**",
            f'"{text}"',
            "",
            "**Required JSON Response Format:**",
            "{",
            '  "category": "category_name",',
            '  "confidence": 0.95',
            "}",
            "",
            f"**Valid categories:** {valid_categories}",
            "",
            "**Response:**"
        ])
        
        return "\n".join(prompt_parts)


class VotingClassificationPrompt(PromptTemplate):
    """Prompt template optimized for voting classifier."""
    
    def generate(
        self,
        text: str,
        categories: CategorySet,
        include_examples: bool = True,
        custom_instructions: Optional[str] = None,
        round_number: Optional[int] = None
    ) -> str:
        """
        Generate a classification prompt for voting rounds.
        
        Args:
            text: Text to classify
            categories: CategorySet containing available categories
            include_examples: Whether to include category examples in prompt
            custom_instructions: Optional custom instructions to add
            round_number: Current voting round number
            
        Returns:
            Generated prompt string
        """
        # Build category descriptions
        category_descriptions = []
        for category in categories.categories:
            desc = f"- **{category.name}**: {category.description}"
            
            if include_examples and category.examples:
                examples_text = ", ".join([f'"{ex}"' for ex in category.examples[:2]])  # Fewer examples for voting
                desc += f"\n  Examples: {examples_text}"
            
            category_descriptions.append(desc)
        
        categories_text = "\n".join(category_descriptions)
        
        # Build the main prompt with voting-specific instructions
        prompt_parts = [
            "You are an expert text classifier participating in an ensemble classification system.",
            "Your goal is to provide the most accurate classification possible.",
        ]
        
        if round_number:
            prompt_parts.append(f"This is voting round {round_number}.")
        
        prompt_parts.extend([
            "",
            "**Available Categories:**",
            categories_text,
            "",
            "**Classification Instructions:**",
            "1. Carefully analyze the text's meaning, sentiment, and context",
            "2. Consider subtle nuances and implicit meanings",
            "3. Select the single most appropriate category",
            "4. Be decisive - avoid ambiguity",
            "5. Respond with ONLY the exact category name",
        ])
        
        if custom_instructions:
            prompt_parts.extend([
                "",
                "**Additional Guidelines:**",
                custom_instructions
            ])
        
        prompt_parts.extend([
            "",
            "**Text to classify:**",
            f'"{text}"',
            "",
            "**Your classification:**"
        ])
        
        return "\n".join(prompt_parts)


class FewShotPrompt(PromptTemplate):
    """Prompt template with few-shot examples."""
    
    def __init__(self, examples: Optional[List[tuple]] = None):
        """
        Initialize few-shot prompt template.
        
        Args:
            examples: List of (text, category) tuples for few-shot examples
        """
        self.examples = examples or []
    
    def generate(
        self,
        text: str,
        categories: CategorySet,
        include_examples: bool = True,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Generate a few-shot classification prompt.
        
        Args:
            text: Text to classify
            categories: CategorySet containing available categories
            include_examples: Whether to include category examples in prompt
            custom_instructions: Optional custom instructions to add
            
        Returns:
            Generated prompt string
        """
        # Build category descriptions
        category_descriptions = []
        for category in categories.categories:
            desc = f"- **{category.name}**: {category.description}"
            category_descriptions.append(desc)
        
        categories_text = "\n".join(category_descriptions)
        
        # Build the main prompt
        prompt_parts = [
            "You are an expert text classifier. Classify text into the predefined categories.",
            "",
            "**Available Categories:**",
            categories_text,
        ]
        
        # Add few-shot examples if available
        if self.examples:
            prompt_parts.extend([
                "",
                "**Examples:**"
            ])
            
            for example_text, example_category in self.examples:
                prompt_parts.extend([
                    f'Text: "{example_text}"',
                    f"Category: {example_category}",
                    ""
                ])
        
        prompt_parts.extend([
            "**Instructions:**",
            "1. Analyze the text carefully",
            "2. Select the most appropriate category",
            "3. Respond with only the category name",
        ])
        
        if custom_instructions:
            prompt_parts.extend([
                "",
                "**Additional Instructions:**",
                custom_instructions
            ])
        
        prompt_parts.extend([
            "",
            "**Text to classify:**",
            f'"{text}"',
            "",
            "**Category:**"
        ])
        
        return "\n".join(prompt_parts)


# Default template instance
DEFAULT_TEMPLATE = DefaultClassificationPrompt()
STRUCTURED_TEMPLATE = StructuredClassificationPrompt()
VOTING_TEMPLATE = VotingClassificationPrompt()


def get_template(template_type: str = "default") -> PromptTemplate:
    """
    Get a prompt template by type.
    
    Args:
        template_type: Type of template ("default", "structured", "voting", "few_shot")
        
    Returns:
        PromptTemplate instance
        
    Raises:
        ValueError: If template type is not supported
    """
    templates = {
        "default": DEFAULT_TEMPLATE,
        "structured": STRUCTURED_TEMPLATE,
        "voting": VOTING_TEMPLATE,
        "few_shot": FewShotPrompt(),
    }
    
    if template_type not in templates:
        raise ValueError(f"Unsupported template type: {template_type}. Available: {list(templates.keys())}")
    
    return templates[template_type]
