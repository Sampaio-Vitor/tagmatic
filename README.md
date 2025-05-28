# Tagmatic - Generic Text Classification Library
<p align="center">
    <img src="images/readme.png" alt="Tagmatic Overview" width="600"/>
</p>
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tagmatic is a flexible, user-defined text classification library that leverages Large Language Models (LLMs) to classify text into custom categories. Simply define your categories with descriptions, and Tagmatic handles all the complexity of prompt engineering and LLM interaction.

## Key Features

- **User-Defined Categories**: Define categories with simple names and descriptions
- **LLM Provider Agnostic**: Works with any LangChain-compatible LLM
- **Voting Classifier**: Run multiple classifications and use majority voting for improved accuracy
- **Batch Processing**: Efficiently classify multiple texts at once
- **Type Safety**: Full type hints throughout the codebase
- **Easy Integration**: Simple API that requires minimal setup

## 📦 Installation

```bash
pip install tagmatic
```


## 🎯 Quick Start

```python
from langchain_openai import ChatOpenAI
from tagmatic import Category, CategorySet, Classifier

# 1. Define your categories
categories = CategorySet(categories=[
    Category(
        name="positive",
        description="Text expressing positive emotions, satisfaction, or happiness",
        examples=["I love this!", "Great job!", "This is amazing!"]
    ),
    Category(
        name="negative", 
        description="Text expressing negative emotions, complaints, or dissatisfaction",
        examples=["I hate this", "This is terrible", "Very disappointed"]
    ),
    Category(
        name="neutral",
        description="Text that is factual or doesn't express strong emotions",
        examples=["The meeting is at 3 PM", "It's 72 degrees outside"]
    )
])

# 2. Initialize your LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 3. Create a classifier
classifier = Classifier(llm=llm, categories=categories)

# 4. Classify text
result = classifier.classify("I absolutely love this new feature!")
print(f"Category: {result.category}")  # Output: positive

# 5. Use voting for higher accuracy
voting_result = classifier.voting_classify(
    "This is amazing!", 
    voting_rounds=5
)
print(f"Category: {voting_result.category}")
print(f"Confidence: {voting_result.confidence}")
print(f"Vote distribution: {voting_result.vote_counts}")
```

## 🔧 Core Components

### Categories

Categories are the foundation of Tagmatic. Each category consists of:

- **Name**: A unique identifier for the category
- **Description**: A clear description of what text belongs in this category
- **Examples** (optional): Sample texts that belong to this category

```python
from tagmatic import Category

category = Category(
    name="spam",
    description="Promotional content, advertisements, or repetitive messages",
    examples=[
        "Buy now! Limited time offer!",
        "Click here to win $1000!",
        "Make money fast with this one trick!"
    ]
)
```

### CategorySet

A CategorySet manages a collection of categories and ensures they have unique names:

```python
from tagmatic import CategorySet

categories = CategorySet(categories=[
    Category(name="urgent", description="Messages requiring immediate attention"),
    Category(name="normal", description="Regular messages that can be processed normally"),
    Category(name="low_priority", description="Messages that can be handled later")
])

# Access categories
print(categories.get_category_names())  # ['urgent', 'normal', 'low_priority']
urgent_cat = categories.get_category("urgent")

# Save/load categories
categories.save_to_file("my_categories.json")
loaded_categories = CategorySet.from_file("my_categories.json")
```

### Classifier

The main interface for text classification:

```python
from tagmatic import Classifier

# Basic classifier
classifier = Classifier(llm=llm, categories=categories)

# With structured output (includes confidence scores)
classifier = Classifier(
    llm=llm, 
    categories=categories,
    use_structured_output=True
)

# Single classification
result = classifier.classify("Hello world!")

# Batch classification
texts = ["Text 1", "Text 2", "Text 3"]
results = classifier.classify_batch(texts)

# Voting classification
voting_result = classifier.voting_classify(
    text="Ambiguous text here",
    voting_rounds=5
)
```

## 🗳️ Voting Classifier

The voting classifier is Tagmatic's "special sauce" - it runs the same classification multiple times and uses majority voting to improve accuracy:

```python
# Run classification 5 times and use majority vote
result = classifier.voting_classify(
    text="This product is okay, I guess.",
    voting_rounds=5
)

print(f"Final category: {result.category}")
print(f"Confidence: {result.confidence}")  # Based on vote distribution
print(f"Individual votes: {result.votes}")  # ['neutral', 'positive', 'neutral', 'neutral', 'positive']
print(f"Vote counts: {result.vote_counts}")  # {'neutral': 3, 'positive': 2}
print(f"Unanimous: {result.is_unanimous}")  # False
```

**Benefits of Voting:**
- Reduces classification errors
- Provides confidence scoring
- Handles edge cases better
- Identifies ambiguous content

**Best Practices:**
- Use odd numbers (3, 5, 7) to avoid ties
- More rounds = higher confidence but slower performance
- Monitor confidence scores to identify problematic content

## 🎨 Use Cases

### Sentiment Analysis
```python
sentiment_categories = CategorySet(categories=[
    Category("positive", "Positive emotions, satisfaction, happiness"),
    Category("negative", "Negative emotions, complaints, dissatisfaction"), 
    Category("neutral", "Factual or emotionally neutral content")
])
```

### Content Moderation
```python
moderation_categories = CategorySet(categories=[
    Category("safe", "Appropriate content following community guidelines"),
    Category("spam", "Promotional or repetitive content"),
    Category("inappropriate", "Content violating community guidelines"),
    Category("suspicious", "Potentially misleading or false information")
])
```

### Customer Support
```python
support_categories = CategorySet(categories=[
    Category("technical_issue", "Problems with product functionality"),
    Category("billing_question", "Questions about payments, invoices, or pricing"),
    Category("feature_request", "Suggestions for new features or improvements"),
    Category("general_inquiry", "General questions about products or services")
])
```

### Topic Classification
```python
topic_categories = CategorySet(categories=[
    Category("technology", "Content about software, hardware, or digital topics"),
    Category("sports", "Content about athletics, games, or competitions"),
    Category("politics", "Content about government, elections, or policy"),
    Category("entertainment", "Content about movies, music, or celebrities")
])
```

## 🔌 LLM Provider Support

Tagmatic works with any LangChain-compatible LLM:

### OpenAI
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # or "gpt-4", "gpt-4-turbo"
    temperature=0.1,
    api_key="your-api-key"
)
```

### Anthropic
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.1,
    api_key="your-api-key"
)
```

### Local Models
```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
```

## 📊 Performance Tips

1. **Use Batch Classification**: More efficient for multiple texts
   ```python
   results = classifier.classify_batch(texts)  # Better than individual calls
   ```

2. **Optimize Temperature**: Lower values (0.1-0.3) for consistent results
   ```python
   llm = ChatOpenAI(temperature=0.1)  # More deterministic
   ```

3. **Cache Categories**: Reuse CategorySet objects
   ```python
   categories.save_to_file("categories.json")  # Save for reuse
   ```

4. **Monitor Confidence**: Use voting for low-confidence cases
   ```python
   result = classifier.classify(text)
   if not result.confidence or result.confidence < 0.8:
       # Use voting for uncertain cases
       result = classifier.voting_classify(text, voting_rounds=3)
   ```

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install tagmatic

# Run tests
pytest

# Run tests with coverage
pytest --cov=tagmatic --cov-report=html
```

## 📚 Examples

Check out the `examples/` directory for more detailed examples:

- `basic_usage.py` - Simple classification examples
- `complaints_classification_demo.py` - Demonstration of a real world example where Tagmatic thrive. Advanced voting classifier usage

## 🤝 Contributing

We welcome contributions! 

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

This project was inspired by real-world experience in data processing and classification at scale. Special thanks to the LangChain community for providing the foundation that makes LLM provider agnosticism possible.

---

**Made with ❤️ by Vitor Sampaio**
