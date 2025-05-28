# Tagmatic Roadmap 🗺️

This document outlines the planned features and improvements for Tagmatic, organized by priority and development phases.

## 📋 Current Status

**Version**: 0.0.1  
**Status**: Stable text classification with LLM support  
**Core Features**: ✅ Text classification, ✅ Voting classifier, ✅ Batch processing, ✅ Multiple LLM providers

---

##  Phase 1: Image Support 

### 🖼️ Image Classification
- **Priority**: High
- **Status**: Planned
- **Description**: Extend Tagmatic to support image classification using vision-language models

#### Features:
- [ ] **ImageCategory**: New category type with image examples
- [ ] **ImageClassifier**: Dedicated classifier for images
- [ ] **Multi-modal Categories**: Categories that support both text and image examples
- [ ] **Vision-Language Model Support**: Integration with GPT-4V, Claude 3, LLaVA, etc.
- [ ] **Image Preprocessing**: Automatic image resizing, format conversion
- [ ] **Batch Image Processing**: Efficient classification of multiple images

#### Example Usage:
```python
from tagmatic import ImageCategory, ImageCategorySet, ImageClassifier

# Define image categories
categories = ImageCategorySet(categories=[
    ImageCategory(
        name="product_photo",
        description="Professional product photography with clean backgrounds",
        image_examples=["examples/product1.jpg", "examples/product2.jpg"]
    ),
    ImageCategory(
        name="lifestyle_photo", 
        description="Images showing products in real-life contexts",
        image_examples=["examples/lifestyle1.jpg", "examples/lifestyle2.jpg"]
    )
])

# Classify images
classifier = ImageClassifier(llm=vision_llm, categories=categories)
result = classifier.classify_image("path/to/image.jpg")
```

### 🏷️ Image Tagging & Description
- **Priority**: High
- **Status**: Planned
- **Description**: Generate descriptive tags and detailed descriptions for images

#### Features:
- [ ] **Automatic Tagging**: Generate relevant tags based on image content
- [ ] **Description Generation**: Create detailed, natural language descriptions
- [ ] **Custom Tag Vocabularies**: Define specific tag sets for different domains
- [ ] **Multi-language Support**: Generate tags/descriptions in multiple languages
- [ ] **Confidence Scoring**: Reliability scores for generated tags and descriptions

#### Example Usage:
```python
from tagmatic import ImageTagger, ImageDescriber

# Generate tags
tagger = ImageTagger(llm=vision_llm, tag_vocabulary=["outdoor", "indoor", "people", "objects"])
tags = tagger.generate_tags("image.jpg")
# Output: ["outdoor", "nature", "trees", "landscape"]

# Generate descriptions
describer = ImageDescriber(llm=vision_llm, style="detailed")
description = describer.describe("image.jpg")
# Output: "A serene forest landscape with tall pine trees..."
```


## 🚀 Phase 2: Enhanced Text Features

### 📝 Advanced Text Processing
- **Priority**: Medium
- **Status**: Planned

#### Features:
- [ ] **Hierarchical Categories**: Support for nested category structures
- [ ] **Multi-label Classification**: Assign multiple categories to single texts
- [ ] **Confidence Thresholds**: Configurable confidence levels for classification
- [ ] **Active Learning**: Improve categories based on user feedback
- [ ] **Custom Prompt Templates**: User-defined prompt engineering

---


## 🛠️ Technical Improvements

### 🧪 Testing & Quality
- [ ] **Expanded Test Coverage**: Achieve 95%+ test coverage
- [ ] **Performance Benchmarks**: Standardized performance testing
- [ ] **Integration Tests**: End-to-end testing scenarios

### 📚 Documentation & Examples
- [x] **Interactive Tutorials**: Jupyter notebook tutorials
- [ ] **API Documentation**: Comprehensive API reference
- [ ] **Best Practices Guide**: Guidelines for optimal usage

## 💡 How to Contribute

We welcome contributions to help achieve this roadmap! Here's how you can help:

1. **Feature Development**: Pick a feature from this roadmap and contribute code
2. **Testing**: Help improve test coverage and quality
3. **Documentation**: Improve docs, tutorials, and examples
4. **Community**: Help answer questions and support other users
5. **Feedback**: Share your use cases and feature requests
