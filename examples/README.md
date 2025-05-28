# Tagmatic Examples

This directory contains examples demonstrating how to use the Tagmatic library for various text classification tasks.

## Files

### `complaints_classification_demo.ipynb`
**Main Demo Notebook** - A comprehensive demonstration using real-world consumer complaint data from the CFPB database. This notebook shows:

- How to define categories using business-friendly descriptions
- Classification of 50 real consumer complaints across 5 categories
- Performance evaluation with accuracy metrics and confusion matrix
- Voting classifier demonstration for improved accuracy
- Real-world application example with automatic routing
- Complete end-to-end workflow

**Categories demonstrated:**
- Account Information Errors
- Debt Not Owed
- Account Status Wrong
- Identity Mix-up/Fraud
- Debt Already Paid

### `sample_complaints.json`
Sample dataset containing 50 consumer complaints (10 from each category) extracted from the original CFPB dataset.

### `basic_usage.py`
Simple example showing basic Tagmatic usage for sentiment analysis.

### `voting_classifier.py`
Example demonstrating the voting classifier feature for improved accuracy.

### `tutorial.ipynb`
Step-by-step tutorial covering all Tagmatic features.

### Legacy Files
- `complaints_example.ipynb` - Original simple example (replaced by the comprehensive demo)
- `rows.csv` - Full CFPB dataset (large file)

## Getting Started

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Set up your API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the main demo:**
   Open `complaints_classification_demo.ipynb` in Jupyter and run all cells.

## Key Features Demonstrated

- ✅ **Zero Training Data Required** - Define categories with simple descriptions
- ✅ **High Accuracy** - Achieves excellent results on real-world data
- ✅ **Voting Classifier** - Multiple rounds with majority voting for better accuracy
- ✅ **LLM Agnostic** - Works with any LangChain-compatible LLM
- ✅ **Production Ready** - Includes confidence scores and error handling
- ✅ **Business Applications** - Shows real customer service routing example

## Real-World Applications

The examples demonstrate how Tagmatic can be used for:

- **Customer Service** - Automatic ticket routing and prioritization
- **Content Moderation** - Classify user-generated content
- **Document Processing** - Organize and categorize documents
- **Survey Analysis** - Categorize open-ended survey responses
- **Email Classification** - Route emails to appropriate departments
- **Compliance** - Classify regulatory documents and reports

## Performance

The consumer complaints demo typically achieves:
- **85-95% accuracy** on the 5-category classification task
- **High confidence scores** (>0.8) for most predictions
- **Fast processing** - ~1-2 seconds per classification
- **Robust handling** of real-world messy text data

## Next Steps

After running the examples:

1. Try modifying the category descriptions to see how it affects performance
2. Add new categories for your specific use case
3. Test with your own text data
4. Experiment with different LLM providers
5. Implement the voting classifier for production use

For more information, see the main [README.md](../README.md).
