"""
Basic usage example for Tagmatic.

This example demonstrates how to use Tagmatic for simple text classification
with user-defined categories.
"""

from langchain_openai import ChatOpenAI
from tagmatic import Category, CategorySet, Classifier
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


def main():
    """Demonstrate basic Tagmatic usage."""
    
    # Step 1: Define your categories
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
            examples=["The meeting is at 3 PM", "It's 72 degrees outside", "The report is ready"]
        )
    ])
    
    # Step 2: Initialize your LLM (requires OpenAI API key)
    # You can use any LangChain-compatible LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,  # Low temperature for consistent results
    )
    
    # Step 3: Create a classifier
    classifier = Classifier(
        llm=llm,
        categories=categories
    )
    
    # Step 4: Classify some text
    sample_texts = [
        "I absolutely love this new feature!",
        "This product is broken and doesn't work",
        "The weather forecast shows rain tomorrow",
        "Best purchase I've ever made!",
        "Customer service was unhelpful and rude",
        "The meeting has been rescheduled to Friday"
    ]
    
    print("=== Basic Classification ===")
    for text in sample_texts:
        result = classifier.classify(text)
        print(f"Text: '{text}'")
        print(f"Category: {result.category}")
        if result.confidence:
            print(f"Confidence: {result.confidence:.2f}")
        print()
    
    # Step 5: Batch classification for efficiency
    print("=== Batch Classification ===")
    batch_results = classifier.classify_batch(sample_texts)
    
    for text, result in zip(sample_texts, batch_results):
        print(f"'{text}' -> {result.category}")
    
    # Step 6: Using voting classifier for higher accuracy
    print("\n=== Voting Classification ===")
    important_text = "This is the most incredible thing I've ever experienced!"
    
    # Run classification 5 times and use majority vote
    voting_result = classifier.voting_classify(
        text=important_text,
        voting_rounds=5
    )
    
    print(f"Text: '{important_text}'")
    print(f"Final Category: {voting_result.category}")
    print(f"Confidence: {voting_result.confidence:.2f}")
    print(f"Vote Distribution: {voting_result.vote_counts}")
    print(f"Individual Votes: {voting_result.votes}")
    print(f"Unanimous: {voting_result.is_unanimous}")


if __name__ == "__main__":
    main()
