"""
Voting classifier example for Tagmatic.

This example demonstrates how to use the voting classifier feature
for improved accuracy in text classification.
"""

from langchain_openai import ChatOpenAI
from tagmatic import Category, CategorySet, Classifier
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


def main():
    """Demonstrate voting classifier usage."""
    
    # Define categories for content moderation
    categories = CategorySet(categories=[
        Category(
            name="safe",
            description="Content that is appropriate, helpful, and follows community guidelines",
            examples=[
                "How do I bake a chocolate cake?",
                "What's the weather like today?",
                "Thank you for your help!"
            ]
        ),
        Category(
            name="spam",
            description="Promotional content, advertisements, or repetitive messages",
            examples=[
                "Buy now! Limited time offer!",
                "Click here to win $1000!",
                "Make money fast with this one trick!"
            ]
        ),
        Category(
            name="inappropriate",
            description="Content that violates community guidelines or is offensive",
            examples=[
                "Hate speech or discriminatory language",
                "Explicit or inappropriate content",
                "Harassment or bullying"
            ]
        ),
        Category(
            name="suspicious",
            description="Content that might be misleading, false, or requires review",
            examples=[
                "Unverified medical claims",
                "Conspiracy theories",
                "Potential misinformation"
            ]
        )
    ])
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
    )
    
    # Create classifier
    classifier = Classifier(
        llm=llm,
        categories=categories
    )
    
    # Test cases with varying difficulty
    test_cases = [
        {
            "text": "How can I improve my cooking skills?",
            "expected": "safe",
            "difficulty": "easy"
        },
        {
            "text": "URGENT! Click now to claim your prize! Limited time only!!!",
            "expected": "spam", 
            "difficulty": "easy"
        },
        {
            "text": "I heard that drinking bleach can cure diseases. Is this true?",
            "expected": "suspicious",
            "difficulty": "medium"
        },
        {
            "text": "This new supplement will cure all your health problems instantly!",
            "expected": "suspicious",  # Could be spam or suspicious
            "difficulty": "hard"
        },
        {
            "text": "Can you recommend a good restaurant in downtown?",
            "expected": "safe",
            "difficulty": "easy"
        }
    ]
    
    print("=== Comparing Single vs Voting Classification ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case["text"]
        expected = test_case["expected"]
        difficulty = test_case["difficulty"]
        
        print(f"Test Case {i} ({difficulty} difficulty):")
        print(f"Text: '{text}'")
        print(f"Expected: {expected}")
        print()
        
        # Single classification
        single_result = classifier.classify(text)
        print(f"Single Classification: {single_result.category}")
        
        # Voting classification with 3 rounds
        voting_result_3 = classifier.voting_classify(text, voting_rounds=3)
        print(f"Voting (3 rounds): {voting_result_3.category} "
              f"(confidence: {voting_result_3.confidence:.2f})")
        print(f"  Votes: {voting_result_3.votes}")
        
        # Voting classification with 5 rounds for higher confidence
        voting_result_5 = classifier.voting_classify(text, voting_rounds=5)
        print(f"Voting (5 rounds): {voting_result_5.category} "
              f"(confidence: {voting_result_5.confidence:.2f})")
        print(f"  Vote counts: {voting_result_5.vote_counts}")
        print(f"  Unanimous: {voting_result_5.is_unanimous}")
        
        # Check if voting improved accuracy
        single_correct = single_result.category == expected
        voting_correct = voting_result_5.category == expected
        
        if voting_correct and not single_correct:
            print("  ✅ Voting classifier corrected the result!")
        elif single_correct and voting_correct:
            print("  ✅ Both methods got it right")
        elif not single_correct and not voting_correct:
            print("  ❌ Both methods missed the expected result")
        else:
            print("  ⚠️  Single classification was better")
        
        print("-" * 60)
        print()
    
    # Demonstrate confidence analysis
    print("=== Confidence Analysis ===\n")
    
    ambiguous_text = "This product is okay, I guess. Not great but not terrible either."
    
    print(f"Analyzing ambiguous text: '{ambiguous_text}'")
    print()
    
    # Run multiple voting rounds to see consistency
    for rounds in [3, 5, 7]:
        result = classifier.voting_classify(ambiguous_text, voting_rounds=rounds)
        print(f"Voting with {rounds} rounds:")
        print(f"  Result: {result.category}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Vote distribution: {result.vote_counts}")
        print(f"  Consistency: {'High' if result.confidence >= 0.8 else 'Medium' if result.confidence >= 0.6 else 'Low'}")
        print()
    
    print("=== Best Practices for Voting Classifier ===")
    print("1. Use odd numbers of voting rounds (3, 5, 7) to avoid ties")
    print("2. More rounds = higher confidence but slower performance")
    print("3. Use voting for critical decisions or ambiguous content")
    print("4. Monitor confidence scores - low confidence may indicate:")
    print("   - Ambiguous text that doesn't fit categories well")
    print("   - Need for additional or refined categories")
    print("   - Inconsistent LLM responses")


if __name__ == "__main__":
    main()
