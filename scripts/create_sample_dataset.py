import polars as pl
import json

def create_sample_dataset():
    """Create a sample dataset with 10 examples from 5 different categories"""
    
    # Read the full dataset
    df = pl.read_csv("examples/rows.csv")
    
    # Filter for rows with both Sub-issue and Consumer complaint narrative
    df_filtered = df.filter(
        (pl.col("Sub-issue").is_not_null()) & 
        (pl.col("Consumer complaint narrative").is_not_null()) &
        (pl.col("Consumer complaint narrative").str.len_chars() > 50)  # Ensure meaningful text
    )
    
    # Select 5 categories with good representation
    selected_categories = [
        "Account information incorrect",
        "Debt is not yours", 
        "Account status incorrect",
        "Information belongs to someone else",
        "Privacy issues"
    ]
    
    sample_data = []
    
    for category in selected_categories:
        # Get 10 examples from each category
        category_samples = df_filtered.filter(
            pl.col("Sub-issue") == category
        ).sample(n=10, seed=42).select([
            "Sub-issue", 
            "Consumer complaint narrative"
        ])
        
        for row in category_samples.iter_rows(named=True):
            sample_data.append({
                "category": row["Sub-issue"],
                "text": row["Consumer complaint narrative"]
            })
    
    return sample_data

if __name__ == "__main__":
    sample_data = create_sample_dataset()
    
    # Save to JSON file
    with open("examples/sample_complaints.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample dataset with {len(sample_data)} examples")
    print("Categories included:")
    categories = {}
    for item in sample_data:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  - {cat}: {count} examples")
