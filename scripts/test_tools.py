import os
import sys

# Ensure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import load_dataset_df
from agent.tools import list_categories, list_intents, count_rows, get_distribution, get_examples, summarize_category

def main():
    # 1. Load the dataset
    print("Loading dataset...")
    df = load_dataset_df()
    print(f"Loaded {len(df)} rows.")

    # 2. Call each of the 6 tools directly
    print("\n--- Categories ---")
    print(list_categories.invoke({"category": None}))

    print("\n--- Intents for REFUND ---")
    print(list_intents.invoke({"category": "REFUND"}))

    print("\n--- Count rows for get_refund ---")
    print(count_rows.invoke({"intent": "get_refund"}))

    print("\n--- Distribution for ACCOUNT ---")
    print(get_distribution.invoke({"category": "ACCOUNT"}))

    print("\n--- Examples for SHIPPING ---")
    examples = get_examples.invoke({"category": "SHIPPING", "n": 3})
    for i, ex in enumerate(examples):
        print(f"Example {i+1}:")
        print(f"  Intent: {ex['intent']}")
        print(f"  Category: {ex['category']}")
        print(f"  Instruction: {ex['instruction']}")
        print(f"  Response: {ex['response']}")

    print("\n--- Summarize FEEDBACK ---")
    print(summarize_category.invoke({"category": "FEEDBACK"}))

if __name__ == "__main__":
    main()
