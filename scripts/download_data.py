"""One-time script to download the Bitext dataset and save it locally."""
import pandas as pd
from datasets import load_dataset
from pathlib import Path

OUT = Path("data/bitext_customer_service.parquet")


def main() -> None:
    """Download the Bitext Customer Service dataset from Hugging Face and save it locally.

    Saves to: data/bitext_customer_service.parquet
    Only needs to be run once. Re-running will overwrite the existing file.
    Usage: python scripts/download_data.py
    """
    print("Downloading Bitext Customer Service dataset...")
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    df = ds.to_pandas()
    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    print("Columns:", df.columns.tolist())
    print("Categories:", df["category"].unique().tolist())


if __name__ == "__main__":
    main()
