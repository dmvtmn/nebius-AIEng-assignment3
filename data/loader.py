"""Dataset loading utilities for the Bitext Customer Service dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/bitext_customer_service.parquet")


def load_dataset_df() -> pd.DataFrame:
    """Load the Bitext dataset from a local parquet file.

    The parquet file is created once by scripts/download_data.py.
    Raises FileNotFoundError if the file does not exist — run the
    download script first.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run: python scripts/download_data.py"
        )
    return pd.read_parquet(DATA_PATH)
