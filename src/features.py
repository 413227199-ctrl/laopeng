"""Feature helpers for SSQ project."""
from typing import Dict
import pandas as pd


def global_number_frequency(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute global frequencies for red numbers (1..33) and blue (1..16).

    Returns dict with keys 'red' and 'blue' mapping to Series indexed by number.
    """
    red_cols = [f"red{i}" for i in range(1, 7)]
    reds = pd.concat([df[c] for c in red_cols], axis=0)
    red_counts = reds.value_counts().sort_index()
    # ensure full range 1..33
    red_index = range(1, 34)
    red_counts = red_counts.reindex(red_index, fill_value=0).astype(int)

    blue_counts = df["blue"].value_counts().sort_index()
    blue_index = range(1, 17)
    blue_counts = blue_counts.reindex(blue_index, fill_value=0).astype(int)

    return {"red": red_counts, "blue": blue_counts}


def window_frequency(df: pd.DataFrame, window: int = 50) -> Dict[str, pd.Series]:
    """Compute frequencies over the last `window` draws."""
    if window <= 0:
        raise ValueError("window must be > 0")
    tail = df.tail(window)
    return global_number_frequency(tail)
