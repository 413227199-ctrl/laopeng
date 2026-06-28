"""Data loader for SSQ history CSV."""
from pathlib import Path
import pandas as pd

EXPECTED_COLS = [
    "draw_date",
    "draw_id",
    "red1",
    "red2",
    "red3",
    "red4",
    "red5",
    "red6",
    "blue",
]


def load_history_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate SSQ history CSV.

    Returns a DataFrame sorted by draw_date (ascending) with integer columns for numbers.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str)

    # Basic validation
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Keep expected columns only
    df = df[EXPECTED_COLS].copy()

    # Parse types
    df["draw_date"] = pd.to_datetime(df["draw_date"], errors="coerce")
    df["draw_id"] = df["draw_id"].astype(str)

    for c in EXPECTED_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(pd.Int64Dtype())

    # Drop rows with nulls in critical fields
    df = df.dropna(subset=["draw_date", "red1", "blue"])  # at least these must exist

    df = df.sort_values("draw_date").reset_index(drop=True)
    df["period_index"] = range(1, len(df) + 1)
    return df
