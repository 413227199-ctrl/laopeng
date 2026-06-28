"""Feature helpers for SSQ project.

Extended with omission (last-seen gap) and dataset builders for per-number ML.
"""
from typing import Dict, List, Tuple
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


def last_seen_gap(df: pd.DataFrame, number: int, upto_index: int) -> int:
    """Return how many draws ago the `number` last appeared before `upto_index` (exclusive).

    If never appeared, returns a large value (upto_index + 1).
    """
    red_cols = [f"red{i}" for i in range(1, 7)]
    # search rows strictly before upto_index
    sub = df.iloc[:upto_index]
    if sub.empty:
        return upto_index + 1

    # combine reds and blue depending on number range
    if 1 <= number <= 33:
        appeared = (sub[red_cols] == number).any(axis=1)
    else:
        # treat as blue
        appeared = (sub["blue"] == number)

    if not appeared.any():
        return upto_index + 1

    last_pos = appeared[::-1].idxmax()
    # compute gap as number of draws since last occurrence
    gap = upto_index - last_pos
    return int(gap)


def build_per_number_dataset(df: pd.DataFrame, lookback: int = 50, number_type: str = "red") -> Tuple[pd.DataFrame, pd.Series]:
    """Build a per-number dataset suitable for per-number binary classification.

    Each row corresponds to (draw_index, number). Features are computed from history up to that draw_index (exclusive).
    The label indicates whether the number appears in the NEXT draw (draw_index + 1).

    number_type: 'red' (1..33) or 'blue' (1..16)

    Returns:
      X: DataFrame of features
      y: Series of binary labels (0/1)
    """
    if number_type == "red":
        numbers = list(range(1, 34))
        red_cols = [f"red{i}" for i in range(1, 7)]
    elif number_type == "blue":
        numbers = list(range(1, 17))
        red_cols = []
    else:
        raise ValueError("number_type must be 'red' or 'blue'")

    rows = []
    labels = []

    # we need to produce features for draws where next draw exists
    for i in range(lookback, len(df) - 1):
        # history is df[:i]
        hist = df.iloc[:i]
        next_draw = df.iloc[i + 1]

        # precompute frequency windows
        freq_10 = global_number_frequency(df.iloc[max(0, i - 10):i])["red"] if i >= 1 else pd.Series(0)
        freq_50 = global_number_frequency(df.iloc[max(0, i - 50):i])["red"] if i >= 1 else pd.Series(0)
        global_freq = global_number_frequency(hist)["red"]

        for num in numbers:
            # features
            if number_type == "red":
                f_freq_10 = int(freq_10.get(num, 0)) if not freq_10.empty else 0
                f_freq_50 = int(freq_50.get(num, 0)) if not freq_50.empty else 0
                f_global = int(global_freq.get(num, 0)) if not global_freq.empty else 0
            else:
                # for blue, compute from blue column counts
                blue_counts_10 = df.iloc[max(0, i - 10):i]["blue"].value_counts()
                blue_counts_50 = df.iloc[max(0, i - 50):i]["blue"].value_counts()
                blue_global = hist["blue"].value_counts()
                f_freq_10 = int(blue_counts_10.get(num, 0))
                f_freq_50 = int(blue_counts_50.get(num, 0))
                f_global = int(blue_global.get(num, 0))

            f_last_seen = last_seen_gap(df, num, upto_index=i)

            row = {
                "draw_index": i,
                "number": num,
                "freq_10": f_freq_10,
                "freq_50": f_freq_50,
                "global_freq": f_global,
                "last_seen_gap": f_last_seen,
            }
            # label: whether num appears in next_draw
            if number_type == "red":
                appeared_next = int(num in next_draw[[f"red{j}" for j in range(1, 7)]].values)
            else:
                appeared_next = int(next_draw["blue"] == num)

            rows.append(row)
            labels.append(appeared_next)

    X = pd.DataFrame(rows)
    y = pd.Series(labels)
    return X, y
