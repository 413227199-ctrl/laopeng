"""Per-number logistic model for SSQ.

Trains a single logistic regression on per-number dataset (rows: draw_index x number).
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_per_number_dataset


class PerNumberLogisticModel:
    def __init__(self, number_type: str = "red"):
        self.number_type = number_type
        self.model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))])
        self.trained = False

    def fit(self, df: pd.DataFrame, lookback: int = 50):
        X, y = build_per_number_dataset(df, lookback=lookback, number_type=self.number_type)
        # use feature columns
        feat_cols = ["freq_10", "freq_50", "global_freq", "last_seen_gap"]
        Xf = X[feat_cols].astype(float).fillna(0)
        self.model.fit(Xf, y.values)
        self.trained = True

    def predict_proba_next(self, df: pd.DataFrame) -> pd.Series:
        """Compute probability for each candidate number to appear in next draw.

        Returns a Series indexed by number -> probability
        """
        if not self.trained:
            raise RuntimeError("Model not trained")
        # build features for the latest draw index = len(df)-1
        i = len(df) - 1
        numbers = list(range(1, 34)) if self.number_type == "red" else list(range(1, 17))
        rows = []
        for num in numbers:
            # compute features against history upto i
            freq_10 = 0
            freq_50 = 0
            global_freq = 0
            # reuse build_per_number_dataset by computing one-sample features
            # but for simplicity, compute counts directly
            hist = df.iloc[:i]
            if not hist.empty:
                if self.number_type == "red":
                    freq_10 = int(df.iloc[max(0, i - 10):i][[f"red{j}" for j in range(1, 7)]].isin([num]).any(axis=1).sum())
                    freq_50 = int(df.iloc[max(0, i - 50):i][[f"red{j}" for j in range(1, 7)]].isin([num]).any(axis=1).sum())
                    global_freq = int(hist[[f"red{j}" for j in range(1, 7)]].isin([num]).any(axis=1).sum())
                else:
                    freq_10 = int(df.iloc[max(0, i - 10):i]["blue"].isin([num]).sum())
                    freq_50 = int(df.iloc[max(0, i - 50):i]["blue"].isin([num]).sum())
                    global_freq = int(hist["blue"].isin([num]).sum())
            last_seen_gap = i + 1
            # compute last seen gap
            from .features import last_seen_gap as lsg

            last_seen_gap = lsg(df, num, upto_index=i)

            rows.append({"freq_10": freq_10, "freq_50": freq_50, "global_freq": global_freq, "last_seen_gap": last_seen_gap})

        Xn = pd.DataFrame(rows)[["freq_10", "freq_50", "global_freq", "last_seen_gap"]].astype(float).fillna(0)
        probs = self.model.predict_proba(Xn)[:, 1]
        return pd.Series(probs, index=numbers)

    def predict_topk(self, df: pd.DataFrame, k: int = 6) -> List[int]:
        probs = self.predict_proba_next(df)
        top = list(probs.sort_values(ascending=False).head(k).index.astype(int))
        return top
