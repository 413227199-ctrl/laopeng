"""Simple baseline frequency model for SSQ."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class FrequencyModel:
    """Model that uses historical frequency to assign appearance probabilities.

    For reds (1..33) and blues (1..16).
    """

    def __init__(self, red_range=33, blue_range=16, smoothing=1.0):
        self.red_range = red_range
        self.blue_range = blue_range
        self.smoothing = smoothing
        self.red_prob = None
        self.blue_prob = None

    def fit(self, df: pd.DataFrame):
        from .features import global_number_frequency

        counts = global_number_frequency(df)
        red_counts = counts["red"].astype(float)
        blue_counts = counts["blue"].astype(float)

        # Add simple Laplace smoothing and normalize by number of draws * 6 (reds)
        draws = len(df)
        red_total_possible = draws * 6
        blue_total_possible = draws * 1

        red_counts = red_counts + self.smoothing
        blue_counts = blue_counts + self.smoothing

        self.red_prob = red_counts / red_counts.sum()
        self.blue_prob = blue_counts / blue_counts.sum()

    def predict_topk(self, k_red=6, k_blue=1) -> Tuple[List[int], List[int]]:
        if self.red_prob is None or self.blue_prob is None:
            raise RuntimeError("Model not fitted")
        red_top = list(self.red_prob.sort_values(ascending=False).head(k_red).index.astype(int))
        blue_top = list(self.blue_prob.sort_values(ascending=False).head(k_blue).index.astype(int))
        return red_top, blue_top

    def sample_combination(self, n_samples=1, k_red=6) -> List[Tuple[List[int], int]]:
        """Sample combinations by sampling reds without replacement according to probs.

        Returns list of tuples (reds_list, blue)
        """
        rng = np.random.default_rng()
        reds = []
        for _ in range(n_samples):
            red_choices = rng.choice(
                a=np.arange(1, self.red_range + 1),
                size=k_red,
                replace=False,
                p=self.red_prob.values,
            )
            blue_choice = int(rng.choice(a=np.arange(1, self.blue_range + 1), p=self.blue_prob.values))
            reds.append((sorted(list(red_choices)), blue_choice))
        return reds
