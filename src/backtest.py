"""Simple rolling backtest for SSQ prediction models."""
from typing import Callable, Dict
import pandas as pd
from collections import Counter


def rolling_backtest(df: pd.DataFrame, model_cls, start_train: int = 100, step: int = 1) -> Dict:
    """Perform a rolling backtest.

    For each step i in range(start_train, len(df)-1): train on df[:i], predict next, record hits.
    Returns summary statistics.
    """
    if len(df) < start_train + 1:
        raise ValueError("Not enough data for backtest; increase data or reduce start_train")

    results = []

    for i in range(start_train, len(df)):
        train = df.iloc[:i]
        test = df.iloc[i : i + 1]

        model = model_cls()
        model.fit(train)
        pred_reds, pred_blue = model.predict_topk()

        actual_reds = set(test[[f"red{j}" for j in range(1, 7)]].values.flatten().astype(int))
        actual_blue = int(test["blue"].iloc[0])

        red_hits = len(set(pred_reds) & actual_reds)
        blue_hit = int(pred_blue[0] == actual_blue)

        results.append({"index": i, "red_hits": red_hits, "blue_hit": blue_hit})

    df_res = pd.DataFrame(results)
    summary = {
        "total_rounds": len(df_res),
        "mean_red_hits": float(df_res["red_hits"].mean()),
        "blue_hit_rate": float(df_res["blue_hit"].mean()),
        "red_hits_distribution": df_res["red_hits"].value_counts().sort_index().to_dict(),
    }
    return summary


if __name__ == "__main__":
    # Quick demo when run as script
    import sys
    from .data_loader import load_history_csv
    from .models import FrequencyModel

    path = sys.argv[1] if len(sys.argv) > 1 else "data/ssq_history_example.csv"
    df = load_history_csv(path)
    # if not enough rows, reduce start_train
    start = min(3, max(1, len(df) - 1))
    print("Running demo backtest on example data...\n")
    print(rolling_backtest(df, FrequencyModel, start_train=start))
