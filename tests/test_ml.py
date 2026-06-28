"""Tests for ML components."""
from src.data_loader import load_history_csv
from src.ml_model import PerNumberLogisticModel


def test_ml_red():
    df = load_history_csv("data/ssq_history_example.csv")
    model = PerNumberLogisticModel(number_type="red")
    # training will likely be trivial on example data but should not error
    model.fit(df, lookback=1)
    probs = model.predict_proba_next(df)
    assert len(probs) == 33


def test_ml_blue():
    df = load_history_csv("data/ssq_history_example.csv")
    model = PerNumberLogisticModel(number_type="blue")
    model.fit(df, lookback=1)
    probs = model.predict_proba_next(df)
    assert len(probs) == 16
