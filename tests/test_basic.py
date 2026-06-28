"""Basic tests for data loader and frequency model."""
from src.data_loader import load_history_csv
from src.models import FrequencyModel


def test_load_example():
    df = load_history_csv("data/ssq_history_example.csv")
    assert len(df) >= 1


def test_frequency_model():
    df = load_history_csv("data/ssq_history_example.csv")
    m = FrequencyModel()
    m.fit(df)
    reds, blues = m.predict_topk()
    assert len(reds) == 6
    assert len(blues) == 1
