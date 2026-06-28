"""Streamlit demo app for SSQ MVP.

Extended with ML training button and improved UI sections.
"""
import streamlit as st
import pandas as pd
from pathlib import Path

from src.data_loader import load_history_csv
from src.models import FrequencyModel
from src.features import global_number_frequency, window_frequency
from src.ml_model import PerNumberLogisticModel

st.set_page_config(page_title="SSQ MVP", layout="wide")

st.title("双色球预测 — MVP")

DATA_PATH = Path("data/ssq_history_example.csv")

st.sidebar.header("设置")
use_example = st.sidebar.checkbox("使用示例数据 (repo/data)", value=True)
window = st.sidebar.slider("最近 N 期窗口（用于窗口频率）", min_value=5, max_value=200, value=50, step=5)

if use_example:
    df = load_history_csv(DATA_PATH)
else:
    upload = st.sidebar.file_uploader("上传历史开奖 CSV", type=["csv"])
    if upload is not None:
        df = load_history_csv(upload)
    else:
        st.warning("请上传 CSV 或启用示例数据")
        st.stop()

st.header("历史统计")
col1, col2 = st.columns([2, 1])
with col1:
    st.write("最近 10 条记录")
    st.dataframe(df.tail(10))
with col2:
    counts = global_number_frequency(df)
    st.write("红球总出现次数 (示例)")
    st.bar_chart(counts["red"]) if not counts["red"].empty else st.write("无数据")

st.header("基线频率预测")
model = FrequencyModel()
model.fit(df)
red_top, blue_top = model.predict_topk()
st.write("按历史频率选出的红球 (6) 与蓝球 (1):")
st.write(red_top, blue_top)

st.subheader("采样若干候选组合")
n_samples = st.number_input("生成候选数量", min_value=1, max_value=100, value=10)
samples = model.sample_combination(n_samples=n_samples)
for i, (reds, blue) in enumerate(samples, 1):
    st.write(f"候选 {i}: 红球={reds} | 蓝球={blue}")

st.header("基于机器学习的单号码概率预测")
st.write("训练一个按号码训练的 LogisticRegression 模型，输出每个号码的出现概率（下一期）。")
col3, col4 = st.columns(2)
with col3:
    if st.button("训练并预测（红球）"):
        with st.spinner("训练中..."):
            mlr = PerNumberLogisticModel(number_type="red")
            try:
                mlr.fit(df, lookback=50)
                probs = mlr.predict_proba_next(df)
                st.write("红球出现概率（下一期）")
                st.dataframe(probs)
                st.write("按概率选出的红球(6):", mlr.predict_topk(df, k=6))
            except Exception as e:
                st.error(f"训练失败: {e}")
with col4:
    if st.button("训练并预测（蓝球）"):
        with st.spinner("训练中..."):
            mlb = PerNumberLogisticModel(number_type="blue")
            try:
                mlb.fit(df, lookback=200)
                probs_b = mlb.predict_proba_next(df)
                st.write("蓝球出现概率（下一期）")
                st.dataframe(probs_b)
                st.write("按概率选出的蓝球(1):", mlb.predict_topk(df, k=1))
            except Exception as e:
                st.error(f"训练失败: {e}")

st.header("回测（滚动时间序列）")
if st.button("运行回测（示例）"):
    with st.spinner("运行中..."):
        from src.backtest import rolling_backtest

        start_train = min(50, max(3, len(df) - 1))
        summary = rolling_backtest(df, FrequencyModel, start_train=start_train)
        st.write(summary)

st.markdown("---")
st.caption("注意：本项目仅作研究用途，彩票具有高度随机性。")
