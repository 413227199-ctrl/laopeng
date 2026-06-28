# 双色球预测 (ssq-mvp)

这是一个双色球（中国体育彩票）预测研究项目的最小可行实现（MVP）。

目标：提供可复现的数据加载、基线频率模型、时间序列回测与一个简单的 Streamlit 界面，以便后续扩展更复杂的模型与特征。

主要内容：

- data/ssq_history.csv: 示例历史开奖数据（请替换为完整版历史数据）
- src/data_loader.py: 数据加载与预处理
- src/models.py: 频率基线模型与概率输出
- src/backtest.py: 简单的时间序列回测（滚动训练/预测）
- src/app.py: Streamlit 演示界面（预测、统计、回测摘要）
- requirements.txt: 依赖

快速开始：

1. 克隆仓库并安装依赖（推荐使用虚拟环境）

   pip install -r requirements.txt

2. 启动演示界面：

   streamlit run src/app.py

3. 替换 data/ssq_history.csv 为完整的历史开奖数据（CSV 列：draw_date,draw_id,red1..red6,blue）

后续计划：
- 添加更多特征（遗漏、窗口频率、和值、三区分布等）
- 引入多标签分类模型（XGBoost/LightGBM）并对比回测结果
- 增加概率采样与下注收益模拟

注意：彩票本质为随机事件，本项目仅作研究和学习用途，不构成任何投资/购彩建议。
