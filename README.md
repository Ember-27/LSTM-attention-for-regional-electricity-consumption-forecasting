# LSTM-attention-for-regional-electricity-consumption-forecasting

一个基于深度学习的用电量预测项目，使用 **LSTM** 和 **LSTM+Attention** 模型，根据历史用电数据和气象特征预测未来1小时的用电量。
---
## 项目简介

本项目旨在构建和对比两种深度学习模型（LSTM基线模型 vs LSTM+Attention模型），用于预测电力消耗量。项目包含完整的数据处理、可视化、模型训练与评估流程。

### 核心功能
-  **数据可视化**：生成6种可视化图表，探索数据规律
-  **数据预处理**：自动清洗、归一化、构建时间序列样本
-  **深度学习模型**：LSTM基线模型 + LSTM+Attention模型
-  **模型评估**：MAE、RMSE、MAPE、R² 多指标对比
-  **可解释性**：Attention权重可视化，观察模型关注的时间步
---
##  数据说明

### 输入数据
- **文件名**：`powerconsumption.csv`
- **时间粒度**：原始数据为10秒级，聚合为小时级
- **特征说明**：

| 特征名 | 说明 |
|--------|------|
| Datetime | 时间戳 |
| Temperature | 温度 (°C) |
| Humidity | 湿度 (%) |
| WindSpeed | 风速 |
| GeneralDiffuseFlows | 总漫射流量 |
| DiffuseFlows | 漫射流量 |
| PowerConsumption_Zone1 | 区域1用电量 |
| PowerConsumption_Zone2 | 区域2用电量 |
| PowerConsumption_Zone3 | 区域3用电量 |

### 预测目标
- **TotalPower** = Zone1 + Zone2 + Zone3（总用电量）
- **预测范围**：未来1小时

---

## 🚀 安装依赖

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
