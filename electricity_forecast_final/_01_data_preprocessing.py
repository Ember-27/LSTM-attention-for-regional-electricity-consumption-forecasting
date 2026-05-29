"""
数据预处理和序列构建
功能：
1. 加载清洗后的数据
2. 添加总用电量特征
3. 选择特征
4. 数据归一化
5. 构建序列数据（使用过去24小时的数据预测未来1小时）
6. 按时间顺序划分训练集(70%)、验证集(15%)、测试集(15%)
7. 保存预处理后的数据
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 加载清洗后的数据
# ==========================================
print("="*60)
print("步骤1: 加载清洗后的数据")
print("="*60)

# 读取清洗后的小时级数据
df = pd.read_csv('data_cleaned_hour.csv', index_col=0, parse_dates=True)

print(f"数据形状: {df.shape}")
print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
print(f"\n数据前5行:")
print(df.head())
print(f"\n数据列名: {df.columns.tolist()}")

# ==========================================
# 2. 添加总用电量特征
# ==========================================
print("\n" + "="*60)
print("步骤2: 添加总用电量特征")
print("="*60)

# 计算总用电量（Zone1 + Zone2 + Zone3）
df['TotalPower'] = df['PowerConsumption_Zone1'] + df['PowerConsumption_Zone2'] + df['PowerConsumption_Zone3']

print("已添加总用电量特征 'TotalPower'")
print(f"总用电量统计:")
print(f"  最小值: {df['TotalPower'].min():.2f}")
print(f"  最大值: {df['TotalPower'].max():.2f}")
print(f"  平均值: {df['TotalPower'].mean():.2f}")

# ==========================================
# 3. 选择特征
# ==========================================
print("\n" + "="*60)
print("步骤3: 选择特征")
print("="*60)

# 选择用于预测的特征
# 说明：
#   - 温度、湿度、风速、光照：气象特征，影响用电量
#   - 时间特征（hour, dayofweek, month）：周期性特征，捕捉用电规律
#   - 总用电量：历史用电量，用于自回归预测
feature_columns = [
    'Temperature',           # 温度
    'Humidity',              # 湿度
    'WindSpeed',             # 风速
    'GeneralDiffuseFlows',   # 总漫射流量
    'DiffuseFlows',         # 漫射流量
    'hour',                  # 小时（0-23）
    'dayofweek',             # 星期几（0-6）
    'month',                 # 月份（1-12）
    'TotalPower'             # 总用电量（预测目标）
]

# 提取特征数据
data = df[feature_columns].values

print(f"选择的特征: {feature_columns}")
print(f"特征数据形状: {data.shape}")

# 检查是否有缺失值
if np.isnan(data).any():
    print("警告: 数据中存在缺失值，进行填充...")
    data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))

# ==========================================
# 4. 数据归一化
# ==========================================
print("\n" + "="*60)
print("步骤4: 数据归一化")
print("="*60)

# 使用MinMaxScaler将数据缩放到[0,1]区间
# 原因：
#   1. 不同特征量纲不同（温度0-30，用电量0-100000）
#   2. 神经网络对输入数据的尺度敏感
#   3. 归一化可以加速训练，提高模型性能
scaler = MinMaxScaler()

# 拟合 scaler 并转换数据
data_normalized = scaler.fit_transform(data)

print("已使用MinMaxScaler进行归一化")
print(f"归一化后数据形状: {data_normalized.shape}")
print(f"归一化后数据范围: [{data_normalized.min():.4f}, {data_normalized.max():.4f}]")

# 保存 scaler，用于后续反归一化
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("已保存 scaler 到 'scaler.pkl'")

# ==========================================
# 5. 构建序列数据
# ==========================================
print("\n" + "="*60)
print("步骤5: 构建序列数据")
print("="*60)

def create_sequences(data, seq_length=24, pred_length=1):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        # 输入：过去seq_length小时的所有特征（反转，使最近的时间步在开头）
        seq = data[i:i+seq_length]
        seq_reversed = seq[::-1]  # 反转序列：t-1, t-2, ..., t-23
        X.append(seq_reversed)

        # 输出：未来pred_length小时的总用电量（最后一列）
        y.append(data[i+seq_length:i+seq_length+pred_length, -1])

    return np.array(X), np.array(y)


# 构建序列
seq_length = 24  # 使用过去24小时的数据
pred_length = 1  # 预测未来1小时

X, y = create_sequences(data_normalized, seq_length, pred_length)

print(f"序列长度: {seq_length} 小时")
print(f"预测长度: {pred_length} 小时")
print(f"序列数据形状:")
print(f"  X: {X.shape}  (样本数, 序列长度, 特征数)")
print(f"  y: {y.shape}  (样本数, 预测长度)")

# ==========================================
# 6. 按时间顺序划分数据集
# ==========================================
print("\n" + "="*60)
print("步骤6: 按时间顺序划分数据集")
print("="*60)

# 重要：必须按时间顺序划分，不能打乱！
# 原因：
#   1. 时间序列数据具有时间依赖性
#   2. 打乱会导致未来数据泄露到训练集
#   3. 正确做法：用过去数据训练，预测未来数据

# 划分比例：训练集70%，验证集15%，测试集15%
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

# 训练集：前70%
X_train, y_train = X[:train_size], y[:train_size]

# 验证集：中间15%
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]

# 测试集：后15%
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"数据集划分比例: 训练集70%，验证集15%，测试集15%")
print(f"样本数量:")
print(f"  训练集: {len(X_train)} 个样本")
print(f"  验证集: {len(X_val)} 个样本")
print(f"  测试集: {len(X_test)} 个样本")

# 检查数据是否有问题
print(f"\n数据检查:")
print(f"  X_train 形状: {X_train.shape}")
print(f"  y_train 形状: {y_train.shape}")
print(f"  X_val 形状: {X_val.shape}")
print(f"  y_val 形状: {y_val.shape}")
print(f"  X_test 形状: {X_test.shape}")
print(f"  y_test 形状: {y_test.shape}")

# ==========================================
# 7. 保存预处理后的数据
# ==========================================
print("\n" + "="*60)
print("步骤7: 保存预处理后的数据")
print("="*60)

# 保存为 NumPy 二进制文件（.npy）
# 优点：读取速度快，文件体积小
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("已保存预处理后的数据:")
print("  X_train.npy")
print("  y_train.npy")
print("  X_val.npy")
print("  y_val.npy")
print("  X_test.npy")
print("  y_test.npy")
print("  scaler.pkl")

# ==========================================
# 8. 输出数据信息，用于模型设计
# ==========================================
print("\n" + "="*60)
print("数据预处理完成！")
print("="*60)
print(f"\n数据信息汇总:")
print(f"  原始数据形状: {df.shape}")
print(f"  特征数量: {len(feature_columns)}")
print(f"  序列长度: {seq_length}")
print(f"  预测长度: {pred_length}")
print(f"  训练集样本数: {len(X_train)}")
print(f"  验证集样本数: {len(X_val)}")
print(f"  测试集样本数: {len(X_test)}")
print(f"\n模型输入形状: (batch_size, {seq_length}, {len(feature_columns)})")
print(f"模型输出形状: (batch_size, {pred_length})")

print("\n" + "="*60)
print("下一步: 运行 _02_model_lstm_baseline.py 创建LSTM基线模型")
print("="*60)


