"""
LSTM基线模型定义（修复dropout警告）
用于用电量预测的LSTM基线模型，作为对比实验的基准
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LSTMModel(nn.Module):
    """
    LSTM基线模型
    使用两层LSTM提取时间序列特征，最后通过全连接层预测用电量
    """

    def __init__(self, input_size, hidden_size1=128, hidden_size2=64,
                 num_layers=2, dropout=0.2, output_size=1):
        """
        初始化LSTM模型

        参数:
            input_size: 输入特征维度 (每个时间步的特征数)
            hidden_size1: 第一层LSTM的隐藏层维度
            hidden_size2: 第二层LSTM的隐藏层维度
            num_layers: LSTM层数（注意：这里定义为2，但实际使用两层独立的LSTM）
            dropout: dropout概率
            output_size: 输出维度 (预测未来1小时的用电量)
        """
        super(LSTMModel, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers

        # 第一层LSTM：返回完整序列 (batch, seq_len, hidden_size1)
        # 注意：dropout设为0，因为我们在外部使用nn.Dropout
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,  # 单层LSTM
            batch_first=True,
            dropout=0  # 设为0，因为我们在外部使用Dropout层
        )

        # Dropout层：防止过拟合（在LSTM层之间使用）
        self.dropout = nn.Dropout(dropout)

        # 第二层LSTM：只返回最后时间步的输出 (batch, hidden_size2)
        # 这里也设为0，因为后面没有更多LSTM层了
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=1,  # 单层LSTM
            batch_first=True,
            dropout=0  # 设为0
        )

        # 全连接层：将LSTM输出映射到预测值
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """
        初始化模型权重
        使用Xavier初始化方法，有助于训练稳定性
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)
               例如: (64, 24, 9) 表示batch_size=64, 序列长度=24小时, 9个特征

        返回:
            out: 预测值，形状为 (batch_size, output_size)
        """
        batch_size = x.size(0)

        # 第一层LSTM
        # 输入: (batch_size, seq_len, input_size)
        # 输出: (batch_size, seq_len, hidden_size1)
        lstm1_out, (h1, c1) = self.lstm1(x)

        # Dropout：在两层LSTM之间应用
        lstm1_out = self.dropout(lstm1_out)

        # 第二层LSTM
        # 输入: (batch_size, seq_len, hidden_size1)
        # 输出: (batch_size, seq_len, hidden_size2)
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)

        # 取最后一个时间步的输出
        # lstm2_out[:, -1, :] 形状: (batch_size, hidden_size2)
        last_output = lstm2_out[:, -1, :]

        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.fc2(out)

        return out

    def get_model_info(self):
        """
        获取模型信息：参数量、模型大小等
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            '模型名称': 'LSTM基线模型',
            '总参数量': total_params,
            '可训练参数量': trainable_params,
            '输入维度': self.lstm1.input_size,
            'LSTM1隐藏层维度': self.hidden_size1,
            'LSTM2隐藏层维度': self.hidden_size2,
            'LSTM层数': self.num_layers,
            'Dropout': self.dropout.p
        }
        return info


def evaluate_model(model, dataloader, device, scaler=None):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        dataloader: 数据加载器 (测试集或验证集)
        device: 设备 (cpu 或 cuda)
        scaler: 归一化器，用于反归一化 (可选)

    返回:
        metrics: 包含MAE, RMSE, MAPE, R2的字典
        predictions: 预测值 (如果scaler不为None，则返回反归一化后的值)
        targets: 真实值 (如果scaler不为None，则返回反归一化后的值)
    """
    model.eval()  # 设置为评估模式
    all_predictions = []
    all_targets = []

    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for batch_x, batch_y in dataloader:
            # 将数据移到设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            predictions = model(batch_x)

            # 收集预测值和真实值
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 如果提供了scaler，进行反归一化
    if scaler is not None:
        # 注意：scaler是在所有9个特征上拟合的
        # 但all_predictions和all_targets只有1列（目标变量，即第9列）
        # 所以需要创建dummy数组，把目标变量放在第9列，然后inverse_transform

        # 对预测值进行反归一化
        dummy_pred = np.zeros((len(all_predictions), 9))
        dummy_pred[:, -1] = all_predictions.flatten()
        all_predictions = scaler.inverse_transform(dummy_pred)[:, -1].reshape(-1, 1)

        # 对真实值进行反归一化
        dummy_target = np.zeros((len(all_targets), 9))
        dummy_target[:, -1] = all_targets.flatten()
        all_targets = scaler.inverse_transform(dummy_target)[:, -1].reshape(-1, 1)

    # 计算评估指标
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    # 计算MAPE，避免除以0
    mape = np.mean(np.abs((all_targets - all_predictions) / (np.abs(all_targets) + 1e-8))) * 100
    r2 = r2_score(all_targets, all_predictions)

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

    return metrics, all_predictions, all_targets


def predict_future(model, input_sequence, device, scaler=None):
    """
    使用训练好的模型进行预测

    参数:
        model: 训练好的模型
        input_sequence: 输入序列，形状为 (1, seq_length, input_size)
        device: 设备 (cpu 或 cuda)
        scaler: 归一化器，用于反归一化 (可选)

    返回:
        prediction: 预测值
    """
    model.eval()

    with torch.no_grad():
        # 将输入转换为tensor并移到设备
        input_tensor = torch.FloatTensor(input_sequence).to(device)

        # 前向传播
        prediction = model(input_tensor)
        prediction = prediction.cpu().numpy()

    # 如果提供了scaler，进行反归一化
    if scaler is not None:
        prediction = scaler.inverse_transform(prediction)

    return prediction


if __name__ == "__main__":
    # 测试代码：创建一个简单的LSTM模型并测试
    print("测试LSTM基线模型...")

    # 模型参数
    input_size = 9      # 9个特征
    seq_length = 24      # 过去24小时
    batch_size = 64      # batch大小

    # 创建模型
    model = LSTMModel(input_size=input_size)
    print("\n模型结构:")
    print(model)

    # 获取模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试前向传播
    test_input = torch.randn(batch_size, seq_length, input_size)
    test_output = model(test_input)
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"测试输出形状: {test_output.shape}")

    print("\nLSTM基线模型定义完成!")
