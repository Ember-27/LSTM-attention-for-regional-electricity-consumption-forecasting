"""
LSTM+Attention模型定义
在LSTM基础上添加Attention机制，提高用电量预测的准确性
Attention机制可以帮助模型关注重要的历史时间步
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class AttentionLayer(nn.Module):
    """
    Attention层：计算注意力权重，对LSTM输出进行加权求和
    使用更复杂的评分函数，增强表达能力
    """

    def __init__(self, hidden_size):
        """
        初始化Attention层

        参数:
            hidden_size: LSTM隐藏层维度
        """
        super(AttentionLayer, self).__init__()

        # 更复杂的注意力评分函数
        self.W = nn.Linear(hidden_size, hidden_size)  # 变换隐藏状态
        self.V = nn.Linear(hidden_size, 1, bias=False)  # 输出注意力分数

        # Softmax层：将注意力分数归一化为权重
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        """
        前向传播：计算注意力权重并加权求和

        参数:
            lstm_output: LSTM的输出，形状为 (batch_size, seq_length, hidden_size)
                        例如: (64, 24, 64) 表示64个样本，24个时间步，64维隐藏状态

        返回:
            context_vector: 上下文向量，形状为 (batch_size, hidden_size)
                           这是对所有时间步的加权求和
            attention_weights: 注意力权重，形状为 (batch_size, seq_length, 1)
                              可以用于可视化，观察模型关注哪些时间步
        """
        # 计算注意力分数（使用tanh增加非线性）
        # scores形状: (batch_size, seq_length, 1)
        scores = self.V(torch.tanh(self.W(lstm_output)))

        # 归一化为注意力权重
        # attention_weights形状: (batch_size, seq_length, 1)
        attention_weights = self.softmax(scores)

        # 加权求和：将LSTM输出与注意力权重相乘并求和
        # lstm_output * attention_weights: (batch_size, seq_length, hidden_size)
        # sum(dim=1): 在seq_length维度求和，得到 (batch_size, hidden_size)
        context_vector = torch.sum(lstm_output * attention_weights, dim=1)

        return context_vector, attention_weights



class LSTMAttentionModel(nn.Module):
    """
    LSTM+Attention模型
    在LSTM基础上添加Attention机制，提高预测准确性
    """

    def __init__(self, input_size, hidden_size1=128, hidden_size2=64,
                 num_layers=2, dropout=0.2, output_size=1):
        """
        初始化LSTM+Attention模型

        参数:
            input_size: 输入特征维度 (每个时间步的特征数)
            hidden_size1: 第一层LSTM的隐藏层维度
            hidden_size2: 第二层LSTM的隐藏层维度
            num_layers: LSTM层数
            dropout: dropout概率
            output_size: 输出维度 (预测未来1小时的用电量)
        """
        super(LSTMAttentionModel, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers

        # 第一层LSTM：返回完整序列 (batch, seq_len, hidden_size1)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 第二层LSTM：返回完整序列 (batch, seq_len, hidden_size2)
        # 注意：这里需要返回完整序列，因为Attention需要处理所有时间步
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
            dropout=0
        )

        # Attention层：计算注意力权重
        self.attention = AttentionLayer(hidden_size2)

        # 全连接层：将上下文向量映射到预测值
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        """
        初始化模型权重
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        """
        正确的前向传播：
        1. 经过LSTM1
        2. 经过Dropout
        3. 经过LSTM2
        4. 经过Attention层
        5. 经过全连接层输出预测值

        参数:
            x: 输入张量，形状为 (batch_size, seq_length, input_size)

        返回:
            predictions: 预测值，形状为 (batch_size, output_size)
            attention_weights: 注意力权重，形状为 (batch_size, seq_length, 1)
        """
        # 1. 第一层LSTM
        # lstm1_out形状: (batch_size, seq_length, hidden_size1)
        lstm1_out, (h_n, c_n) = self.lstm1(x)

        # 2. Dropout
        lstm1_out = self.dropout(lstm1_out)

        # 3. 第二层LSTM
        # lstm2_out形状: (batch_size, seq_length, hidden_size2)
        lstm2_out, (h_n, c_n) = self.lstm2(lstm1_out)

        # 4. Attention层
        # context_vector形状: (batch_size, hidden_size2)
        # attention_weights形状: (batch_size, seq_length, 1)
        context_vector, attention_weights = self.attention(lstm2_out)

        # 5. 全连接层
        # fc1_out形状: (batch_size, 32)
        fc1_out = self.relu(self.fc1(context_vector))

        # predictions形状: (batch_size, output_size)
        predictions = self.fc2(fc1_out)

        return predictions, attention_weights

    def get_model_info(self):
        """
        获取模型信息：参数量、模型大小等
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            '模型名称': 'LSTM+Attention模型',
            '总参数量': total_params,
            '可训练参数量': trainable_params,
            '输入维度': self.lstm1.input_size,
            'LSTM1隐藏层维度': self.hidden_size1,
            'LSTM2隐藏层维度': self.hidden_size2,
            'LSTM层数': self.num_layers,
            'Dropout': self.dropout.p,
            '是否有Attention': '是'
        }
        return info

    def visualize_attention(self, attention_weights, sample_idx=0, save_path=None):
        """
        可视化注意力权重

        参数:
            attention_weights: 注意力权重，形状为 (batch_size, seq_length, 1)
            sample_idx: 要可视化的样本索引
            save_path: 保存路径（可选）
        """
        # 获取指定样本的注意力权重
        attn = attention_weights[sample_idx].cpu().detach().numpy().flatten()

        # 创建时间轴
        time_steps = np.arange(len(attn))

        # 绘制条形图
        plt.figure(figsize=(12, 4))
        plt.bar(time_steps, attn, color='steelblue', alpha=0.7)
        plt.xlabel('Time Step (Past 24 Hours)', fontsize=12)
        plt.ylabel('Attention Weight', fontsize=12)
        plt.title(f'Attention Weight Distribution - Sample {sample_idx}', fontsize=14)
        plt.xticks(np.arange(0, 24, 2))
        plt.grid(True, alpha=0.3)

        # 在顶部显示总注意力权重
        plt.text(0.02, 0.98, f'Total Weight: {attn.sum():.3f}',
                 transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力权重图已保存到: {save_path}")

        plt.show()


def evaluate_model_with_attention(model, dataloader, device, scaler=None):
    """
    评估LSTM+Attention模型性能，并返回注意力权重

    参数:
        model: 训练好的LSTM+Attention模型
        dataloader: 数据加载器 (测试集或验证集)
        device: 设备 (cpu 或 cuda)
        scaler: 归一化器，用于反归一化 (可选)

    返回:
        metrics: 包含MAE, RMSE, MAPE, R2的字典
        predictions: 预测值
        targets: 真实值
        all_attention_weights: 所有样本的注意力权重
    """
    model.eval()  # 设置为评估模式
    all_predictions = []
    all_targets = []
    all_attention_weights = []

    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in dataloader:
            # 将数据移到设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播（获取预测值和注意力权重）
            predictions, attention_weights = model(batch_x)

            # 收集预测值、真实值和注意力权重
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())

    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_attention_weights = np.concatenate(all_attention_weights, axis=0)

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

    return metrics, all_predictions, all_targets, all_attention_weights


if __name__ == "__main__":
    # 测试代码：创建LSTM+Attention模型并测试
    print("测试LSTM+Attention模型...")

    # 模型参数
    input_size = 9  # 9个特征
    seq_length = 24  # 过去24小时
    batch_size = 64  # batch大小

    # 创建模型
    model = LSTMAttentionModel(input_size=input_size)
    print("\n模型结构:")
    print(model)

    # 获取模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试前向传播
    test_input = torch.randn(batch_size, seq_length, input_size)
    test_output, test_attention = model(test_input)
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"测试输出形状: {test_output.shape}")
    print(f"注意力权重形状: {test_attention.shape}")

    # 显示注意力权重统计信息
    attn_np = test_attention[0].detach().numpy().flatten()
    print(f"\n注意力权重统计 (第一个样本):")
    print(f"  最小值: {attn_np.min():.4f}")
    print(f"  最大值: {attn_np.max():.4f}")
    print(f"  平均值: {attn_np.mean():.4f}")
    print(f"  总和: {attn_np.sum():.4f}")

    print("\nLSTM+Attention模型定义完成!")
    print("=" * 60)
    print("注意：LSTM+Attention模型比LSTM基线模型多了")
    print("      Attention层，可以关注重要的历史时间步。")
    print("      这使得模型具有更好的可解释性。")
    print("=" * 60)
