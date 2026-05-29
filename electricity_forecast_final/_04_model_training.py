"""
模型训练与评估脚本
训练LSTM基线模型和LSTM+Attention模型，评估性能，保存结果
"""
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from datetime import datetime

# 导入模型定义
# 导入模型定义
from _02_model_lstm_baseline import LSTMModel, evaluate_model
from _03_model_lstm_attention import LSTMAttentionModel, evaluate_model_with_attention

class EarlyStopping:
    """
    早停机制：当验证损失不再下降时停止训练，防止过拟合
    """

    def __init__(self, patience=10, min_delta=0):
        """
        初始化早停机制

        参数:
            patience: 允许验证损失不下降的轮数
            min_delta: 最小改善幅度，只有损失改善超过这个值才认为是改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        检查是否应该早停

        参数:
            val_loss: 当前验证损失

        返回:
            True: 应该停止训练
            False: 继续训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return False


def train_model(model, train_loader, val_loader, device,
                epochs=200, lr=0.001, model_name='model'):  # 修改1：增加epochs到200
    """
    训练模型
    参数:
        model: 要训练的模型
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        device: 训练设备 (cpu 或 cuda)
        epochs: 训练轮数
        lr: 学习率
        model_name: 模型名称（用于保存）

    返回:
        model: 训练好的模型
        train_losses: 训练损失历史
        val_losses: 验证损失历史
    """

    print(f"\n{'=' * 60}")
    print(f"开始训练 {model_name}")
    print(f"{'=' * 60}")

    # 将模型移到设备
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器：当验证损失停止下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 早停机制
    early_stopping = EarlyStopping(patience=20)

    # 记录训练历史
    train_losses = []
    val_losses = []

    # 开始训练
    start_time = time.time()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            # 将数据移到设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 前向传播
            if model_name == 'LSTM+Attention':
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)

            # 计算损失
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            optimizer.step()

            train_loss += loss.item()

        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                if model_name == 'LSTM+Attention':
                    outputs, _ = model(batch_x)
                else:
                    outputs = model(batch_x)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # 计算平均验证损失
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # +++ 添加以下代码 +++
        current_lr = optimizer.param_groups[0]['lr']

        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.6f}")  # 修改：使用 current_lr

        # 检查早停
        if early_stopping(val_loss):
            print(f"\n早停触发！在 epoch {epoch + 1} 停止训练")
            break

    # 训练完成
    training_time = time.time() - start_time
    print(f"\n训练完成！")
    print(f"总轮数: {epoch + 1}")
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"最终训练损失: {train_losses[-1]:.6f}")
    print(f"最终验证损失: {val_losses[-1]:.6f}")

    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, model_name, save_path=None):
    """
    绘制训练曲线
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name} Training Curves', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    #标记最佳epoch
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]
    plt.scatter(best_epoch, best_val_loss,
                color='red', s=100, zorder=5,
                label=f'Best Epoch ({best_epoch + 1})')
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.show()


def compare_models(lstm_metrics, lstm_attn_metrics, save_path=None):
    """
    对比两个模型的性能（学术规范版：分两个子图）
    - 子图1：MAE, RMSE（误差指标，越小越好）
    - 子图2：MAPE（左y轴）, R2（右y轴）
    """
    # 创建画布（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 子图1：MAE, RMSE ==========
    metrics_group1 = ['MAE', 'RMSE']
    x1 = np.arange(len(metrics_group1))
    width = 0.35

    lstm_vals1 = [lstm_metrics[m] for m in metrics_group1]
    attn_vals1 = [lstm_attn_metrics[m] for m in metrics_group1]

    bars1_1 = ax1.bar(x1 - width / 2, lstm_vals1, width,
                      label='LSTM', color='steelblue', alpha=0.8)
    bars1_2 = ax1.bar(x1 + width / 2, attn_vals1, width,
                      label='LSTM+Attention', color='coral', alpha=0.8)

    ax1.set_ylabel('Error Value', fontsize=12)
    ax1.set_title('(a) Error Metrics', fontsize=13, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics_group1, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 在柱子上显示数值
    def autolabel(ax, bars, fmt='{:.2f}'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(ax1, bars1_1, '{:.2f}')
    autolabel(ax1, bars1_2, '{:.2f}')

    # 添加改进百分比（误差指标：越小越好）
    for i, metric in enumerate(metrics_group1):
        lstm_val = lstm_metrics[metric]
        attn_val = lstm_attn_metrics[metric]
        improvement = (lstm_val - attn_val) / lstm_val * 100
        color = 'green' if improvement > 0 else 'red'
        y_pos = max(lstm_val, attn_val) * 1.05
        ax1.text(i, y_pos, f'{improvement:+.1f}%',
                 ha='center', va='bottom', color=color,
                 fontsize=10, fontweight='bold')

    # ========== 子图2：MAPE（左y轴）, R2（右y轴）==========
    # 左y轴：MAPE
    x2_left = np.array([0])
    lstm_mape = lstm_metrics['MAPE']
    attn_mape = lstm_attn_metrics['MAPE']

    bars2_1_left = ax2.bar(x2_left - width / 2, [lstm_mape], width,
                           label='LSTM (MAPE)', color='steelblue', alpha=0.8)
    bars2_2_left = ax2.bar(x2_left + width / 2, [attn_mape], width,
                           label='LSTM+Attention (MAPE)', color='coral', alpha=0.8)

    ax2.set_ylabel('MAPE (%)', fontsize=12)
    ax2.set_title('(b) MAPE and R²', fontsize=13, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['MAPE', 'R²'], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, max(lstm_mape, attn_mape) * 1.5])  # 提高上限，给百分比留空间

    # 在MAPE柱子上显示数值
    autolabel(ax2, bars2_1_left, '{:.2f}%')
    autolabel(ax2, bars2_2_left, '{:.2f}%')

    # ✅ 在MAPE图上添加改进百分比（位置更高，避免重叠）
    improvement_mape = (lstm_mape - attn_mape) / lstm_mape * 100
    color_mape = 'green' if improvement_mape > 0 else 'red'
    ax2.text(0, max(lstm_mape, attn_mape) * 1.08,
             f'{improvement_mape:+.1f}%',
             ha='center', va='bottom', color=color_mape,
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_mape, alpha=0.8))  # 添加白色背景框

    # 右y轴：R2
    ax2_right = ax2.twinx()
    x2_right = np.array([1])
    lstm_r2 = lstm_metrics['R2']
    attn_r2 = lstm_attn_metrics['R2']

    bars2_1_right = ax2_right.bar(x2_right - width / 2, [lstm_r2], width,
                                  label='LSTM (R²)', color='steelblue', alpha=0.4)
    bars2_2_right = ax2_right.bar(x2_right + width / 2, [attn_r2], width,
                                  label='LSTM+Attention (R²)', color='coral', alpha=0.4)

    ax2_right.set_ylabel('R² Score', fontsize=12)
    ax2_right.set_ylim([min(lstm_r2, attn_r2) - 0.05, 1.0])

    # 在R2柱子上显示数值
    autolabel(ax2_right, bars2_1_right, '{:.4f}')
    autolabel(ax2_right, bars2_2_right, '{:.4f}')

    # ✅ 在R2图上添加改进百分比（使用右轴坐标）
    improvement_r2 = (attn_r2 - lstm_r2) / abs(lstm_r2) * 100
    color_r2 = 'green' if improvement_r2 > 0 else 'red'
    ax2_right.text(1, max(lstm_r2, attn_r2) + 0.02,  # 位置稍微提高
                   f'{improvement_r2:+.1f}%',
                   ha='center', va='bottom', color=color_r2,
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_r2, alpha=0.8))  # 添加白色背景框

    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存到: {save_path}")

    plt.show()


def main():
    """
    主函数：训练两个模型，评估性能，保存结果
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已固定为: {seed}")

    print("=" * 60)
    print("用电量预测模型训练与评估")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    if device.type == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    # 加载预处理后的数据
    print("\n加载预处理后的数据...")
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_val = np.load('X_val.npy')
        y_val = np.load('y_val.npy')
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        print(f"  训练集: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"  验证集: X_val {X_val.shape}, y_val {y_val.shape}")
        print(f"  测试集: X_test {X_test.shape}, y_test {y_test.shape}")
        # 加载归一化器
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        print(f"  归一化器已加载: scaler.pkl")

    except FileNotFoundError as e:
        print(f"\n错误：找不到数据文件！")
        print(f"请确保已运行 1_data_preprocessing.py 生成数据文件")
        return

    # 创建数据加载器
    print("\n创建数据加载器...")
    batch_size = 64

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"  训练集批次数: {len(train_loader)}")
    print(f"  验证集批次数: {len(val_loader)}")
    print(f"  测试集批次数: {len(test_loader)}")

    # 获取模型参数
    input_size = X_train.shape[2]  # 每个时间步的特征数
    seq_length = X_train.shape[1]  # 序列长度

    # 训练LSTM基线模型
    print("\n" + "=" * 60)
    print("训练 LSTM 基线模型")
    print("=" * 60)

    lstm_model = LSTMModel(input_size=input_size)
    lstm_model, lstm_train_losses, lstm_val_losses = train_model(
        lstm_model, train_loader, val_loader, device,
        epochs=100, lr=0.001, model_name='LSTM'
    )

    # 保存LSTM模型
    torch.save(lstm_model.state_dict(), 'models/lstm_baseline.pth')
    print(f"\nLSTM模型已保存到: models/lstm_baseline.pth")

    # 绘制LSTM训练曲线
    plot_training_curves(
        lstm_train_losses, lstm_val_losses, 'LSTM',
        save_path='results/lstm_training_curves.png'
    )

    # 评估LSTM模型
    print("\n评估 LSTM 基线模型...")
    lstm_metrics, lstm_preds, lstm_targets = evaluate_model(
        lstm_model, test_loader, device, scaler=scaler
    )

    print(f"  MAE:  {lstm_metrics['MAE']:.4f}")
    print(f"  RMSE: {lstm_metrics['RMSE']:.4f}")
    print(f"  MAPE: {lstm_metrics['MAPE']:.2f}%")
    print(f"  R²:   {lstm_metrics['R2']:.4f}")

    # 训练LSTM+Attention模型
    print("\n" + "=" * 60)
    print("训练 LSTM+Attention 模型")
    print("=" * 60)

    lstm_attn_model = LSTMAttentionModel(input_size=input_size)
    lstm_attn_model, attn_train_losses, attn_val_losses = train_model(
        lstm_attn_model, train_loader, val_loader, device,
        epochs=100, lr=0.001, model_name='LSTM+Attention'
    )

    # 保存LSTM+Attention模型
    torch.save(lstm_attn_model.state_dict(), 'models/lstm_attention.pth')
    print(f"\nLSTM+Attention模型已保存到: models/lstm_attention.pth")

    # 绘制LSTM+Attention训练曲线
    plot_training_curves(
        attn_train_losses, attn_val_losses, 'LSTM+Attention',
        save_path='results/attention_training_curves.png'
    )

    # 评估LSTM+Attention模型
    print("\n评估 LSTM+Attention 模型...")
    attn_metrics, attn_preds, attn_targets, attention_weights = evaluate_model_with_attention(
        lstm_attn_model, test_loader, device, scaler=scaler
    )

    print(f"  MAE:  {attn_metrics['MAE']:.4f}")
    print(f"  RMSE: {attn_metrics['RMSE']:.4f}")
    print(f"  MAPE: {attn_metrics['MAPE']:.2f}%")
    print(f"  R²:   {attn_metrics['R2']:.4f}")

    print("\n第一个样本的注意力权重:")
    print(attention_weights[0].flatten())
    print(f"注意力权重总和: {attention_weights[0].sum():.4f}")

    # 绘制注意力权重分布图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(attention_weights[0])), attention_weights[0].flatten(),
            color='steelblue', alpha=0.7)
    plt.xlabel('Time Step (Past 24 Hours)') ##时间步（过去24小时）
    plt.ylabel('Attention Weight')    ##注意力权重
    plt.title('Attention Weight Distribution (First Sample)')  ##注意力权重分布（第一个样本）
    plt.grid(True, alpha=0.3)
    plt.savefig('results/attention_weights_sample.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 对比两个模型
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    compare_models(lstm_metrics, attn_metrics,
                   save_path='results/model_comparison.png')

    # 打印对比表格
    print("\n性能对比表格:")
    print("-" * 60)
    print(f"{'指标':<10} {'LSTM':<15} {'LSTM+Attention':<15} {'改进':<10}")
    print("-" * 60)

    for metric in ['MAE', 'RMSE', 'MAPE']:
        lstm_val = lstm_metrics[metric]
        attn_val = attn_metrics[metric]
        improvement = (lstm_val - attn_val) / lstm_val * 100
        print(f"{metric:<10} {lstm_val:<15.4f} {attn_val:<15.4f} {improvement:>+.1f}%")

    metric = 'R2'
    lstm_val = lstm_metrics[metric]
    attn_val = attn_metrics[metric]
    improvement = (attn_val - lstm_val) / abs(lstm_val) * 100
    print(f"{metric:<10} {lstm_val:<15.4f} {attn_val:<15.4f} {improvement:>+.1f}%")
    print("-" * 60)

    # 保存结果到文件
    print("\n保存结果...")
    results_df = pd.DataFrame({
        'Model': ['LSTM', 'LSTM+Attention'],
        'MAE': [lstm_metrics['MAE'], attn_metrics['MAE']],
        'RMSE': [lstm_metrics['RMSE'], attn_metrics['RMSE']],
        'MAPE': [lstm_metrics['MAPE'], attn_metrics['MAPE']],
        'R2': [lstm_metrics['R2'], attn_metrics['R2']]
    })
    results_df.to_csv('results/model_results.csv', index=False)
    print(f"结果已保存到: results/model_results.csv")

    print("\n" + "=" * 60)
    print("训练与评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 运行主函数
    main()
