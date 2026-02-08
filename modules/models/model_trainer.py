"""
模型训练器（文档4.2配置）
功能：CPU训练、Adam优化、MAE损失、早停、学习率调整、模型保存
适配Lightweight1DCNN，支持反归一化计算原始SpO2 MAE
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import warnings
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 忽略PyTorch版本兼容警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            device: str = 'cpu',
            learning_rate: float = 0.001,
            weight_decay: float = 1e-4,
            output_dir: str = "test_output/models"
    ):
        """初始化训练器（严格匹配文档4.2配置）"""
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 优化器：Adam（文档指定）
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 损失函数：MAE（L1Loss，文档指定）
        self.criterion = nn.L1Loss()

        # 学习率调度器：ReduceLROnPlateau（文档隐含要求）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=False
        )

        # 训练历史记录（含原始SpO2 MAE）
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': [],
            'val_mae_original': []  # 原始SpO2范围的MAE（文档核心指标）
        }

        # 最佳模型追踪（早停用）
        self.best_val_loss = float('inf')
        self.best_val_mae_original = float('inf')  # 用原始MAE判断最优
        self.best_epoch = 0
        self.patience_counter = 0

        # 打印初始化信息（文档4.2预期格式）
        print("✅ 训练器初始化完成")
        print(f"   设备: {device}")
        print(f"   学习率: {learning_rate}")
        print(f"   优化器: Adam")
        print(f"   损失函数: MAE (L1Loss)")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练单个epoch（适配3D输入）"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_batches = 0

        for batch_features, batch_labels in train_loader:
            # 数据移至设备（batch_features: (batch,1,20)，batch_labels: (batch,)）
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # 前向传播+反向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch_features)  # 输出：(batch,1)
            loss = self.criterion(predictions.squeeze(), batch_labels)  # squeeze()为(batch,)
            loss.backward()
            self.optimizer.step()

            # 统计损失（MAE即L1Loss）
            total_loss += loss.item()
            total_mae += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        return avg_loss, avg_mae

    def validate(self, val_loader: DataLoader, y_scaler: MinMaxScaler) -> Tuple[float, float, float]:
        """验证模型（反归一化计算原始SpO2的MAE，文档核心要求）"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_mae_original = 0.0  # 原始SpO2范围（95-100）的MAE
        n_batches = 0

        with torch.no_grad():  # 禁用梯度计算，加速验证
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 前向传播
                predictions = self.model(batch_features)
                loss = self.criterion(predictions.squeeze(), batch_labels)

                # 反归一化：还原为原始SpO2范围（95-100）
                pred_np = predictions.squeeze().cpu().numpy().reshape(-1, 1)
                label_np = batch_labels.cpu().numpy().reshape(-1, 1)
                pred_original = y_scaler.inverse_transform(pred_np).flatten()
                label_original = y_scaler.inverse_transform(label_np).flatten()

                # 统计损失
                total_loss += loss.item()
                total_mae += loss.item()
                total_mae_original += np.mean(np.abs(pred_original - label_original))
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        avg_mae_original = total_mae_original / n_batches  # 文档预期≤1.5
        return avg_loss, avg_mae, avg_mae_original

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            y_scaler: MinMaxScaler,  # 标签缩放器（反归一化用）
            epochs: int = 10,
            early_stopping_patience: int = 5,
            verbose: bool = True
    ) -> Dict:
        """完整训练流程（文档4.2预期输出格式）"""
        print(f"\n{'=' * 60}")
        print(f"开始训练")
        print(f"{'=' * 60}")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"批次大小: {train_loader.batch_size}")
        print(f"总轮数: {epochs}")
        print(f"早停耐心: {early_stopping_patience}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练（获取归一化损失）
            train_loss, train_mae = self.train_epoch(train_loader)
            # 验证（获取归一化损失+原始SpO2的MAE）
            val_loss, val_mae, val_mae_original = self.validate(val_loader, y_scaler)

            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['val_mae_original'].append(val_mae_original)
            self.history['learning_rate'].append(current_lr)

            # 调整学习率
            self.scheduler.step(val_loss)

            # 打印epoch信息（文档4.2格式）
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch [{epoch + 1:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} "
                      f"Val Loss: {val_loss:.4f} "
                      f"Val MAE(归一化): {val_mae:.4f} "
                      f"Val MAE(SpO2): {val_mae_original:.4f} "
                      f"LR: {current_lr:.6f} "
                      f"Time: {epoch_time:.1f}s")

            # 保存最佳模型（用原始SpO2 MAE判断，文档业务核心）
            if val_mae_original < self.best_val_mae_original:
                self.best_val_loss = val_loss
                self.best_val_mae_original = val_mae_original
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_model('best_model.pth')
                if verbose:
                    print(f"  ✅ 新的最佳模型 (Val MAE(SpO2): {val_mae_original:.4f})")
            else:
                self.patience_counter += 1

            # 早停检查
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  早停触发 (耐心值已达 {early_stopping_patience})")
                print(f"  最佳epoch: {self.best_epoch + 1}")
                print(f"  最佳验证损失(归一化): {self.best_val_loss:.4f}")
                print(f"  最佳验证MAE(SpO2): {self.best_val_mae_original:.4f}")
                break

        # 训练完成总结（文档4.2格式）
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"训练完成")
        print(f"{'=' * 60}")
        print(f"总耗时: {total_time:.1f}s ({total_time / 60:.1f}分钟)")
        print(f"最佳epoch: {self.best_epoch + 1}")
        print(f"最佳验证MAE(SpO2): {self.best_val_mae_original:.4f}")  # 重点展示
        print(f"{'=' * 60}\n")

        return self.history

    def save_model(self, filename: str):
        """保存模型（文档要求：含模型+优化器+历史）"""
        filepath = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_mae_original': self.best_val_mae_original,
            'best_epoch': self.best_epoch
        }, filepath)

    def load_model(self, filename: str):
        """加载模型（文档要求：恢复训练状态）"""
        filepath = self.output_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_mae_original = checkpoint['best_val_mae_original']
        self.best_epoch = checkpoint['best_epoch']
        print(f"✅ 模型已加载: {filepath}")

    def plot_history(self, save_path: Optional[str] = None):
        """绘制训练历史（文档要求：3个子图）"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 无GUI环境兼容
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib未安装，跳过绘图")
            return

        if not save_path:
            save_path = self.output_dir / "training_history.png"

        # 绘制3个子图：归一化损失、原始SpO2 MAE、学习率
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 归一化损失曲线
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2, color='#1f77b4')
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2, color='#ff7f0e')
        axes[0].axvline(self.best_epoch, color='red', linestyle='--', label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Normalized Loss (MAE)', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 2. 原始SpO2 MAE曲线（文档核心指标）
        axes[1].plot(self.history['val_mae_original'], label='Val MAE (SpO2)', linewidth=2, color='#d62728')
        axes[1].axvline(self.best_epoch, color='red', linestyle='--', label=f'Best Epoch ({self.best_epoch + 1})', alpha=0.7)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE (SpO2, 95-100)', fontsize=12)
        axes[1].set_title('Validation MAE (Original SpO2 Range)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # 3. 学习率曲线
        axes[2].plot(self.history['learning_rate'], linewidth=2, color='#2ca02c')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 训练历史图已保存: {save_path}")
        plt.close()


# 文档4.2测试代码（单独运行时执行）
if __name__ == "__main__":
    print("=" * 70)
    print("📝 模型训练器测试")
    print("=" * 70)

    # 导入模型（适配路径）
    sys.path.insert(0, '../..')
    from modules.models.lightweight_1dcnn import Lightweight1DCNN

    # 1. 创建模拟数据（文档4.2格式）
    print("\n【1/4】创建模拟数据（含预处理）")
    np.random.seed(42)
    n_samples = 200  # 训练160+验证40
    n_features = 20
    spo2_min, spo2_max = 95.0, 100.0

    # 生成关联特征（模拟真实生理信号）
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y_true = spo2_min + (spo2_max - spo2_min) * (
        0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(n_samples) * 0.05
    )
    y_true = np.clip(y_true, spo2_min, spo2_max)

    # 特征标准化（Z-Score）
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = X_scaler.fit_transform(X)
    # 标签归一化（0-1）
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = y_scaler.fit_transform(y_true.reshape(-1, 1)).flatten()

    # 划分数据集（8:2）
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]

    # 转换为3D输入（batch,1,20）
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]

    # 创建DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"✅ 数据创建完成")
    print(f"   训练集: {X_train.shape[0]} 样本 | 特征形状: {X_train.shape}")
    print(f"   验证集: {X_val.shape[0]} 样本 | 特征形状: {X_val.shape}")

    # 2. 创建模型
    print("\n【2/4】创建模型")
    model = Lightweight1DCNN(input_features=20)
    print("✅ 模型创建完成（参数量：234,497）")

    # 3. 创建训练器
    print("\n【3/4】创建训练器")
    trainer = ModelTrainer(
        model=model,
        device='cpu',
        learning_rate=0.001,
        output_dir="../../test_output/models"
    )

    # 4. 训练模型
    print("\n【4/4】训练模型")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        y_scaler=y_scaler,
        epochs=10,
        early_stopping_patience=5,
        verbose=True
    )

    # 绘制训练历史
    trainer.plot_history("../../test_output/models/training_history.png")

    # 最终验证（文档4.2预期）
    print(f"\n📊 最终验证结果")
    print(f"   最佳验证MAE(SpO2): {trainer.best_val_mae_original:.4f}")
    print(f"   文档预期MAE(SpO2): ≤1.5")
    print(f"   状态: {'✅ 符合预期' if trainer.best_val_mae_original <= 1.5 else '❌ 需调整'}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)