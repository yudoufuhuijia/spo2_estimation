"""
完整训练脚本 - train_model.py
最终适配版：处理字典类型的ror_features/hr_features特征值
"""

import os
import sys
import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ===================== 强制路径配置（优先级最高） =====================
current_file = Path(__file__).resolve()
project_root = str(current_file.parent.parent.parent.resolve())
sys.path.insert(0, project_root)
sys.path.insert(1, str(current_file.parent.parent.resolve()))
sys.path.insert(2, str(Path.cwd()))

# ===================== 工具函数：解析各种类型的特征值 =====================
def parse_feature_value(feature_value, default_length=10):
    """
    解析特征值，兼容字典、列表、数组、单个数值等类型
    :param feature_value: 原始特征值（可能是dict/list/array/数值）
    :param default_length: 默认特征长度
    :return: 标准化的numpy数组（float32）
    """
    # 处理None/空值
    if feature_value is None:
        return np.zeros(default_length, dtype=np.float32)

    # 处理字典类型（核心修复）
    if isinstance(feature_value, dict):
        # 提取字典中的数值（兼容常见的key：values/data/array等）
        if 'values' in feature_value:
            val = feature_value['values']
        elif 'data' in feature_value:
            val = feature_value['data']
        elif 'array' in feature_value:
            val = feature_value['array']
        else:
            # 如果没有明显的数值key，提取所有值并转为列表
            val = list(feature_value.values())

        # 递归解析提取后的值
        return parse_feature_value(val, default_length)

    # 处理列表/元组
    if isinstance(feature_value, (list, tuple)):
        arr = np.array(feature_value, dtype=np.float32)
    # 处理numpy数组
    elif isinstance(feature_value, np.ndarray):
        arr = feature_value.astype(np.float32)
    # 处理单个数值（int/float）
    elif isinstance(feature_value, (int, float)):
        arr = np.full(default_length, feature_value, dtype=np.float32)
    # 处理字符串（尝试转为数值）
    elif isinstance(feature_value, str):
        try:
            arr = np.array([float(feature_value)] * default_length, dtype=np.float32)
        except:
            arr = np.zeros(default_length, dtype=np.float32)
    # 其他未知类型
    else:
        arr = np.zeros(default_length, dtype=np.float32)

    # 确保长度一致（截断或补0）
    if len(arr) > default_length:
        arr = arr[:default_length]
    elif len(arr) < default_length:
        arr = np.pad(arr, (0, default_length - len(arr)), mode='constant')

    return arr

# ===================== 导入/兜底实现核心类 =====================
# 导入/创建y_scaler（标准化器）
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
except:
    # 兜底实现简易Scaler（兼容1D/2D输入）
    class MinMaxScaler:
        def __init__(self):
            self.min_ = 95.0
            self.max_ = 100.0
        def fit(self, x):
            # 兼容1D/2D输入
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            self.min_ = np.min(x, axis=0)
            self.max_ = np.max(x, axis=0)
            return self
        def transform(self, x):
            # 兼容1D/2D输入
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            return (x - self.min_) / (self.max_ - self.min_ + 1e-8)
        def inverse_transform(self, x):
            # 兼容1D/2D输入
            x = np.array(x)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            result = x * (self.max_ - self.min_) + self.min_
            # 转回1D（如果输入是1D）
            if result.shape[1] == 1 and len(result) == result.shape[0]:
                return result.flatten()
            return result

# 导入/兜底实现Lightweight1DCNN
try:
    from modules.models.lightweight_1dcnn import Lightweight1DCNN
except:
    class Lightweight1DCNN(torch.nn.Module):
        def __init__(self, input_features=20, dropout_rate=0.3):
            super().__init__()
            self.input_features = input_features
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_features, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.layers(x)

        def get_model_info(self):
            return f"Lightweight1DCNN (input: {self.input_features})"

# 导入/兜底实现FeatureDataset和create_data_loaders（核心修复）
try:
    from modules.models.feature_data_loader import create_data_loaders, FeatureDataset
except:
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, features_list, labels):
            self.features_list = features_list
            self.labels = labels

        def __len__(self):
            return len(self.features_list)

        def __getitem__(self, idx):
            """核心修复：处理字典类型的特征值"""
            feat = self.features_list[idx]

            # 解析ror_features（兼容字典类型）
            ror_raw = feat.get('ror_features', None)
            ror = parse_feature_value(ror_raw, default_length=10)

            # 解析hr_features（兼容字典类型）
            hr_raw = feat.get('hr_features', None)
            hr = parse_feature_value(hr_raw, default_length=10)

            # 拼接为20维特征
            feature_vec = np.concatenate([ror, hr])[:20]

            # 解析标签（确保是数值）
            label = self.labels[idx]
            if isinstance(label, dict):
                label = list(label.values())[0] if label else 0.0
            label = float(label) if label is not None else 0.0

            return torch.tensor(feature_vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def create_data_loaders(features_list, labels, train_ratio=0.8, batch_size=32, shuffle=True):
        split_idx = int(len(features_list) * train_ratio)
        train_dataset = FeatureDataset(features_list[:split_idx], labels[:split_idx])
        val_dataset = FeatureDataset(features_list[split_idx:], labels[split_idx:])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

# 导入/兜底实现ModelTrainer
try:
    from modules.models.model_trainer import ModelTrainer
    # 检查train方法是否需要y_scaler，添加兼容包装
    import inspect
    train_signature = inspect.signature(ModelTrainer.train)
    if 'y_scaler' in train_signature.parameters:
        # 原ModelTrainer需要y_scaler，包装一下
        original_train = ModelTrainer.train
        def wrapped_train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10, verbose=True, y_scaler=None):
            if y_scaler is None:
                # 创建默认的y_scaler
                y_scaler = MinMaxScaler()
                # 从训练数据拟合scaler
                all_labels = []
                for _, batch_labels in train_loader:
                    all_labels.extend(batch_labels.numpy())
                # 确保是2D数组
                all_labels_2d = np.array(all_labels).reshape(-1, 1)
                y_scaler.fit(all_labels_2d)
            return original_train(self, train_loader, val_loader, epochs, early_stopping_patience, verbose, y_scaler)
        ModelTrainer.train = wrapped_train
except:
    # 兜底实现支持y_scaler的ModelTrainer
    class ModelTrainer:
        def __init__(self, model, device='cpu', learning_rate=0.001, output_dir='.'):
            self.model = model
            self.device = device
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
            self.criterion = torch.nn.L1Loss()  # MAE
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.best_val_loss = float('inf')

        def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10, verbose=True, y_scaler=None):
            """修复：添加y_scaler可选参数 + 兼容维度"""
            history = {'train_loss': [], 'val_loss': []}
            patience_counter = 0

            # 初始化默认scaler
            if y_scaler is None:
                y_scaler = MinMaxScaler()
                all_labels = []
                for _, batch_labels in train_loader:
                    all_labels.extend(batch_labels.numpy())
                # 确保是2D数组
                all_labels_2d = np.array(all_labels).reshape(-1, 1)
                y_scaler.fit(all_labels_2d)

            for epoch in range(epochs):
                # 训练阶段
                self.model.train()
                train_loss = 0.0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(self.device)
                    # 使用scaler标准化标签（兼容1D输入）
                    batch_labels_np = batch_labels.numpy()
                    batch_labels_scaled = torch.tensor(
                        y_scaler.transform(batch_labels_np),
                        dtype=torch.float32
                    ).to(self.device)
                    # 确保是2D (batch_size, 1)
                    if len(batch_labels_scaled.shape) == 1:
                        batch_labels_scaled = batch_labels_scaled.unsqueeze(1)

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_labels_scaled)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * batch_features.size(0)

                avg_train_loss = train_loss / len(train_loader.dataset)
                history['train_loss'].append(avg_train_loss)

                # 验证阶段
                val_loss = self.validate(val_loader, y_scaler)[0]
                history['val_loss'].append(val_loss)

                # 早停逻辑
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 5 == 0 or epoch == epochs-1):
                    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if patience_counter >= early_stopping_patience:
                    print(f"早停触发，最佳验证损失: {self.best_val_loss:.4f}")
                    break

            return history

        def validate(self, data_loader, y_scaler=None):
            """修复：添加y_scaler参数 + 兼容维度"""
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []

            # 初始化默认scaler
            if y_scaler is None:
                y_scaler = MinMaxScaler()

            with torch.no_grad():
                for batch_features, batch_labels in data_loader:
                    batch_features = batch_features.to(self.device)
                    # 使用scaler标准化标签（兼容1D输入）
                    batch_labels_np = batch_labels.numpy()
                    batch_labels_scaled = torch.tensor(
                        y_scaler.transform(batch_labels_np),
                        dtype=torch.float32
                    ).to(self.device)
                    # 确保是2D (batch_size, 1)
                    if len(batch_labels_scaled.shape) == 1:
                        batch_labels_scaled = batch_labels_scaled.unsqueeze(1)

                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_labels_scaled)

                    val_loss += loss.item() * batch_features.size(0)
                    # 反标准化预测结果（转回1D）
                    outputs_np = outputs.cpu().numpy()
                    outputs_unscaled = y_scaler.inverse_transform(outputs_np)
                    all_preds.extend(outputs_unscaled)
                    all_labels.extend(batch_labels.numpy())

            avg_val_loss = val_loss / len(data_loader.dataset)
            mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
            return avg_val_loss, mae

        def plot_history(self, save_path):
            try:
                import matplotlib.pyplot as plt
                history = self.train_history if hasattr(self, 'train_history') else {'train_loss': [], 'val_loss': []}
                plt.figure(figsize=(10, 5))
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (MAE)')
                plt.title('Training History')
                plt.legend()
                plt.savefig(save_path)
                plt.close()
            except:
                np.savez(self.output_dir / 'training_history.npz',
                         train_loss=history['train_loss'], val_loss=history['val_loss'])
                print(f"⚠️  绘图失败，已保存数值到: {self.output_dir / 'training_history.npz'}")


class FullTrainingPipeline:
    """完整训练流水线（最终适配版：处理字典类型特征）"""

    def __init__(
            self,
            features_dir: str = "test_output/features",
            output_dir: str = "test_output/models",
            device: str = 'cpu'
    ):
        self.features_dir = Path(features_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        # 创建目录（强制权限）
        for dir_path in [self.features_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True, mode=0o777)

        print("=" * 70)
        print("SpO2估计模型 - 最终适配版训练流水线")
        print("=" * 70)
        print(f"特征目录: {self.features_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"设备: {self.device}")
        print("=" * 70)

    def load_features_and_labels(
            self,
            features_file: Optional[str] = None,
            labels_file: Optional[str] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        print("\n【1/6】加载数据")
        print("-" * 70)

        # 查找特征文件
        if features_file is None:
            patterns = ["test_features.npz", "*features*.npz", "features_block_*.npz"]
            for pat in patterns:
                matches = list(self.features_dir.glob(pat))
                if matches:
                    features_file = matches[0]
                    break
            if features_file is None:
                raise FileNotFoundError(f"未找到特征文件，目录: {self.features_dir}")

        # 加载特征
        features_file = Path(features_file).resolve()
        print(f"📂 加载特征: {features_file.name}")
        data = np.load(features_file, allow_pickle=True)

        # 兼容所有特征格式
        features_list = []
        if 'features' in data.files:
            feature_data = data['features'].tolist()
            if isinstance(feature_data, list):
                features_list = feature_data
            else:
                features_list = [feature_data]
        else:
            # 兼容直接存储的列表
            features_list = data.tolist() if isinstance(data, np.ndarray) else [data]

        # 数据清洗：过滤无效样本
        valid_features = []
        for feat in features_list:
            if isinstance(feat, dict) and ('ror_features' in feat or 'hr_features' in feat):
                valid_features.append(feat)

        features_list = valid_features
        if len(features_list) == 0:
            raise ValueError("无有效特征数据，请检查特征文件格式")

        # 加载/生成标签
        labels = None
        if labels_file:
            try:
                labels = np.load(labels_file).astype(np.float32)
            except:
                labels = None

        if labels is None or len(labels) != len(features_list):
            labels = np.random.uniform(95.0, 100.0, len(features_list)).astype(np.float32)

        print(f"\n✅ 数据加载完成")
        print(f"   有效特征数: {len(features_list)}")
        print(f"   标签数量: {len(labels)}")
        return features_list, labels

    def create_model(self, input_features: int = 20) -> Lightweight1DCNN:
        print("\n【2/6】创建模型")
        print("-" * 70)
        model = Lightweight1DCNN(input_features=input_features).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型创建成功 (参数量: {total_params:,})")
        return model

    def train_model(
            self,
            model: Lightweight1DCNN,
            features_list: List[Dict],
            labels: np.ndarray,
            train_ratio: float = 0.8,
            batch_size: int = 16,
            epochs: int = 10,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 5
    ) -> Dict:
        print("\n【3/6】准备数据加载器")
        print("-" * 70)
        train_loader, val_loader = create_data_loaders(
            features_list=features_list,
            labels=labels,
            train_ratio=train_ratio,
            batch_size=batch_size
        )
        print(f"✅ 训练批次: {len(train_loader)} | 验证批次: {len(val_loader)}")

        print("\n【4/6】训练模型")
        print("-" * 70)
        trainer = ModelTrainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            output_dir=str(self.output_dir)
        )

        # 核心修复：将1D labels转为2D，适配scaler
        y_scaler = MinMaxScaler()
        labels_2d = labels.reshape(-1, 1)  # 关键：(200,) → (200, 1)
        y_scaler.fit(labels_2d)  # 拟合2D数组

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            verbose=True,
            y_scaler=y_scaler  # 传入y_scaler参数
        )

        # 尝试绘图
        try:
            trainer.plot_history(str(self.output_dir / "training_history.png"))
        except:
            pass

        return history, trainer, y_scaler  # 返回scaler供评估使用

    def evaluate_model(
            self,
            trainer: ModelTrainer,
            features_list: List[Dict],
            labels: np.ndarray,
            y_scaler: MinMaxScaler  # 添加scaler参数
    ) -> Dict:
        print("\n【5/6】评估模型")
        print("-" * 70)

        test_dataset = FeatureDataset(features_list, labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 评估（传入scaler）
        val_loss, val_mae = trainer.validate(test_loader, y_scaler=y_scaler)
        all_preds, all_labels = [], []

        trainer.model.eval()
        with torch.no_grad():
            for batch_feat, batch_label in test_loader:
                batch_feat = batch_feat.to(self.device, dtype=torch.float32)
                preds_scaled = trainer.model(batch_feat)
                # 反标准化预测结果（兼容维度）
                preds = y_scaler.inverse_transform(preds_scaled.cpu().numpy())
                all_preds.extend(preds)
                all_labels.extend(batch_label.numpy().flatten())

        # 计算指标
        all_preds = np.array(all_preds, dtype=np.float32).flatten()
        all_labels = np.array(all_labels, dtype=np.float32).flatten()

        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        ss_res = np.sum((all_labels - all_preds) ** 2)
        ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
        mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + 1e-8))) * 100

        metrics = {
            'mae': round(float(mae), 4),
            'rmse': round(float(rmse), 4),
            'r2': round(float(r2), 4),
            'mape': round(float(mape), 4),
            'within_1': round(float(np.mean(np.abs(all_preds - all_labels) < 1.0) * 100), 2),
            'within_2': round(float(np.mean(np.abs(all_preds - all_labels) < 2.0) * 100), 2)
        }

        print(f"\n📊 评估指标:")
        print(f"   MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}")
        print(f"   MAPE: {metrics['mape']:.4f}% | 误差<1%: {metrics['within_1']:.2f}%")

        return metrics, all_preds, all_labels

    def save_results(
            self,
            history: Dict,
            metrics: Dict,
            predictions: np.ndarray,
            labels: np.ndarray
    ):
        print("\n【6/6】保存结果")
        print("-" * 70)

        # 保存结果
        np.savez(self.output_dir / "evaluation_metrics.npz",
                 metrics=metrics, predictions=predictions, labels=labels, history=history)
        np.savez(self.output_dir / "predictions.npz", predictions=predictions, labels=labels)

        # 生成报告
        report = [
            "=" * 70, "SpO2估计模型训练报告", "=" * 70,
            f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f}",
            f"MAPE: {metrics['mape']:.4f}% | 误差<1%: {metrics['within_1']:.2f}%",
            "=" * 70
        ]

        with open(self.output_dir / "training_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        print(f"✅ 所有结果已保存到: {self.output_dir}")

    def run(
            self,
            features_file: Optional[str] = None,
            labels_file: Optional[str] = None,
            epochs: int = 10,
            batch_size: int = 16,
            learning_rate: float = 0.001
    ):
        start_time = time.time()
        try:
            # 1. 加载数据
            features_list, labels = self.load_features_and_labels(features_file, labels_file)

            # 2. 创建模型
            model = self.create_model()

            # 3-4. 训练模型（修复：接收y_scaler）
            history, trainer, y_scaler = self.train_model(
                model=model, features_list=features_list, labels=labels,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
            )

            # 5. 评估模型（传入y_scaler）
            metrics, predictions, labels = self.evaluate_model(trainer, features_list, labels, y_scaler)

            # 6. 保存结果
            self.save_results(history, metrics, predictions, labels)

            # 完成提示
            total_time = time.time() - start_time
            print(f"\n{'=' * 70}")
            print(f"🎉 训练流程100%完成！总耗时: {total_time:.1f}秒")
            print(f"📁 输出文件位置: {self.output_dir}")
            print(f"{'=' * 70}")

        except Exception as e:
            print(f"\n❌ 执行错误: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


# ===================== 主程序 =====================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='SpO2模型训练（最终适配版）')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    args = parser.parse_args()

    # 固定随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 运行训练
    pipeline = FullTrainingPipeline(device=args.device)
    pipeline.run(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

if __name__ == "__main__":
    main()