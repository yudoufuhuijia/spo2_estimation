"""
特征数据加载器（文档4.3配置）
功能：从特征列表创建训练/验证DataLoader，适配20维特征
输入：特征列表（含20维向量）、标签 → 输出：训练/验证Loader
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict


def create_data_loaders(
        features_list: List[Dict],
        labels: np.ndarray,
        train_ratio: float = 0.8,
        batch_size: int = 16,
        shuffle: bool = True,
        random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练/验证数据加载器（文档4.3预期功能）
    Args:
        features_list: 特征列表，每个元素含'vector'键（20维numpy数组）
        labels: 标签数组（SpO2值，已归一化或原始值）
        train_ratio: 训练集比例（默认0.8，文档4.3要求）
        batch_size: 批次大小（默认16，文档4.3要求）
        shuffle: 是否打乱训练集（默认True）
        random_state: 随机种子（确保划分可复现）
    Returns:
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
    """
    # 步骤1：提取20维特征矩阵（文档4.3要求输入维度20）
    X = np.array([feat['vector'] for feat in features_list], dtype=np.float32)
    y = labels.astype(np.float32)

    # 验证特征维度（必须为20）
    assert X.shape[1] == 20, f"❌ 特征维度错误！预期20维，实际{X.shape[1]}维"
    assert len(X) == len(y), f"❌ 特征与标签数量不匹配！特征{len(X)}个，标签{len(y)}个"

    # 步骤2：转换为1D-CNN输入格式（batch,1,20）
    X = X[:, np.newaxis, :]  # 新增通道维度（文档3.2要求）

    # 步骤3：划分训练集/验证集（文档4.3要求8:2）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_ratio, shuffle=shuffle, random_state=random_state
    )

    # 步骤4：创建TensorDataset
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    # 步骤5：创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 打印加载器信息（文档4.3预期输出）
    print(f"✅ 数据集创建完成")
    print(f"   训练集: {len(train_dataset)} 样本 | 特征维度: {X_train.shape[2]} | 标签范围: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"   验证集: {len(val_dataset)} 样本 | 特征维度: {X_val.shape[2]} | 标签范围: {y_val.min():.2f} - {y_val.max():.2f}")

    # 测试批次加载（文档4.3要求）
    test_batch = next(iter(train_loader))
    feat_shape, label_shape = test_batch[0].shape, test_batch[1].shape
    print(f"\n✅ 数据加载器创建完成")
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   验证集: {len(val_loader.dataset)} 样本")
    print(f"   批次大小: {batch_size}")
    print(f"   测试批次加载:")
    print(f"     批次特征形状: {feat_shape} (应为(batch_size,1,20))")
    print(f"     批次标签形状: {label_shape} (应为(batch_size,))")
    print(f"     特征范围: {test_batch[0].min().item():.4f} ~ {test_batch[0].max().item():.4f}")
    print(f"     标签范围: {test_batch[1].min().item():.2f} ~ {test_batch[1].max().item():.2f}")

    return train_loader, val_loader


# 文档4.3测试代码（单独运行时执行）
if __name__ == "__main__":
    print("=" * 70)
    print("📝 特征数据加载器测试")
    print("=" * 70)

    # 【1/2】创建模拟特征（文档4.3格式）
    print("\n【1/2】创建模拟特征")
    n_samples = 100  # 文档4.3测试样本数
    features_list = []
    for _ in range(n_samples):
        # 模拟2.12特征提取模块输出（含20维向量）
        features = {
            'valid': True,
            'vector': np.random.randn(20).astype(np.float32)  # 20维特征
        }
        features_list.append(features)

    # 模拟SpO2标签（95-100，文档4.3范围）
    labels = np.random.uniform(95.0, 100.0, n_samples).astype(np.float32)
    print(f"✅ 模拟数据创建完成")
    print(f"   样本数: {n_samples}")
    print(f"   标签范围: {labels.min():.2f} - {labels.max():.2f}")

    # 【2/2】创建数据加载器（文档4.3配置）
    print("\n【2/2】创建数据加载器")
    train_loader, val_loader = create_data_loaders(
        features_list=features_list,
        labels=labels,
        train_ratio=0.8,
        batch_size=16
    )

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)