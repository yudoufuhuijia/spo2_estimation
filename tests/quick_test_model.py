"""
2.13 轻量化模型搭建 - 一键快速测试（文档5要求）
功能：验证模块导入、模型结构、数据加载、训练流程、结果保存
预期输出：与文档5完全一致
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 配置项目根路径（确保能导入modules）
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)
print(f"🔧 项目根路径: {project_root}")

# 确保输出目录存在
output_dir = os.path.join(project_root, "test_output", "models")
os.makedirs(output_dir, exist_ok=True)

# 打印标题（文档5格式）
print("\n" + "=" * 60)
print("2.13 轻量化模型搭建 - 一键快速测试")
print("=" * 60)

# 【1/5】导入模块（文档5步骤1）
print(f"\n【1/5】导入模块...")
try:
    from modules.models.lightweight_1dcnn import Lightweight1DCNN
    from modules.models.model_trainer import ModelTrainer
    from modules.models.feature_data_loader import create_data_loaders
    print("✅ modules.models.lightweight_1dcnn 导入成功")
    print("✅ modules.models.model_trainer 导入成功")
    print("✅ modules.models.feature_data_loader 导入成功")
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {str(e)}")
    sys.exit(1)

# 【2/5】创建模型（文档5步骤2）
print(f"\n【2/5】创建模型...")
model = Lightweight1DCNN(input_features=20)
model.get_model_info()

# 统计参数量（文档5要求234,497）
params = model.count_parameters()
print(f"\n📊 参数统计:")
print(f"   总参数: {params['total']:,}")
print(f"   可训练参数: {params['trainable']:,}")
print(f"   目标: ≤500,000")
print(f"   状态: {'✅ Within limit（符合要求）' if params['total'] <= 500000 else '❌ 超出限制'}")

# 【3/5】创建模拟数据（文档5步骤3）
print(f"\n【3/5】创建模拟数据...")
def generate_valid_20d_features(n_samples=200):
    """生成文档要求的20维特征（RoR+HR+HRV+质量+频域）"""
    features_list = []
    for _ in range(n_samples):
        # 模拟2.12特征提取模块输出（文档特征向量化定义）
        features = {
            'valid': True,
            'vector': np.array([
                # 0-5: RoR特征（6维）
                2.0 + np.random.randn() * 0.5, 0.8 + np.random.randn() * 0.1,
                0.4 + np.random.randn() * 0.1, 1.5 + np.random.randn() * 0.2,
                4.0 + np.random.randn() * 0.3, 10 + np.random.randint(-1, 2),
                # 6-9: 心率特征（4维）
                75.0 + np.random.randn() * 3.0, 5.0 + np.random.randn() * 1.0,
                75.0 + np.random.randn() * 2.0, 15.0 + np.random.randn() * 2.0,
                # 10-12: HRV特征（3维）
                45.0 + np.random.randn() * 5.0, 35.0 + np.random.randn() * 4.0,
                15.0 + np.random.randn() * 3.0,
                # 13-16: 质量特征（4维）
                12.0 + np.random.randn() * 1.5, 0.1 + np.random.randn() * 0.05,
                -0.5 + np.random.randn() * 0.1, 0.05 + np.random.randn() * 0.01,
                # 17-19: 频域特征（3维）
                75.0 + np.random.randn() * 3.0, 0.7 + np.random.randn() * 0.05,
                4.5 + np.random.randn() * 0.3
            ], dtype=np.float32)
        }
        features_list.append(features)
    return features_list

# 生成200样本（文档5要求）
n_samples = 200
features_list = generate_valid_20d_features(n_samples)
# 生成SpO2标签（95-100，文档范围）
labels = np.random.uniform(95.0, 100.0, n_samples).astype(np.float32)
# 标签归一化（0-1，适配模型输出）
y_scaler = MinMaxScaler(feature_range=(0, 1))
labels_scaled = y_scaler.fit_transform(labels.reshape(-1, 1)).flatten()

# 打印数据信息（文档5格式）
print(f"✅ 数据集创建完成")
print(f"   训练集: {int(n_samples * 0.8)} 样本 | 特征维度: 20 | 标签范围: {labels.min():.2f} - {labels.max():.2f}")
print(f"   验证集: {int(n_samples * 0.2)} 样本 | 特征维度: 20 | 标签范围: {labels.min():.2f} - {labels.max():.2f}")

# 创建数据加载器（文档5要求批次大小16）
train_loader, val_loader = create_data_loaders(
    features_list=features_list,
    labels=labels_scaled,
    train_ratio=0.8,
    batch_size=16
)
print(f"✅ 模拟数据创建完成")
print(f"   样本数: {n_samples}")

# 【4/5】测试训练流程（文档5步骤4，5个epoch）
print(f"\n【4/5】测试训练流程（5个epoch）...")
trainer = ModelTrainer(
    model=model,
    device='cpu',
    learning_rate=0.001,
    output_dir=output_dir
)

# 开始训练（文档5配置）
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    y_scaler=y_scaler,
    epochs=5,
    early_stopping_patience=3,
    verbose=True
)

# 【5/5】保存结果（文档5步骤5）
print(f"\n【5/5】保存结果...")
trainer.plot_history(
    save_path=os.path.join(output_dir, "quick_test_history.png")
)

# 测试报告（文档5格式）
print(f"\n" + "=" * 60)
print("测试报告")
print("=" * 60)

print(f"\n【模型结构】")
print(f"  输入特征维度: 20")
print(f"  网络层数: 5层Conv1D + 2层FC")
print(f"  参数量: {params['total']:,} ({'✅符合' if params['total'] <= 500000 else '❌超出'})")

print(f"\n【训练测试】")
print(f"  训练轮数: {len(history['train_loss'])}")
print(f"  最终训练损失: {history['train_loss'][-1]:.4f}")
print(f"  最终验证损失: {history['val_loss'][-1]:.4f}")
print(f"  最佳验证MAE(SpO2): {trainer.best_val_mae_original:.4f}")

print(f"\n【输出文件】")
print(f"  模型文件: {os.path.join(output_dir, 'best_model.pth')}")
print(f"  训练图表: {os.path.join(output_dir, 'quick_test_history.png')}")

print(f"\n【结论】")
if params['total'] <= 500000 and trainer.best_val_mae_original <= 1.5:
    print(f"  轻量化模型搭建完成，可用于真实数据训练✅")
else:
    print(f"  模型需要调整⚠️")

print("=" * 60)
print(f"\n🎉 快速测试完成！")