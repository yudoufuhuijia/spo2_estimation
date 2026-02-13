import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_training_trend():
    """从已保存的预测文件中读取训练历史，生成趋势图（修复版）"""

    # 1. 配置绝对路径（关键！确保图片生成在正确的目录）
    # 获取当前脚本所在目录，然后拼接出model_output的绝对路径
    project_root = Path(__file__).parent.resolve()
    predictions_path = project_root / "model_output" / "predictions.npz"
    save_path = project_root / "model_output" / "training_trend.png"

    # 2. 读取已保存的训练历史
    if not predictions_path.exists():
        print(f"❌ 未找到文件: {predictions_path}")
        print("💡 请确认路径是否正确，或已完成训练")
        return

    data = np.load(predictions_path, allow_pickle=True)
    history = data['history'].item()  # 提取训练历史字典

    # 3. 配置绘图（使用英文标签，避免字体问题）
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.figure(figsize=(12, 5))

    # 4. 绘制损失曲线（左图）
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=1.5)
    plt.title('Training/Validation Loss Trend', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('MAE Loss', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    # 5. 绘制MAE曲线（右图）
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=1.5)
    plt.plot(epochs, history['val_mae'], 'r--', label='Val MAE', linewidth=1.5)
    plt.title('Training/Validation MAE Trend', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('MAE (%)', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

    # 6. 保存图片（使用绝对路径）
    plt.tight_layout()  # 自动调整布局，避免重叠
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 训练趋势图已生成！路径: {save_path}")
    print("\n📊 Trend Interpretation:")
    print(f"   - Training Epochs: {len(epochs)}")
    print(f"   - Final Train MAE: {history['train_mae'][-1]:.4f}%")
    print(f"   - Final Val MAE: {history['val_mae'][-1]:.4f}%")
    print(f"   - Best Val Loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    plot_training_trend()