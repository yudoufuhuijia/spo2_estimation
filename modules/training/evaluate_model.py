"""
模型评估与可视化 - evaluate_model.py
功能：详细评估训练好的模型，生成可视化报告
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

from modules.models.lightweight_1dcnn import Lightweight1DCNN


class ModelEvaluator:
    """
    模型评估器

    功能：
    1. 加载训练好的模型
    2. 详细性能评估
    3. 误差分析
    4. 可视化结果
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化评估器

        Args:
            model_path: 模型文件路径
            device: 设备
        """
        self.model_path = Path(model_path)
        self.device = device

        # 加载模型
        self.model = self._load_model()

        print(f"✅ 模型评估器初始化完成")
        print(f"   模型: {model_path}")

    def _load_model(self) -> Lightweight1DCNN:
        """加载模型"""
        # 创建模型
        model = Lightweight1DCNN(input_features=20)

        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def evaluate(
            self,
            predictions: np.ndarray,
            labels: np.ndarray
    ) -> Dict:
        """
        详细评估

        Args:
            predictions: 预测值
            labels: 真实值

        Returns:
            评估指标字典
        """
        print(f"\n📊 详细评估")
        print("=" * 70)

        # 基本统计
        print(f"\n【数据统计】")
        print(f"  样本数: {len(labels)}")
        print(f"  真实值范围: {labels.min():.2f} - {labels.max():.2f}")
        print(f"  预测值范围: {predictions.min():.2f} - {predictions.max():.2f}")

        # 误差统计
        errors = predictions - labels
        abs_errors = np.abs(errors)

        print(f"\n【误差统计】")
        print(f"  平均误差: {np.mean(errors):.4f}")
        print(f"  误差标准差: {np.std(errors):.4f}")
        print(f"  最大正误差: {np.max(errors):.4f}")
        print(f"  最大负误差: {np.min(errors):.4f}")

        # 评估指标
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        mape = np.mean(abs_errors / labels) * 100

        print(f"\n【评估指标】")
        print(f"  MAE (平均绝对误差): {mae:.4f}")
        print(f"  RMSE (均方根误差): {rmse:.4f}")
        print(f"  R² (决定系数): {r2:.4f}")
        print(f"  MAPE (平均绝对百分比误差): {mape:.4f}%")

        # 误差分布
        within_1 = np.sum(abs_errors < 1.0) / len(abs_errors) * 100
        within_2 = np.sum(abs_errors < 2.0) / len(abs_errors) * 100
        within_3 = np.sum(abs_errors < 3.0) / len(abs_errors) * 100

        print(f"\n【误差分布】")
        print(f"  误差<1%: {within_1:.2f}%")
        print(f"  误差<2%: {within_2:.2f}%")
        print(f"  误差<3%: {within_3:.2f}%")

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'within_1': within_1,
            'within_2': within_2,
            'within_3': within_3,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

        return metrics

    def plot_results(
            self,
            predictions: np.ndarray,
            labels: np.ndarray,
            save_dir: str = "test_output/models"
    ):
        """
        可视化结果

        Args:
            predictions: 预测值
            labels: 真实值
            save_dir: 保存目录
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 1. 预测vs真实（散点图）
            axes[0, 0].scatter(labels, predictions, alpha=0.5, s=30)
            axes[0, 0].plot([labels.min(), labels.max()],
                            [labels.min(), labels.max()],
                            'r--', linewidth=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('True SpO2 (%)', fontsize=12)
            axes[0, 0].set_ylabel('Predicted SpO2 (%)', fontsize=12)
            axes[0, 0].set_title('Predictions vs True Values', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. 误差分布（直方图）
            errors = predictions - labels
            axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            axes[0, 1].set_xlabel('Prediction Error (%)', fontsize=12)
            axes[0, 1].set_ylabel('Frequency', fontsize=12)
            axes[0, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 残差图
            axes[1, 0].scatter(predictions, errors, alpha=0.5, s=30)
            axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_xlabel('Predicted SpO2 (%)', fontsize=12)
            axes[1, 0].set_ylabel('Residuals (%)', fontsize=12)
            axes[1, 0].set_title('Residual Plot', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. 累积误差分布
            sorted_abs_errors = np.sort(np.abs(errors))
            cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100

            axes[1, 1].plot(sorted_abs_errors, cumulative, linewidth=2)
            axes[1, 1].axvline(1.0, color='red', linestyle='--', alpha=0.7, label='1% Error')
            axes[1, 1].axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='2% Error')
            axes[1, 1].axvline(3.0, color='yellow', linestyle='--', alpha=0.7, label='3% Error')
            axes[1, 1].set_xlabel('Absolute Error (%)', fontsize=12)
            axes[1, 1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
            axes[1, 1].set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xlim([0, 5])

            plt.tight_layout()

            # 保存图表
            plot_path = save_dir / "evaluation_plots.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n✅ 评估图表已保存: {plot_path}")

        except ImportError:
            print(f"\n⚠️  matplotlib未安装，跳过可视化")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--model', type=str,
                        default='test_output/models/best_model.pth',
                        help='模型文件路径')
    parser.add_argument('--predictions', type=str,
                        default='test_output/models/predictions.npz',
                        help='预测结果文件')

    args = parser.parse_args()

    print("=" * 70)
    print("SpO2估计模型评估")
    print("=" * 70)

    # 加载预测结果
    print(f"\n【加载数据】")
    data = np.load(args.predictions)
    predictions = data['predictions']
    labels = data['labels']

    print(f"✅ 加载完成")
    print(f"   样本数: {len(labels)}")

    # 创建评估器
    evaluator = ModelEvaluator(args.model)

    # 评估
    metrics = evaluator.evaluate(predictions, labels)

    # 可视化
    evaluator.plot_results(predictions, labels)

    print(f"\n{'=' * 70}")
    print(f"评估完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()