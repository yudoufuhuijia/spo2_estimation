"""
模型推理脚本 - model_infer.py
任务2.15 - 步骤2：优化推理（关闭梯度，批量推理）
功能：
1. 加载量化模型/原始模型
2. 从OSS批量加载数据
3. 优化推理（torch.no_grad()）
4. 统计推理延迟
5. 保存推理结果
使用方法：
# 使用量化模型推理
python model_infer.py --model_path model_quantized/quantized_model_int8.pth --quantized
# 使用原始模型推理
python model_infer.py --model_path model_output_v2/best_model.pth
"""
import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
import os
import sys
import time
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

# 添加项目路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# ==================== 模型定义 ====================
class SpO2Model(nn.Module):
    """SpO2估计模型"""
    def __init__(self, input_dim=30, hidden_dim=64, init_bias=97.5):
        super(SpO2Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        nn.init.constant_(self.fc3.bias, init_bias)

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get('combined_features', x.get('ror_features', x.get('raw_features')))
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if not self.training:
            x = torch.clamp(x, 85, 100)
        return x

# ==================== 推理器类 ====================
class SpO2Inferencer:
    """优化的SpO2推理器（修复参数量统计+计时失真）"""
    def __init__(self, model_path: str, is_quantized: bool = False, device: str = 'cpu'):
        """
        初始化推理器
        Args:
            model_path: 模型路径
            is_quantized: 是否为量化模型
            device: 设备（cpu/cuda）
        """
        self.model_path = model_path
        self.is_quantized = is_quantized
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型（修复量化模型参数量统计）"""
        print(f"\n{'=' * 70}")
        print(f"加载模型")
        print(f"{'=' * 70}")
        try:
            # 1. 创建原始模型结构
            self.model = SpO2Model(input_dim=30, hidden_dim=64, init_bias=97.5)

            # 2. 若为量化模型，执行动态量化（与model_quantize.py逻辑一致）
            if self.is_quantized:
                self.model = quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},  # 仅量化全连接层
                    dtype=torch.qint8  # INT8量化类型
                )

            # 3. 加载模型权重
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            # 4. 设置为评估模式并移至目标设备
            self.model.eval()
            self.model.to(self.device)

            # 修复：量化模型参数量统计（适配打包参数）
            original_model = SpO2Model(input_dim=30, hidden_dim=64, init_bias=97.5)
            total_params = sum(p.numel() for p in original_model.parameters())
            for param in self.model.parameters():
                # 处理量化层的打包参数（qint8 Linear层）
                if hasattr(param, '_packed_params'):
                    packed_params = param._packed_params
                    # 统计权重数量
                    if hasattr(packed_params, 'weight'):
                        total_params += packed_params.weight().numel()
                    # 统计偏置数量（若存在）
                    if hasattr(packed_params, 'bias') and packed_params.bias() is not None:
                        total_params += packed_params.bias().numel()
                else:
                    # 非量化层直接统计参数
                    total_params += param.numel()

            # 计算模型大小
            model_size = os.path.getsize(self.model_path) / 1024 / 1024
            print(f"✅ 模型加载成功")
            print(f"   路径: {self.model_path}")
            print(f"   类型: {'量化模型(INT8)' if self.is_quantized else '原始模型(FP32)'}")
            print(f"   大小: {model_size:.2f} MB")
            print(f"   设备: {self.device}")
            print(f"   参数量: {total_params:,}")  # 修复后的参数量统计
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def infer_single(self, features: np.ndarray) -> float:
        """
        单样本推理（优化版）
        Args:
            features: 特征向量 (30,)
        Returns:
            SpO2预测值
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        input_tensor = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        prediction = output.cpu().numpy()
        if prediction.ndim > 0:
            prediction = prediction[0]
        return float(prediction)

    def infer_batch(self, features: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        批量推理（优化版，添加高精度耗时打印）
        Args:
            features: 特征矩阵 (N, 30)
            batch_size: 批次大小
        Returns:
            SpO2预测值数组 (N,)
        """
        n_samples = len(features)
        predictions = []
        # 高精度计时（用于中间打印）
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_features = features[i:i + batch_size]
                input_tensor = torch.FloatTensor(batch_features).to(self.device)
                batch_output = self.model(input_tensor)
                batch_predictions = batch_output.cpu().numpy()
                predictions.extend(batch_predictions)
        # 打印单次批量推理真实耗时（微秒级）
        batch_time = (time.perf_counter() - start_time) * 1000
        print(f"真实批量推理耗时: {batch_time:.2f} ms")
        return np.array(predictions, dtype=np.float32)

    def benchmark(self, features: np.ndarray, n_runs: int = 100) -> Dict:
        """
        性能基准测试
        Args:
            features: 测试特征 (可以是单个样本或批量)
            n_runs: 运行次数
        Returns:
            性能统计字典
        """
        print(f"\n{'=' * 70}")
        print(f"性能基准测试")
        print(f"{'=' * 70}")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        batch_size = len(features)

        # 预热模型
        print(f"预热模型...")
        for _ in range(10):
            _ = self.infer_batch(features, batch_size=batch_size)

        # 运行性能测试
        print(f"运行性能测试（{n_runs}次）...")
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()  # 高精度计时
            _ = self.infer_batch(features, batch_size=batch_size)
            latency = (time.perf_counter() - start) * 1000  # 转换为毫秒
            latencies.append(latency)

        # 统计性能指标
        latencies = np.array(latencies)
        stats = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'batch_size': batch_size,
            'n_runs': n_runs
        }
        per_sample_latency = stats['mean'] / batch_size

        # 输出统计结果
        print(f"\n📊 性能统计（批量={batch_size}）：")
        print(f"   平均延迟: {stats['mean']:.2f} ms")
        print(f"   标准差: {stats['std']:.2f} ms")
        print(f"   最小值: {stats['min']:.2f} ms")
        print(f"   最大值: {stats['max']:.2f} ms")
        print(f"   P50: {stats['p50']:.2f} ms")
        print(f"   P95: {stats['p95']:.2f} ms")
        print(f"   P99: {stats['p99']:.2f} ms")
        print(f"\n📊 单样本延迟：")
        print(f"   平均: {per_sample_latency:.2f} ms/样本")
        if per_sample_latency < 100:
            print(f"   ✅ 满足要求（<100ms）")
        else:
            print(f"   ⚠️  超过目标（100ms）")
        stats['per_sample_latency'] = per_sample_latency
        return stats

    def load_and_infer_from_file(self, data_path: str, batch_size: int = 32, save_results: bool = True):
        """
        从文件加载数据并推理（修复总时间计时失真）
        Args:
            data_path: 数据文件路径
            batch_size: 批次大小
            save_results: 是否保存结果
        """
        print(f"\n{'=' * 70}")
        print(f"批量推理")
        print(f"{'=' * 70}")

        # 加载数据
        print(f"\n加载数据...")
        try:
            data = np.load(data_path)
            features = data['features'] if 'features' in data.files else data['arr_0']
            if features.ndim == 1:
                features = features.reshape(-1, 30)
            elif features.ndim == 3:
                features = features.reshape(-1, features.shape[-1])
            print(f"✅ 数据加载成功")
            print(f"   样本数: {len(features)}")
            print(f"   特征维度: {features.shape}")
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None

        # 执行批量推理（修复：高精度计时统计总时间）
        print(f"\n执行批量推理（批次大小={batch_size}）...")
        start_time = time.perf_counter()  # 替换time.time()，微秒级精度
        predictions = self.infer_batch(features, batch_size=batch_size)
        total_time = time.perf_counter() - start_time  # 真实总耗时
        avg_time = total_time / len(features) * 1000  # 转换为ms/样本

        # 输出修复后的真实统计结果
        print(f"✅ 推理完成")
        print(f"   总时间: {total_time:.4f} 秒")  # 保留4位小数，避免0.00显示
        print(f"   平均延迟: {avg_time:.2f} ms/样本")
        print(f"   吞吐量: {len(features) / total_time:.1f} 样本/秒")

        # 预测统计
        print(f"\n📊 预测统计：")
        print(f"   预测范围: {predictions.min():.2f} ~ {predictions.max():.2f}%")
        print(f"   预测均值: {predictions.mean():.2f}%")
        print(f"   预测标准差: {predictions.std():.2f}%")

        # 保存结果
        if save_results:
            output_dir = Path("inference_results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"predictions_{'quantized' if self.is_quantized else 'original'}.npz"
            np.savez(
                output_path,
                predictions=predictions,
                features=features,
                latency_ms=avg_time
            )
            print(f"\n✅ 结果已保存: {output_path}")
        return predictions

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='SpO2模型推理工具')
    parser.add_argument('--model_path', type=str,
                        default='model_output_v2/best_model.pth',
                        help='模型路径')
    parser.add_argument('--quantized', action='store_true',
                        help='是否为量化模型')
    parser.add_argument('--data_path', type=str,
                        default='training_data/combined_features.npz',
                        help='推理数据路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--benchmark', action='store_true',
                        help='运行性能基准测试')
    parser.add_argument('--n_runs', type=int, default=100,
                        help='基准测试运行次数')
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 SpO2模型推理工具 - 任务2.15")
    print("=" * 70)

    # 创建推理器
    inferencer = SpO2Inferencer(
        model_path=args.model_path,
        is_quantized=args.quantized,
        device='cpu'
    )

    # 运行基准测试
    if args.benchmark:
        if os.path.exists(args.data_path):
            data = np.load(args.data_path)
            test_features = data['features'] if 'features' in data.files else data['arr_0']
            if test_features.ndim == 1:
                test_features = test_features.reshape(-1, 30)
            elif test_features.ndim == 3:
                test_features = test_features.reshape(-1, test_features.shape[-1])
            test_features = test_features[:10]  # 取前10个样本测试
        else:
            test_features = np.random.randn(10, 30).astype(np.float32)

        # 执行基准测试
        stats = inferencer.benchmark(test_features, n_runs=args.n_runs)

        # 保存基准测试结果
        output_dir = Path("inference_results")
        output_dir.mkdir(exist_ok=True)
        stats_file = output_dir / f"benchmark_{'quantized' if args.quantized else 'original'}.txt"
        with open(stats_file, 'w') as f:
            f.write("性能基准测试结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"类型: {'量化模型' if args.quantized else '原始模型'}\n")
            f.write(f"批次大小: {stats['batch_size']}\n")
            f.write(f"运行次数: {stats['n_runs']}\n")
            f.write(f"\n延迟统计（批量）:\n")
            f.write(f"  平均: {stats['mean']:.2f} ms\n")
            f.write(f"  标准差: {stats['std']:.2f} ms\n")
            f.write(f"  最小: {stats['min']:.2f} ms\n")
            f.write(f"  最大: {stats['max']:.2f} ms\n")
            f.write(f"  P50: {stats['p50']:.2f} ms\n")
            f.write(f"  P95: {stats['p95']:.2f} ms\n")
            f.write(f"  P99: {stats['p99']:.2f} ms\n")
            f.write(f"\n单样本延迟:\n")
            f.write(f"  平均: {stats['per_sample_latency']:.2f} ms\n")
        print(f"\n✅ 基准测试结果已保存: {stats_file}")

    # 执行批量推理
    else:
        predictions = inferencer.load_and_infer_from_file(
            data_path=args.data_path,
            batch_size=args.batch_size,
            save_results=True
        )

    print("\n" + "=" * 70)
    print("🎉 推理完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()