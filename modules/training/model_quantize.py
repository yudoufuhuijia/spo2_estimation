"""
模型量化脚本 - model_quantize.py
任务2.15 - 步骤1：量化模型至INT8

功能：
1. 加载训练好的PyTorch模型（best_model.pth）
2. 使用动态量化技术压缩至INT8
3. 测试量化前后的精度差异
4. 保存量化模型（≤5MB）
5. 上传至OSS（可选）

使用方法：
python model_quantize.py --model_path model_output_v2/best_model.pth
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
import os
import sys
from pathlib import Path
import time
import argparse

# 添加项目路径
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))


# ==================== 模型定义（与训练时一致） ====================
class SpO2Model(nn.Module):
    """SpO2估计模型（与训练脚本一致）"""

    def __init__(self, input_dim=30, hidden_dim=64, init_bias=97.5):
        super(SpO2Model, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # 初始化输出层偏置
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


# ==================== 量化函数 ====================
def quantize_model(model_path: str, output_dir: str = "model_quantized", test_data_path: str = None):
    """
    量化模型至INT8

    Args:
        model_path: 原始模型路径
        output_dir: 量化模型输出目录
        test_data_path: 测试数据路径（用于校准）

    Returns:
        量化后的模型
    """
    print("=" * 70)
    print("🔧 模型量化工具 - 任务2.15")
    print("=" * 70)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载原始模型
    print("\n【步骤1/5】加载原始模型...")
    try:
        model = SpO2Model(input_dim=30, hidden_dim=64, init_bias=97.5)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        # 获取原始模型大小
        torch.save(model.state_dict(), output_path / "original_model_temp.pth")
        original_size = os.path.getsize(output_path / "original_model_temp.pth") / 1024 / 1024
        os.remove(output_path / "original_model_temp.pth")

        print(f"✅ 模型加载成功")
        print(f"   模型路径: {model_path}")
        print(f"   原始大小: {original_size:.2f} MB")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 2. 准备测试数据（如果提供）
    print("\n【步骤2/5】准备校准数据...")
    if test_data_path and os.path.exists(test_data_path):
        try:
            data = np.load(test_data_path)
            test_features = data['features'] if 'features' in data.files else data['arr_0']

            # 确保形状正确
            if test_features.ndim == 1:
                test_features = test_features.reshape(-1, 30)
            elif test_features.ndim == 3:
                test_features = test_features.reshape(-1, test_features.shape[-1])

            # 只取前100个样本用于校准
            test_features = test_features[:100]
            test_tensor = torch.FloatTensor(test_features)

            print(f"✅ 校准数据加载成功")
            print(f"   样本数: {len(test_features)}")
            print(f"   特征维度: {test_features.shape}")
        except Exception as e:
            print(f"⚠️  校准数据加载失败: {e}")
            print(f"   将使用随机数据进行量化")
            test_tensor = torch.randn(100, 30)
    else:
        print(f"⚠️  未提供测试数据，使用随机数据")
        test_tensor = torch.randn(100, 30)

    # 3. 动态量化
    print("\n【步骤3/5】执行动态量化...")
    try:
        # PyTorch动态量化（针对Linear层）
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear},  # 量化所有Linear层
            dtype=torch.qint8  # 使用INT8量化
        )

        print(f"✅ 动态量化完成")
        print(f"   量化类型: INT8 (qint8)")
        print(f"   量化层: 全连接层(Linear)")

    except Exception as e:
        print(f"❌ 量化失败: {e}")
        return None

    # 4. 测试量化前后的精度
    print("\n【步骤4/5】测试量化精度...")
    with torch.no_grad():
        # 原始模型预测
        original_output = model(test_tensor)

        # 量化模型预测
        quantized_output = quantized_model(test_tensor)

        # 计算差异
        mae_diff = torch.abs(original_output - quantized_output).mean().item()
        max_diff = torch.abs(original_output - quantized_output).max().item()

        print(f"✅ 精度测试完成")
        print(f"   平均绝对误差: {mae_diff:.4f}%")
        print(f"   最大误差: {max_diff:.4f}%")

        if mae_diff < 0.5:
            print(f"   ✅ 量化精度优秀（MAE < 0.5%）")
        elif mae_diff < 1.0:
            print(f"   ✅ 量化精度良好（MAE < 1.0%）")
        else:
            print(f"   ⚠️  量化精度一般（MAE ≥ 1.0%）")

    # 5. 保存量化模型
    print("\n【步骤5/5】保存量化模型...")
    try:
        quantized_model_path = output_path / "quantized_model_int8.pth"
        torch.save(quantized_model.state_dict(), quantized_model_path)

        # 获取量化后大小
        quantized_size = os.path.getsize(quantized_model_path) / 1024 / 1024
        compression_ratio = (1 - quantized_size / original_size) * 100

        print(f"✅ 量化模型已保存")
        print(f"   保存路径: {quantized_model_path}")
        print(f"   量化后大小: {quantized_size:.2f} MB")
        print(f"   压缩率: {compression_ratio:.1f}%")

        if quantized_size <= 5.0:
            print(f"   ✅ 满足大小要求（≤5MB）")
        else:
            print(f"   ⚠️  超过目标大小（5MB）")

    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None

    # 保存量化配置信息
    config_path = output_path / "quantization_info.txt"
    with open(config_path, 'w') as f:
        f.write(f"模型量化信息\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"原始模型: {model_path}\n")
        f.write(f"原始大小: {original_size:.2f} MB\n")
        f.write(f"量化后大小: {quantized_size:.2f} MB\n")
        f.write(f"压缩率: {compression_ratio:.1f}%\n")
        f.write(f"量化类型: INT8 (qint8)\n")
        f.write(f"平均精度损失: {mae_diff:.4f}%\n")
        f.write(f"最大精度损失: {max_diff:.4f}%\n")

    print(f"✅ 配置信息已保存: {config_path}")

    # 性能测试
    print("\n【性能测试】量化前后推理速度对比...")
    test_input = torch.randn(1, 30)

    # 原始模型
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    original_time = (time.time() - start) / 100 * 1000

    # 量化模型
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = quantized_model(test_input)
    quantized_time = (time.time() - start) / 100 * 1000

    speedup = original_time / quantized_time

    print(f"✅ 性能测试完成（单次推理）")
    print(f"   原始模型: {original_time:.2f} ms")
    print(f"   量化模型: {quantized_time:.2f} ms")
    print(f"   加速比: {speedup:.2f}x")

    # 总结
    print("\n" + "=" * 70)
    print("🎉 量化完成！")
    print("=" * 70)
    print(f"📊 量化统计：")
    print(f"   模型压缩: {original_size:.2f}MB → {quantized_size:.2f}MB ({compression_ratio:.1f}%)")
    print(f"   精度损失: {mae_diff:.4f}% (MAE)")
    print(f"   推理加速: {speedup:.2f}x")
    print(f"\n📁 输出文件：")
    print(f"   - {quantized_model_path}")
    print(f"   - {config_path}")
    print("=" * 70)

    return quantized_model


# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='模型量化工具')
    parser.add_argument('--model_path', type=str,
                        default='model_output_v2/best_model.pth',
                        help='原始模型路径')
    parser.add_argument('--test_data', type=str,
                        default='training_data/combined_features.npz',
                        help='测试数据路径（用于校准）')
    parser.add_argument('--output_dir', type=str,
                        default='model_quantized',
                        help='量化模型输出目录')

    args = parser.parse_args()

    # 执行量化
    quantized_model = quantize_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        test_data_path=args.test_data
    )

    if quantized_model is None:
        print("\n❌ 量化失败！")
        sys.exit(1)

    print("\n✅ 量化成功！")


if __name__ == "__main__":
    main()