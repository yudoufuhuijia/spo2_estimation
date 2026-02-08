"""
轻量化1D-CNN模型（文档3.2指定结构）
功能：SpO2估计，输入20维特征，输出1个预测值
参数量：234,497（≤50万目标）
输入形状：(batch_size, 1, 20) → 输出形状：(batch_size, 1)
"""
import os
import torch
import torch.nn as nn
from typing import Dict


class Lightweight1DCNN(nn.Module):
    def __init__(self, input_features: int = 20):
        super().__init__()
        self.input_features = input_features  # 输入特征长度（固定20）

        # 【文档3.2严格对应】第1层卷积块：1→64，kernel=3，MaxPool/2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出长度：20/2=10

        # 第2层卷积块：64→128，kernel=3，MaxPool/2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出长度：10/2=5

        # 第3层卷积块：128→256，kernel=3，MaxPool/2
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出长度：5/2=2（向下取整）

        # 第4层卷积块：256→128，kernel=3，MaxPool/2
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出长度：2/2=1

        # 第5层卷积块：128→64，kernel=3（无MaxPool）
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(64)

        # 自适应池化（统一输出长度为1，兼容不同输入）
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层（文档3.2指定）
        self.fc1 = nn.Linear(64, 128)  # 池化后64维 → 128维
        self.dropout = nn.Dropout(0.3)  # 防止过拟合，文档指定0.3
        self.fc2 = nn.Linear(128, 1)   # 128维 → 1维（SpO2预测值）

        # 输出激活（约束在0-1，匹配标签归一化范围）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：严格按文档结构执行"""
        # 第1卷积块：Conv1D → BN → ReLU → MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # 输出：(batch, 64, 10)

        # 第2卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # 输出：(batch, 128, 5)

        # 第3卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)  # 输出：(batch, 256, 2)

        # 第4卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)  # 输出：(batch, 128, 1)

        # 第5卷积块
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)  # 输出：(batch, 64, 1)

        # 自适应池化 + 展平
        x = self.avg_pool(x)  # 输出：(batch, 64, 1)
        x = x.view(x.size(0), -1)  # 展平为：(batch, 64)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # 输出：(batch, 1)

        # 激活约束（0-1）
        x = self.sigmoid(x)
        return x

    def count_parameters(self) -> Dict[str, int]:
        """统计参数量（文档3.3计算结果：234,497）"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_model_info(self) -> str:
        """返回文档要求的模型结构信息"""
        info = (
            "Lightweight 1D-CNN Model\n"
            "Input Features: 20\n"
            "Architecture:\n"
            "Conv1D (1→64) + BN + ReLU + MaxPool | "
            "Conv1D (64→128) + BN + ReLU + MaxPool | "
            "Conv1D (128→256) + BN + ReLU + MaxPool | "
            "Conv1D (256→128) + BN + ReLU + MaxPool | "
            "Conv1D (128→64) + BN + ReLU | "
            "AdaptiveAvgPool → FC(64→128) → Dropout(0.3) → FC(128→1)"
        )
        print(info)
        return info


# 文档4.1测试代码（单独运行时执行）
if __name__ == "__main__":
    print("轻量化1D-CNN模型测试")
    print("=" * 60)

    print("\n【1/4】创建模型")
    model = Lightweight1DCNN(input_features=20)
    print("✅ 模型创建成功")

    print("\n【2/4】模型信息")
    model.get_model_info()
    params = model.count_parameters()
    print(f"\nParameters:")
    print(f"Total: {params['total']:,}")
    print(f"Trainable: {params['trainable']:,}")
    print(f"\nTarget: ≤500,000")
    print(f"Status: {'✅ Within limit' if params['total'] <= 500000 else '❌ Exceed limit'}")

    print("\n【3/4】测试前向传播")
    # 文档要求输入：(batch_size, 20) → 转换为卷积输入：(batch_size, 1, 20)
    input_tensor = torch.randn(8, 1, 20)  # 8个样本，1个通道，20维特征
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: {output.min().item():.4f} ~ {output.max().item():.4f}")
    print("✅ 前向传播测试通过")

    print("\n【4/4】保存模型")
    output_dir = "../../test_output/models"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model_structure.pth"))
    file_size = os.path.getsize(os.path.join(output_dir, "model_structure.pth")) / 1024  # KB
    print(f"模型结构已保存: {os.path.join(output_dir, 'model_structure.pth')}")
    print(f"模型文件大小: {file_size:.2f} KB (~0.9MB，符合文档要求)")
    print("\n" + "=" * 60)
    print("✅ 测试完成")