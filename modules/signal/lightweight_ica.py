"""
轻量化ICA去噪模块 - lightweight_ica.py
功能：使用轻量化ICA算法去除信号噪声
算法：FastICA（轻量级实现）
应用场景：多通道信号去噪（可选）
性能目标：单次处理≤200ms
"""

import numpy as np
import time
from typing import Tuple, Optional
from scipy.linalg import eigh


class LightweightICA:
    """
    轻量化独立成分分析(ICA)

    核心功能：
    1. 盲源分离：从混合信号中分离独立成分
    2. 噪声去除：选择信号成分，去除噪声成分
    3. 轻量级实现：优化计算效率

    注意：ICA适用于多通道信号，对单通道效果有限
    """

    def __init__(
            self,
            n_components: Optional[int] = None,  # 独立成分数量
            max_iter: int = 200,  # 最大迭代次数
            tol: float = 1e-4,  # 收敛阈值
            random_state: Optional[int] = 42  # 随机种子
    ):
        """
        初始化ICA

        Args:
            n_components: 独立成分数量（None=自动）
            max_iter: 最大迭代次数
            tol: 收敛阈值
            random_state: 随机种子
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # 内部变量
        self.mixing_matrix = None  # 混合矩阵
        self.unmixing_matrix = None  # 分离矩阵
        self.mean = None
        self.whitening_matrix = None

        # 性能统计
        self.processing_count = 0
        self.total_time = 0.0

    def fit(self, X: np.ndarray) -> 'LightweightICA':
        """
        训练ICA模型

        Args:
            X: 输入信号矩阵 (n_samples, n_channels)

        Returns:
            self
        """
        start_time = time.time()

        # 验证输入
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # 确定成分数量
        if self.n_components is None:
            self.n_components = n_features

        # 中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 白化（Whitening）
        X_white, self.whitening_matrix = self._whiten(X_centered)

        # FastICA算法
        self.unmixing_matrix = self._fastica(X_white)

        # 计算混合矩阵
        self.mixing_matrix = np.linalg.pinv(self.unmixing_matrix)

        # 性能统计
        elapsed = time.time() - start_time
        self.processing_count += 1
        self.total_time += elapsed

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用ICA变换

        Args:
            X: 输入信号矩阵 (n_samples, n_channels)

        Returns:
            独立成分矩阵 (n_samples, n_components)
        """
        if self.unmixing_matrix is None:
            raise RuntimeError("请先调用fit()训练模型")

        # 中心化
        X_centered = X - self.mean

        # 白化
        X_white = np.dot(X_centered, self.whitening_matrix.T)

        # 应用分离矩阵
        sources = np.dot(X_white, self.unmixing_matrix.T)

        return sources

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        训练并变换

        Args:
            X: 输入信号矩阵

        Returns:
            独立成分矩阵
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """
        逆变换（从独立成分恢复原始信号）

        Args:
            sources: 独立成分矩阵

        Returns:
            恢复的信号矩阵
        """
        if self.mixing_matrix is None:
            raise RuntimeError("请先调用fit()训练模型")

        # 逆白化
        X_white = np.dot(sources, self.mixing_matrix.T)

        # 逆白化变换
        whitening_inv = np.linalg.pinv(self.whitening_matrix)
        X_centered = np.dot(X_white, whitening_inv.T)

        # 恢复均值
        X = X_centered + self.mean

        return X

    def denoise(
            self,
            X: np.ndarray,
            n_keep: Optional[int] = None,
            variance_threshold: float = 0.95
    ) -> np.ndarray:
        """
        去噪（保留主要成分）

        Args:
            X: 输入信号矩阵
            n_keep: 保留的成分数量（None=自动）
            variance_threshold: 方差累计阈值

        Returns:
            去噪后的信号
        """
        # 获取独立成分
        sources = self.fit_transform(X)

        # 计算每个成分的方差
        variances = np.var(sources, axis=0)

        # 如果未指定保留数量，根据方差阈值自动选择
        if n_keep is None:
            total_var = np.sum(variances)
            cumsum_var = np.cumsum(np.sort(variances)[::-1])
            n_keep = np.argmax(cumsum_var / total_var >= variance_threshold) + 1

        # 选择方差最大的n_keep个成分
        top_indices = np.argsort(variances)[::-1][:n_keep]

        # 创建去噪后的源信号（其他成分置零）
        sources_denoised = np.zeros_like(sources)
        sources_denoised[:, top_indices] = sources[:, top_indices]

        # 逆变换恢复信号
        X_denoised = self.inverse_transform(sources_denoised)

        return X_denoised

    def _whiten(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        白化（去除相关性）

        Args:
            X: 中心化后的信号

        Returns:
            X_white: 白化后的信号
            whitening_matrix: 白化矩阵
        """
        # 计算协方差矩阵
        cov = np.cov(X.T)

        # 特征值分解
        eigenvalues, eigenvectors = eigh(cov)

        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 选择前n_components个成分
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]

        # 白化矩阵
        whitening_matrix = (eigenvectors / np.sqrt(eigenvalues + 1e-8)).T

        # 应用白化
        X_white = np.dot(X, whitening_matrix.T)

        return X_white, whitening_matrix

    def _fastica(self, X_white: np.ndarray) -> np.ndarray:
        """
        FastICA算法核心

        Args:
            X_white: 白化后的信号

        Returns:
            分离矩阵
        """
        n_samples, n_components = X_white.shape

        # 初始化随机分离矩阵
        if self.random_state is not None:
            np.random.seed(self.random_state)

        W = np.random.randn(n_components, n_components)

        # 正交化
        W = self._symmetric_decorrelation(W)

        # 迭代优化
        for iteration in range(self.max_iter):
            # 计算非线性函数（使用tanh）
            gX = np.tanh(np.dot(X_white, W.T))
            g_prime = 1 - gX ** 2

            # 更新W
            W_new = np.dot(gX.T, X_white) / n_samples - \
                    np.dot(np.diag(np.mean(g_prime, axis=0)), W)

            # 正交化
            W_new = self._symmetric_decorrelation(W_new)

            # 检查收敛
            if np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1)) < self.tol:
                break

            W = W_new

        return W

    def _symmetric_decorrelation(self, W: np.ndarray) -> np.ndarray:
        """
        对称去相关

        Args:
            W: 矩阵

        Returns:
            正交化后的矩阵
        """
        # W * (W^T * W)^(-1/2)
        s, u = np.linalg.eigh(np.dot(W, W.T))
        W_orth = np.dot(np.dot(u * (1. / np.sqrt(s + 1e-8)), u.T), W)
        return W_orth

    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        if self.processing_count == 0:
            return {
                'total_processed': 0,
                'avg_time_ms': 0.0,
                'meets_target': False
            }

        avg_time_ms = (self.total_time / self.processing_count) * 1000

        return {
            'total_processed': self.processing_count,
            'avg_time_ms': round(avg_time_ms, 2),
            'meets_target': avg_time_ms <= 200
        }


# ===================== 测试代码 =====================
def test_lightweight_ica():
    """轻量化ICA测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')

    print("=" * 70)
    print("📝 轻量化ICA去噪测试")
    print("=" * 70)

    # 加载信号
    print("\n【1/3】加载原始信号")
    signal_file = "../../test_output/signal/rppg_signal_raw.npz"

    if not os.path.exists(signal_file):
        print(f"❌ 信号文件不存在: {signal_file}")
        return

    data = np.load(signal_file)

    # 构建多通道信号（RGB三通道）
    signals = np.column_stack([
        data['raw_R'],
        data['raw_G'],
        data['raw_B']
    ])

    print(f"✅ 成功加载信号")
    print(f"   信号形状: {signals.shape}")
    print(f"   (样本数, 通道数) = ({signals.shape[0]}, {signals.shape[1]})")

    # ICA去噪
    print("\n【2/3】应用ICA去噪")
    ica = LightweightICA(
        n_components=3,
        max_iter=200,
        random_state=42
    )

    # 去噪（保留前2个主成分）
    signals_denoised = ica.denoise(signals, n_keep=2)

    print(f"✅ ICA去噪完成")
    print(f"   保留成分: 2/3")

    # 性能统计
    print("\n【3/3】性能统计")
    stats = ica.get_performance_stats()
    print(f"   处理次数: {stats['total_processed']}")
    print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

    # 保存结果
    output_dir = "../../test_output/signal"
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        f"{output_dir}/ica_denoised.npz",
        original=signals,
        denoised=signals_denoised
    )
    print(f"\n✅ 去噪结果已保存: {output_dir}/ica_denoised.npz")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    print("\n💡 注意: ICA适用于多通道信号，对单通道CHROM信号不适用")
    print("   推荐使用带通滤波+去趋势进行单通道信号预处理")


if __name__ == "__main__":
    test_lightweight_ica()