"""
rPPG信号提取模块 - chrom_extractor.py
功能：基于CHROM算法从ROI提取原始rPPG信号
算法：Chrominance-based (CHROM) - De Haan & Jeanne, 2013
性能目标：单帧信号提取≤5ms
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque


class CHROMExtractor:
    """
    CHROM (Chrominance-based) rPPG信号提取器

    核心原理：
    1. 从ROI区域提取RGB通道均值
    2. 使用色度差分消除运动伪影
    3. 生成原始脉搏波信号

    算法公式：
    Xs = 3*R - 2*G
    Ys = 1.5*R + G - 1.5*B
    S = Xs - α*Ys  (α = σ(Xs)/σ(Ys))
    """

    def __init__(
            self,
            fps: int = 30,  # 视频帧率
            window_size: int = 300,  # 滑动窗口大小（帧数，默认10秒）
            min_signal_length: int = 60,  # 最小信号长度（2秒）
            use_forehead_only: bool = True  # 仅使用额头ROI（推荐）
    ):
        """
        初始化CHROM提取器

        Args:
            fps: 视频帧率（用于信号处理）
            window_size: 滑动窗口大小（帧数）
            min_signal_length: 最小信号长度（帧数）
            use_forehead_only: 是否仅使用额头ROI
        """
        self.fps = fps
        self.window_size = window_size
        self.min_signal_length = min_signal_length
        self.use_forehead_only = use_forehead_only

        # 信号缓冲区（使用deque实现滑动窗口）
        self.rgb_buffer = deque(maxlen=window_size)

        # 原始RGB信号
        self.raw_signals = {
            'R': deque(maxlen=window_size),
            'G': deque(maxlen=window_size),
            'B': deque(maxlen=window_size)
        }

        # CHROM信号
        self.chrom_signal = deque(maxlen=window_size)

        # 性能统计
        self.extraction_count = 0
        self.total_time = 0.0

        # 时间戳
        self.timestamps = deque(maxlen=window_size)

    def extract_roi_signal(
            self,
            roi_image: np.ndarray,
            timestamp: Optional[float] = None
    ) -> Optional[float]:
        """
        从单个ROI图像提取RGB均值

        Args:
            roi_image: ROI图像（BGR格式）
            timestamp: 时间戳（秒）

        Returns:
            当前帧的CHROM信号值（如果缓冲区足够）
        """
        start_time = time.time()

        # 验证输入
        if roi_image is None or roi_image.size == 0:
            return None

        # 提取RGB通道均值
        # OpenCV使用BGR顺序，需要转换
        b_mean = np.mean(roi_image[:, :, 0])
        g_mean = np.mean(roi_image[:, :, 1])
        r_mean = np.mean(roi_image[:, :, 2])

        # 归一化（避免数值不稳定）
        total = r_mean + g_mean + b_mean
        if total == 0:
            return None

        r_norm = r_mean / total
        g_norm = g_mean / total
        b_norm = b_mean / total

        # 添加到缓冲区
        self.raw_signals['R'].append(r_norm)
        self.raw_signals['G'].append(g_norm)
        self.raw_signals['B'].append(b_norm)

        # 记录时间戳
        if timestamp is None:
            timestamp = time.time()
        self.timestamps.append(timestamp)

        # 计算CHROM信号（需要足够的历史数据）
        chrom_value = None
        if len(self.raw_signals['R']) >= self.min_signal_length:
            chrom_value = self._compute_chrom()
            self.chrom_signal.append(chrom_value)

        # 更新性能统计
        elapsed = time.time() - start_time
        self.extraction_count += 1
        self.total_time += elapsed

        return chrom_value

    def extract_from_rois(
            self,
            rois: Dict[str, np.ndarray],
            timestamp: Optional[float] = None
    ) -> Optional[float]:
        """
        从ROI字典提取信号

        Args:
            rois: ROI字典（2.9模块输出）
                {
                    'forehead': np.ndarray,
                    'left_cheek': np.ndarray,
                    'right_cheek': np.ndarray
                }
            timestamp: 时间戳

        Returns:
            CHROM信号值
        """
        if self.use_forehead_only:
            # 仅使用额头ROI（推荐，信号质量最好）
            if 'forehead' in rois:
                return self.extract_roi_signal(rois['forehead'], timestamp)
            else:
                return None
        else:
            # 使用多个ROI的平均（可选）
            signals = []

            for roi_name in ['forehead', 'left_cheek', 'right_cheek']:
                if roi_name in rois:
                    roi_signal = self.extract_roi_signal(rois[roi_name], timestamp)
                    if roi_signal is not None:
                        signals.append(roi_signal)

            if signals:
                return np.mean(signals)
            else:
                return None

    def _compute_chrom(self) -> float:
        """
        计算CHROM信号（当前帧）

        CHROM算法核心：
        1. Xs = 3*R - 2*G
        2. Ys = 1.5*R + G - 1.5*B
        3. α = σ(Xs) / σ(Ys)
        4. S = Xs - α*Ys

        Returns:
            当前帧的CHROM信号值
        """
        # 获取最近的RGB数据（滑动窗口）
        R = np.array(self.raw_signals['R'])
        G = np.array(self.raw_signals['G'])
        B = np.array(self.raw_signals['B'])

        # 计算Xs和Ys
        Xs = 3 * R - 2 * G
        Ys = 1.5 * R + G - 1.5 * B

        # 计算标准差比α
        std_xs = np.std(Xs)
        std_ys = np.std(Ys)

        if std_ys == 0:
            alpha = 0
        else:
            alpha = std_xs / std_ys

        # 计算CHROM信号
        S = Xs - alpha * Ys

        # 返回当前帧的信号值（最后一个）
        return S[-1]

    def get_signal_buffer(self) -> Dict[str, np.ndarray]:
        """
        获取当前信号缓冲区

        Returns:
            信号字典：
            {
                'raw_R': np.ndarray,
                'raw_G': np.ndarray,
                'raw_B': np.ndarray,
                'chrom': np.ndarray,
                'timestamps': np.ndarray
            }
        """
        return {
            'raw_R': np.array(self.raw_signals['R']),
            'raw_G': np.array(self.raw_signals['G']),
            'raw_B': np.array(self.raw_signals['B']),
            'chrom': np.array(self.chrom_signal),
            'timestamps': np.array(self.timestamps)
        }

    def get_signal_quality(self) -> Dict:
        """
        评估信号质量

        Returns:
            质量指标：
            {
                'snr': float,           # 信噪比
                'signal_length': int,   # 信号长度
                'is_valid': bool        # 是否有效
            }
        """
        chrom_len = len(self.chrom_signal)
        if chrom_len < self.min_signal_length:
            return {
                'snr': 0.0,
                'signal_length': chrom_len,
                'is_valid': False
            }

        signal = np.array(self.chrom_signal)

        # 简单的SNR估计（信号功率 / 噪声功率）
        signal_power = np.var(signal)

        # 使用高频成分作为噪声估计
        if chrom_len > 10:
            diff = np.diff(signal)
            noise_power = np.var(diff)

            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = float('inf')
        else:
            snr = 0.0

        return {
            'snr': round(snr, 2),
            'signal_length': chrom_len,
            'is_valid': chrom_len >= self.min_signal_length and snr > 0
        }

    def reset(self):
        """重置缓冲区（开始新的提取会话）"""
        self.rgb_buffer.clear()
        for key in self.raw_signals:
            self.raw_signals[key].clear()
        self.chrom_signal.clear()
        self.timestamps.clear()

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if self.extraction_count == 0:
            return {
                'total_extractions': 0,
                'avg_time_ms': 0.0,
                'meets_target': False
            }

        avg_time_ms = (self.total_time / self.extraction_count) * 1000

        return {
            'total_extractions': self.extraction_count,
            'avg_time_ms': round(avg_time_ms, 2),
            'meets_target': avg_time_ms <= 5  # 目标≤5ms
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self.extraction_count = 0
        self.total_time = 0.0


# ===================== 测试代码 =====================
def test_chrom_extractor():
    """CHROM提取器测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')

    from modules.detection.face_detector import FaceDetector
    from modules.roi.roi_extractor import ROIExtractor

    print("=" * 70)
    print("📝 rPPG信号提取模块测试")
    print("=" * 70)

    # 初始化模块
    print("\n【1/5】初始化模块")
    face_detector = FaceDetector(method='mtcnn')
    roi_extractor = ROIExtractor()
    chrom_extractor = CHROMExtractor(
        fps=30,
        window_size=300,
        use_forehead_only=True
    )
    print("✅ 所有模块初始化完成")

    # 读取测试视频
    print("\n【2/5】读取测试视频")
    test_video = "../../test_videos/test_video_1.avi"

    if not os.path.exists(test_video):
        print(f"❌ 测试视频不存在: {test_video}")
        return

    cap = cv2.VideoCapture(test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✅ 视频信息:")
    print(f"   帧率: {fps} FPS")
    print(f"   总帧数: {total_frames}")

    # 处理视频帧
    print(f"\n【3/5】提取rPPG信号（处理前100帧）")

    frame_count = 0
    signal_count = 0
    max_frames = 100

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 人脸检测
        detections = face_detector.detect(frame)
        if not detections:
            frame_count += 1
            continue

        # ROI提取
        rois = roi_extractor.extract_rois(frame, detections[0])
        if not rois or 'forehead' not in rois:
            frame_count += 1
            continue

        # rPPG信号提取
        timestamp = frame_count / fps
        signal_value = chrom_extractor.extract_from_rois(rois, timestamp)

        if signal_value is not None:
            signal_count += 1

        frame_count += 1

        # 每20帧打印进度
        if frame_count % 20 == 0:
            print(f"   处理 {frame_count} 帧: 提取信号 {signal_count} 个")

    cap.release()

    print(f"✅ 信号提取完成")
    print(f"   处理帧数: {frame_count}")
    print(f"   有效信号: {signal_count}")

    # 获取信号数据
    print(f"\n【4/5】分析信号质量")
    signals = chrom_extractor.get_signal_buffer()
    quality = chrom_extractor.get_signal_quality()

    print(f"📊 信号统计:")
    print(f"   原始R信号: {len(signals['raw_R'])} 个采样点")
    print(f"   原始G信号: {len(signals['raw_G'])} 个采样点")
    print(f"   原始B信号: {len(signals['raw_B'])} 个采样点")
    print(f"   CHROM信号: {len(signals['chrom'])} 个采样点")

    print(f"\n📈 信号质量:")
    print(f"   信噪比(SNR): {quality['snr']:.2f} dB")
    print(f"   信号长度: {quality['signal_length']}")
    print(f"   信号有效: {'✅' if quality['is_valid'] else '❌'}")

    # 保存信号数据
    print(f"\n【5/5】保存信号数据")
    output_dir = "../../test_output/signal"
    os.makedirs(output_dir, exist_ok=True)

    # 保存为numpy文件
    np.savez(
        f"{output_dir}/rppg_signal_raw.npz",
        raw_R=signals['raw_R'],
        raw_G=signals['raw_G'],
        raw_B=signals['raw_B'],
        chrom=signals['chrom'],
        timestamps=signals['timestamps'],
        fps=fps
    )
    print(f"✅ 原始信号已保存: {output_dir}/rppg_signal_raw.npz")

    # 绘制信号图（核心修改：解决维度不匹配）
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无GUI后端
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # 原始RGB信号
        axes[0].plot(signals['raw_R'], 'r-', label='R', alpha=0.7)
        axes[0].plot(signals['raw_G'], 'g-', label='G', alpha=0.7)
        axes[0].plot(signals['raw_B'], 'b-', label='B', alpha=0.7)
        axes[0].set_title('Raw RGB Signals')
        axes[0].set_ylabel('Normalized Value')
        axes[0].legend()
        axes[0].grid(True)

        # CHROM信号（原始）
        if len(signals['chrom']) > 0:
            axes[1].plot(signals['chrom'], 'purple', linewidth=1.5)
        axes[1].set_title('CHROM Signal (Raw)')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True)

        # 时域分析（核心修改：截取匹配的时间戳切片）
        chrom_len = len(signals['chrom'])
        time_len = len(signals['timestamps'])
        if chrom_len > 0 and time_len > 0:
            # CHROM信号从第min_signal_length帧开始，对应timestamps切片
            start_idx = max(0, time_len - chrom_len)  # 时间戳起始索引（匹配chrom长度）
            matched_timestamps = signals['timestamps'][start_idx:]
            time_axis = matched_timestamps - matched_timestamps[0]  # 相对时间
            axes[2].plot(time_axis, signals['chrom'], 'purple', linewidth=1.5)
        axes[2].set_title('CHROM Signal (Time Domain)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True)

        # 频域分析（简单FFT）- 仅当信号长度足够
        if chrom_len >= 60:
            from scipy import signal as scipy_signal

            # FFT
            fft_result = np.fft.fft(signals['chrom'])
            fft_freq = np.fft.fftfreq(chrom_len, 1 / fps)

            # 仅显示正频率
            positive_freq = fft_freq > 0
            axes[3].plot(fft_freq[positive_freq] * 60,
                         np.abs(fft_result[positive_freq]),
                         'purple')
            axes[3].set_title('Frequency Spectrum')
            axes[3].set_xlabel('Frequency (BPM)')
            axes[3].set_ylabel('Magnitude')
            axes[3].set_xlim([40, 180])  # 正常心率范围
            axes[3].grid(True)
        else:
            axes[3].set_title('Frequency Spectrum (Insufficient Signal)')
            axes[3].set_xlabel('Frequency (BPM)')
            axes[3].set_ylabel('Magnitude')
            axes[3].grid(True)
            axes[3].text(0.5, 0.5, '信号长度不足（需≥60）',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[3].transAxes)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/rppg_signal_plot.png", dpi=150)
        plt.close()

        print(f"✅ 信号图表已保存: {output_dir}/rppg_signal_plot.png")

    except ImportError as e:
        print(f"⚠️  matplotlib未安装，跳过绘图: {str(e)[:50]}")
    except Exception as e:
        print(f"⚠️  绘图时出现异常，跳过绘图: {str(e)[:50]}")

    # 性能统计
    stats = chrom_extractor.get_performance_stats()
    print(f"\n📈 性能统计:")
    print(f"   提取次数: {stats['total_extractions']}")
    print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

    # 提示：100帧数据偏少，建议运行完整测试
    if chrom_len < 60:
        print(f"\n⚠️  提示：100帧数据仅生成{chrom_len}个有效CHROM信号点（需≥60）")
        print(f"   建议运行完整测试（200帧+）: python tests/test_signal_extraction.py")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_chrom_extractor()