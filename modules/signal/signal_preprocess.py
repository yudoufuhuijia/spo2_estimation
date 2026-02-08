"""
信号预处理模块 - signal_preprocess.py
功能：对原始rPPG信号进行滤波、去趋势、归一化处理
处理流程：
1. 带通滤波（0.5-4Hz，对应30-240 BPM）
2. 去趋势处理（detrend）
3. 归一化（z-score标准化）
4. 可选：轻量化ICA去噪
性能目标：单次处理≤100ms
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple
from scipy import signal
from scipy.signal import butter, filtfilt, detrend


class SignalPreprocessor:
    """
    rPPG信号预处理器

    核心功能：
    1. 带通滤波：保留心率频率范围（0.5-4Hz）
    2. 去趋势：移除低频漂移
    3. 归一化：标准化信号幅度
    4. 质量检查：评估预处理后信号质量
    """

    def __init__(
            self,
            fps: int = 30,  # 采样率
            lowcut: float = 0.5,  # 低频截止（Hz），对应30 BPM
            highcut: float = 4.0,  # 高频截止（Hz），对应240 BPM
            filter_order: int = 4,  # 滤波器阶数
            enable_detrend: bool = True,  # 是否去趋势
            enable_normalization: bool = True,  # 是否归一化
            min_signal_length: int = 60  # 最小信号长度
    ):
        """
        初始化预处理器

        Args:
            fps: 信号采样率（帧率）
            lowcut: 低频截止频率（Hz）
            highcut: 高频截止频率（Hz）
            filter_order: 滤波器阶数（越高越陡峭）
            enable_detrend: 是否启用去趋势
            enable_normalization: 是否启用归一化
            min_signal_length: 最小信号长度
        """
        self.fps = fps
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.enable_detrend = enable_detrend
        self.enable_normalization = enable_normalization
        self.min_signal_length = min_signal_length

        # 计算归一化频率（相对于奈奎斯特频率）
        nyquist = 0.5 * fps
        self.low_norm = lowcut / nyquist
        self.high_norm = highcut / nyquist

        # 设计带通滤波器
        self.b, self.a = self._design_bandpass_filter()

        # 性能统计
        self.processing_count = 0
        self.total_time = 0.0

        print(f"✅ 信号预处理器初始化完成")
        print(f"   采样率: {fps} Hz")
        print(f"   带通范围: {lowcut}-{highcut} Hz ({lowcut * 60:.0f}-{highcut * 60:.0f} BPM)")
        print(f"   滤波器阶数: {filter_order}")

    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        设计巴特沃斯带通滤波器

        Returns:
            b, a: 滤波器系数
        """
        # 检查频率范围合法性
        if self.low_norm <= 0 or self.high_norm >= 1:
            raise ValueError(
                f"频率范围无效: {self.lowcut}-{self.highcut} Hz "
                f"(采样率: {self.fps} Hz)"
            )

        # 巴特沃斯带通滤波器
        b, a = butter(
            self.filter_order,
            [self.low_norm, self.high_norm],
            btype='band',
            analog=False
        )

        return b, a

    def preprocess(
            self,
            raw_signal: np.ndarray,
            return_intermediate: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        完整预处理流程

        Args:
            raw_signal: 原始rPPG信号（1D数组）
            return_intermediate: 是否返回中间结果

        Returns:
            预处理结果字典：
            {
                'raw': 原始信号,
                'filtered': 滤波后信号（可选）,
                'detrended': 去趋势后信号（可选）,
                'processed': 最终处理信号,
                'quality': 信号质量指标
            }
        """
        start_time = time.time()

        # 验证输入
        if raw_signal is None or len(raw_signal) < self.min_signal_length:
            return {
                'raw': raw_signal,
                'processed': None,
                'quality': {'valid': False, 'reason': 'Signal too short'}
            }

        # 转换为numpy数组
        signal_array = np.array(raw_signal, dtype=np.float64)

        # 检查是否有NaN或Inf
        if np.any(np.isnan(signal_array)) or np.any(np.isinf(signal_array)):
            return {
                'raw': raw_signal,
                'processed': None,
                'quality': {'valid': False, 'reason': 'Contains NaN or Inf'}
            }

        # 初始化结果字典
        result = {'raw': signal_array.copy()}

        # 步骤1: 带通滤波
        filtered_signal = self._apply_bandpass_filter(signal_array)
        if return_intermediate:
            result['filtered'] = filtered_signal.copy()

        # 步骤2: 去趋势
        if self.enable_detrend:
            detrended_signal = self._apply_detrend(filtered_signal)
            if return_intermediate:
                result['detrended'] = detrended_signal.copy()
        else:
            detrended_signal = filtered_signal

        # 步骤3: 归一化
        if self.enable_normalization:
            normalized_signal = self._apply_normalization(detrended_signal)
        else:
            normalized_signal = detrended_signal

        # 最终处理信号
        result['processed'] = normalized_signal

        # 步骤4: 质量评估
        quality = self._assess_quality(normalized_signal)
        result['quality'] = quality

        # 更新性能统计
        elapsed = time.time() - start_time
        self.processing_count += 1
        self.total_time += elapsed

        return result

    def _apply_bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        应用带通滤波器

        Args:
            signal_data: 输入信号

        Returns:
            滤波后的信号
        """
        # 使用filtfilt实现零相位滤波
        try:
            filtered = filtfilt(self.b, self.a, signal_data)
            return filtered
        except Exception as e:
            print(f"⚠️  滤波失败: {e}")
            return signal_data

    def _apply_detrend(self, signal_data: np.ndarray) -> np.ndarray:
        """
        去趋势处理（移除线性或多项式趋势）

        Args:
            signal_data: 输入信号

        Returns:
            去趋势后的信号
        """
        # 使用scipy的detrend函数（默认移除线性趋势）
        detrended = detrend(signal_data, type='linear')
        return detrended

    def _apply_normalization(self, signal_data: np.ndarray) -> np.ndarray:
        """
        归一化（z-score标准化）

        Args:
            signal_data: 输入信号

        Returns:
            归一化后的信号
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)

        if std == 0:
            # 避免除零
            return signal_data - mean

        # z-score标准化：(x - μ) / σ
        normalized = (signal_data - mean) / std
        return normalized

    def _assess_quality(self, processed_signal: np.ndarray) -> Dict:
        """
        评估预处理后的信号质量

        Args:
            processed_signal: 预处理后的信号

        Returns:
            质量指标字典
        """
        # 计算SNR（简单估计）
        signal_power = np.var(processed_signal)

        # 使用高频成分估计噪声
        if len(processed_signal) > 10:
            diff = np.diff(processed_signal)
            noise_power = np.var(diff)

            if noise_power > 0 and signal_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 0.0
        else:
            snr = 0.0

        # 计算峰峰值
        peak_to_peak = np.max(processed_signal) - np.min(processed_signal)

        # 计算零交叉率（心率估计的辅助指标）
        zero_crossings = np.sum(np.diff(np.sign(processed_signal)) != 0)
        zcr = zero_crossings / len(processed_signal)

        # 判断信号有效性
        is_valid = (
                snr > 5.0 and  # SNR > 5dB
                peak_to_peak > 0.1 and  # 有足够的幅度变化
                len(processed_signal) >= self.min_signal_length
        )

        return {
            'valid': is_valid,
            'snr': round(snr, 2),
            'peak_to_peak': round(peak_to_peak, 4),
            'zero_crossing_rate': round(zcr, 4),
            'mean': round(np.mean(processed_signal), 4),
            'std': round(np.std(processed_signal), 4),
            'length': len(processed_signal)
        }

    def preprocess_batch(
            self,
            signal_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        批量预处理多个信号

        Args:
            signal_dict: 信号字典，例如：
                {
                    'chrom': chrom_signal,
                    'raw_R': r_signal,
                    'raw_G': g_signal
                }

        Returns:
            预处理结果字典
        """
        results = {}

        for signal_name, signal_data in signal_dict.items():
            result = self.preprocess(signal_data)
            results[signal_name] = result

        return results

    def get_performance_stats(self) -> Dict:
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
            'total_time_s': round(self.total_time, 2),
            'meets_target': avg_time_ms <= 100  # 目标≤100ms
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self.processing_count = 0
        self.total_time = 0.0

    def save_processed_signal(
            self,
            result: Dict,
            output_path: str,
            metadata: Optional[Dict] = None
    ):
        """
        保存预处理后的信号

        Args:
            result: preprocess()的返回结果
            output_path: 输出文件路径（.npz）
            metadata: 额外的元数据
        """
        save_data = {
            'processed_signal': result['processed'],
            'quality': result['quality'],
            'fps': self.fps,
            'lowcut': self.lowcut,
            'highcut': self.highcut,
        }

        # 添加原始信号（可选）
        if 'raw' in result:
            save_data['raw_signal'] = result['raw']

        # 添加中间结果（如果有）
        if 'filtered' in result:
            save_data['filtered_signal'] = result['filtered']
        if 'detrended' in result:
            save_data['detrended_signal'] = result['detrended']

        # 添加元数据
        if metadata:
            save_data['metadata'] = metadata

        np.savez(output_path, **save_data)
        print(f"✅ 预处理信号已保存: {output_path}")


# ===================== 测试代码 =====================
def test_signal_preprocessor():
    """信号预处理器测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')

    print("=" * 70)
    print("📝 信号预处理模块测试")
    print("=" * 70)

    # 加载原始信号
    print("\n【1/5】加载原始rPPG信号")
    signal_file = "../../test_output/signal/rppg_signal_raw.npz"

    if not os.path.exists(signal_file):
        print(f"❌ 信号文件不存在: {signal_file}")
        print("💡 请先运行2.10任务的测试脚本生成信号数据")
        return

    data = np.load(signal_file)
    raw_chrom = data['chrom']
    fps = float(data['fps']) if 'fps' in data else 30.0

    print(f"✅ 成功加载信号")
    print(f"   信号长度: {len(raw_chrom)} 个采样点")
    print(f"   采样率: {fps} Hz")
    print(f"   时间跨度: {len(raw_chrom) / fps:.2f} 秒")

    # 初始化预处理器
    print("\n【2/5】初始化预处理器")
    preprocessor = SignalPreprocessor(
        fps=int(fps),
        lowcut=0.5,  # 30 BPM
        highcut=4.0,  # 240 BPM
        filter_order=4,
        enable_detrend=True,
        enable_normalization=True,
        min_signal_length=60
    )

    # 预处理信号
    print("\n【3/5】预处理信号")
    result = preprocessor.preprocess(
        raw_chrom,
        return_intermediate=True
    )

    if result['processed'] is None:
        print(f"❌ 预处理失败")
        if 'quality' in result:
            print(f"   原因: {result['quality'].get('reason', 'Unknown')}")
        return

    print(f"✅ 预处理完成")

    # 分析结果
    print("\n【4/5】分析预处理结果")
    quality = result['quality']

    print(f"📊 信号质量:")
    print(f"   有效性: {'✅ 有效' if quality['valid'] else '❌ 无效'}")
    print(f"   SNR: {quality['snr']:.2f} dB")
    print(f"   峰峰值: {quality['peak_to_peak']:.4f}")
    print(f"   零交叉率: {quality['zero_crossing_rate']:.4f}")
    print(f"   均值: {quality['mean']:.4f}")
    print(f"   标准差: {quality['std']:.4f}")

    # 保存结果
    print("\n【5/5】保存预处理信号")
    output_dir = "../../test_output/signal"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/rppg_signal_processed.npz"
    preprocessor.save_processed_signal(
        result,
        output_file,
        metadata={'source': 'test_video_1.avi'}
    )

    # 绘制对比图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        # 原始信号
        axes[0].plot(result['raw'], 'gray', linewidth=1, alpha=0.7)
        axes[0].set_title('Raw Signal', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # 滤波后信号
        if 'filtered' in result:
            axes[1].plot(result['filtered'], 'blue', linewidth=1)
            axes[1].set_title('After Bandpass Filter (0.5-4Hz)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)

        # 去趋势后信号
        if 'detrended' in result:
            axes[2].plot(result['detrended'], 'green', linewidth=1)
            axes[2].set_title('After Detrending', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Amplitude')
            axes[2].grid(True, alpha=0.3)

        # 最终处理信号
        axes[3].plot(result['processed'], 'red', linewidth=1.5)
        axes[3].set_title('Final Processed Signal (Normalized)', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Sample Index')
        axes[3].set_ylabel('Amplitude (z-score)')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = f"{output_dir}/signal_preprocessing_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 对比图已保存: {plot_file}")

    except ImportError:
        print(f"⚠️  matplotlib未安装，跳过可视化")

    # 性能统计
    stats = preprocessor.get_performance_stats()
    print(f"\n📈 性能统计:")
    print(f"   处理次数: {stats['total_processed']}")
    print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_signal_preprocessor()