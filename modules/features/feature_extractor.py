"""
特征提取模块 - feature_extractor.py
功能：从预处理后的rPPG信号提取SpO2相关特征
特征类型：
1. RoR特征（AC/DC比值）
2. 心率特征（HR）
3. 心率变异性特征（HRV）
4. 信号质量特征（SNR、峰度、偏度）
5. 频域特征（功率谱密度）
性能目标：单次特征提取≤200ms
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq


class FeatureExtractor:
    """
    rPPG信号特征提取器

    核心功能：
    1. RoR特征：R峰检测与AC/DC比值计算
    2. 心率特征：瞬时心率、平均心率
    3. HRV特征：SDNN、RMSSD、pNN50
    4. 质量特征：SNR、峰度、偏度、能量
    5. 频域特征：主频率、功率谱密度
    """

    def __init__(
            self,
            fps: int = 30,  # 采样率
            min_hr: int = 40,  # 最小心率（BPM）
            max_hr: int = 200,  # 最大心率（BPM）
            peak_distance_factor: float = 0.6,  # R峰检测距离因子
            enable_hrv: bool = True,  # 是否计算HRV
            enable_frequency: bool = True  # 是否计算频域特征
    ):
        """
        初始化特征提取器

        Args:
            fps: 信号采样率
            min_hr: 最小有效心率
            max_hr: 最大有效心率
            peak_distance_factor: R峰检测最小距离因子
            enable_hrv: 是否启用HRV特征
            enable_frequency: 是否启用频域特征
        """
        self.fps = fps
        self.min_hr = min_hr
        self.max_hr = max_hr
        self.peak_distance_factor = peak_distance_factor
        self.enable_hrv = enable_hrv
        self.enable_frequency = enable_frequency

        # 计算R峰最小距离（样本数）
        # 最大心率对应的最小R-R间隔
        self.min_peak_distance = int(fps * 60 / max_hr * peak_distance_factor)

        # 性能统计
        self.extraction_count = 0
        self.total_time = 0.0

        print(f"✅ 特征提取器初始化完成")
        print(f"   采样率: {fps} Hz")
        print(f"   心率范围: {min_hr}-{max_hr} BPM")
        print(f"   R峰最小间距: {self.min_peak_distance} 个样本")

    def extract_features(
            self,
            signal: np.ndarray,
            return_intermediate: bool = False
    ) -> Dict:
        """
        提取所有特征

        Args:
            signal: 预处理后的rPPG信号（归一化后）
            return_intermediate: 是否返回中间结果

        Returns:
            特征字典：
            {
                'ror_features': {...},      # RoR特征
                'hr_features': {...},       # 心率特征
                'hrv_features': {...},      # HRV特征
                'quality_features': {...},  # 质量特征
                'frequency_features': {...} # 频域特征
            }
        """
        start_time = time.time()

        # 验证输入
        if signal is None or len(signal) < 60:
            return {
                'valid': False,
                'reason': 'Signal too short (< 60 samples)'
            }

        # 转换为numpy数组
        signal = np.array(signal, dtype=np.float64)

        # 检查异常值
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return {
                'valid': False,
                'reason': 'Signal contains NaN or Inf'
            }

        # 初始化特征字典
        features = {'valid': True}

        # 中间结果（如果需要）
        if return_intermediate:
            features['intermediate'] = {}

        # 1. R峰检测
        peaks, peak_properties = self._detect_r_peaks(signal)

        if return_intermediate:
            features['intermediate']['r_peaks'] = peaks
            features['intermediate']['peak_properties'] = peak_properties

        # 2. RoR特征
        ror_features = self._extract_ror_features(signal, peaks)
        features['ror_features'] = ror_features

        # 3. 心率特征
        hr_features = self._extract_hr_features(peaks)
        features['hr_features'] = hr_features

        # 4. HRV特征
        if self.enable_hrv and len(peaks) >= 3:
            hrv_features = self._extract_hrv_features(peaks)
            features['hrv_features'] = hrv_features
        else:
            features['hrv_features'] = {'available': False}

        # 5. 信号质量特征
        quality_features = self._extract_quality_features(signal, peaks)
        features['quality_features'] = quality_features

        # 6. 频域特征
        if self.enable_frequency:
            freq_features = self._extract_frequency_features(signal)
            features['frequency_features'] = freq_features
        else:
            features['frequency_features'] = {'available': False}

        # 更新性能统计
        elapsed = time.time() - start_time
        self.extraction_count += 1
        self.total_time += elapsed

        features['extraction_time_ms'] = round(elapsed * 1000, 2)

        return features

    def _detect_r_peaks(
            self,
            signal: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        检测R峰（波峰）

        Args:
            signal: 输入信号

        Returns:
            peaks: R峰位置索引
            properties: R峰属性（高度、宽度等）
        """
        # 使用scipy的find_peaks
        peaks, properties = scipy_signal.find_peaks(
            signal,
            distance=self.min_peak_distance,  # 最小峰间距
            prominence=0.5,  # 峰的显著性
            width=3  # 最小峰宽度
        )

        return peaks, properties

    def _extract_ror_features(
            self,
            signal: np.ndarray,
            peaks: np.ndarray
    ) -> Dict:
        """
        提取RoR（Rate of Ratios）特征

        RoR定义：AC分量与DC分量的比值
        AC: 信号的交流分量（变化部分）
        DC: 信号的直流分量（平均值）

        Args:
            signal: 输入信号
            peaks: R峰位置

        Returns:
            RoR特征字典
        """
        # 计算AC分量（标准差）
        ac_component = np.std(signal)

        # 计算DC分量（均值的绝对值）
        dc_component = abs(np.mean(signal))

        # 避免除零
        if dc_component < 1e-8:
            dc_component = 1e-8

        # RoR比值
        ror = ac_component / dc_component

        # 如果有R峰，计算峰值相关的RoR
        if len(peaks) > 0:
            peak_amplitudes = signal[peaks]
            peak_ac = np.std(peak_amplitudes)
            peak_dc = abs(np.mean(peak_amplitudes))

            if peak_dc < 1e-8:
                peak_dc = 1e-8

            peak_ror = peak_ac / peak_dc
        else:
            peak_ror = 0.0

        # 峰峰值特征
        peak_to_peak = np.ptp(signal)  # max - min

        return {
            'ror': round(ror, 6),
            'ac_component': round(ac_component, 6),
            'dc_component': round(dc_component, 6),
            'peak_ror': round(peak_ror, 6),
            'peak_to_peak': round(peak_to_peak, 6),
            'n_peaks': len(peaks)
        }

    def _extract_hr_features(self, peaks: np.ndarray) -> Dict:
        """
        提取心率特征

        Args:
            peaks: R峰位置

        Returns:
            心率特征字典
        """
        if len(peaks) < 2:
            return {
                'available': False,
                'mean_hr': 0.0,
                'std_hr': 0.0
            }

        # 计算R-R间隔（样本数）
        rr_intervals = np.diff(peaks)

        # 转换为时间（秒）
        rr_intervals_sec = rr_intervals / self.fps

        # 转换为心率（BPM）
        instantaneous_hr = 60.0 / rr_intervals_sec

        # 过滤异常心率
        valid_hr = instantaneous_hr[
            (instantaneous_hr >= self.min_hr) &
            (instantaneous_hr <= self.max_hr)
            ]

        if len(valid_hr) == 0:
            return {
                'available': False,
                'mean_hr': 0.0,
                'std_hr': 0.0
            }

        # 心率统计
        mean_hr = np.mean(valid_hr)
        std_hr = np.std(valid_hr)
        median_hr = np.median(valid_hr)
        min_hr = np.min(valid_hr)
        max_hr = np.max(valid_hr)

        return {
            'available': True,
            'mean_hr': round(mean_hr, 2),
            'std_hr': round(std_hr, 2),
            'median_hr': round(median_hr, 2),
            'min_hr': round(min_hr, 2),
            'max_hr': round(max_hr, 2),
            'hr_range': round(max_hr - min_hr, 2),
            'n_valid_intervals': len(valid_hr)
        }

    def _extract_hrv_features(self, peaks: np.ndarray) -> Dict:
        """
        提取心率变异性（HRV）特征

        常用HRV指标：
        - SDNN: R-R间隔标准差
        - RMSSD: 连续R-R间隔差值的均方根
        - pNN50: 相邻R-R间隔差值>50ms的百分比

        Args:
            peaks: R峰位置

        Returns:
            HRV特征字典
        """
        if len(peaks) < 3:
            return {
                'available': False
            }

        # R-R间隔（毫秒）
        rr_intervals = np.diff(peaks) / self.fps * 1000

        # SDNN: 标准差
        sdnn = np.std(rr_intervals)

        # RMSSD: 连续差值的均方根
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))

        # pNN50: 差值>50ms的百分比
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0

        # 其他统计量
        mean_rr = np.mean(rr_intervals)
        median_rr = np.median(rr_intervals)

        return {
            'available': True,
            'sdnn': round(sdnn, 2),  # ms
            'rmssd': round(rmssd, 2),  # ms
            'pnn50': round(pnn50, 2),  # %
            'mean_rr': round(mean_rr, 2),  # ms
            'median_rr': round(median_rr, 2),  # ms
            'n_intervals': len(rr_intervals)
        }

    def _extract_quality_features(
            self,
            signal: np.ndarray,
            peaks: np.ndarray
    ) -> Dict:
        """
        提取信号质量特征

        Args:
            signal: 输入信号
            peaks: R峰位置

        Returns:
            质量特征字典
        """
        # 基本统计量
        mean_val = np.mean(signal)
        std_val = np.std(signal)

        # 偏度（Skewness）
        skewness = self._calculate_skewness(signal)

        # 峰度（Kurtosis）
        kurtosis = self._calculate_kurtosis(signal)

        # 能量
        energy = np.sum(signal ** 2)

        # 信噪比估计
        if len(peaks) > 0:
            # 使用峰值作为信号，其他作为噪声
            signal_power = np.var(signal[peaks])
            noise_indices = np.setdiff1d(np.arange(len(signal)), peaks)
            noise_power = np.var(signal[noise_indices]) if len(noise_indices) > 0 else 1e-8

            if noise_power < 1e-8:
                noise_power = 1e-8

            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0.0

        # 零交叉率
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        zero_crossing_rate = zero_crossings / len(signal)

        return {
            'mean': round(mean_val, 6),
            'std': round(std_val, 6),
            'skewness': round(skewness, 4),
            'kurtosis': round(kurtosis, 4),
            'energy': round(energy, 4),
            'snr': round(snr, 2),
            'zero_crossing_rate': round(zero_crossing_rate, 4)
        }

    def _extract_frequency_features(self, signal: np.ndarray) -> Dict:
        """
        提取频域特征

        Args:
            signal: 输入信号

        Returns:
            频域特征字典
        """
        # FFT
        fft_result = fft(signal)
        fft_freq = fftfreq(len(signal), 1 / self.fps)

        # 仅使用正频率
        positive_freq_mask = fft_freq > 0
        fft_freq_positive = fft_freq[positive_freq_mask]
        fft_magnitude = np.abs(fft_result[positive_freq_mask])

        # 转换为BPM
        freq_bpm = fft_freq_positive * 60

        # 在心率有效范围内寻找主频率
        valid_freq_mask = (freq_bpm >= self.min_hr) & (freq_bpm <= self.max_hr)

        if np.sum(valid_freq_mask) > 0:
            valid_freq_bpm = freq_bpm[valid_freq_mask]
            valid_magnitude = fft_magnitude[valid_freq_mask]

            # 主频率（最大幅度对应的频率）
            peak_idx = np.argmax(valid_magnitude)
            dominant_freq = valid_freq_bpm[peak_idx]
            dominant_magnitude = valid_magnitude[peak_idx]

            # 总功率
            total_power = np.sum(valid_magnitude ** 2)

            # 主频率功率占比
            dominant_power_ratio = (dominant_magnitude ** 2) / total_power if total_power > 0 else 0
        else:
            dominant_freq = 0.0
            dominant_magnitude = 0.0
            dominant_power_ratio = 0.0
            total_power = 0.0

        # 频谱熵（频谱的分散程度）
        spectrum_entropy = self._calculate_spectrum_entropy(fft_magnitude)

        return {
            'available': True,
            'dominant_freq_bpm': round(dominant_freq, 2),
            'dominant_magnitude': round(dominant_magnitude, 4),
            'dominant_power_ratio': round(dominant_power_ratio, 4),
            'total_power': round(total_power, 4),
            'spectrum_entropy': round(spectrum_entropy, 4)
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-8:
            return 0.0

        n = len(data)
        skewness = (np.sum((data - mean) ** 3) / n) / (std ** 3)

        return skewness

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-8:
            return 0.0

        n = len(data)
        kurtosis = (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3

        return kurtosis

    def _calculate_spectrum_entropy(self, magnitude: np.ndarray) -> float:
        """计算频谱熵"""
        # 归一化为概率分布
        magnitude_sum = np.sum(magnitude)

        if magnitude_sum < 1e-8:
            return 0.0

        prob = magnitude / magnitude_sum

        # 避免log(0)
        prob = prob[prob > 1e-10]

        # 计算熵
        entropy = -np.sum(prob * np.log2(prob))

        return entropy

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
            'total_time_s': round(self.total_time, 2),
            'meets_target': avg_time_ms <= 200  # 目标≤200ms
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self.extraction_count = 0
        self.total_time = 0.0


# ===================== 测试代码 =====================
def test_feature_extractor():
    """特征提取器测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')

    print("=" * 70)
    print("📝 特征提取模块测试")
    print("=" * 70)

    # 加载预处理信号
    print("\n【1/4】加载预处理信号")
    signal_file = "../../test_output/signal/rppg_signal_processed.npz"

    if not os.path.exists(signal_file):
        print(f"❌ 预处理信号文件不存在: {signal_file}")
        print("💡 请先运行2.11任务的测试脚本")
        return

    data = np.load(signal_file, allow_pickle=True)
    processed_signal = data['processed_signal']
    fps = float(data['fps']) if 'fps' in data else 30

    print(f"✅ 成功加载信号")
    print(f"   信号长度: {len(processed_signal)} 个采样点")
    print(f"   采样率: {fps} Hz")

    # 初始化特征提取器
    print("\n【2/4】初始化特征提取器")
    extractor = FeatureExtractor(
        fps=int(fps),
        min_hr=40,
        max_hr=200,
        enable_hrv=True,
        enable_frequency=True
    )

    # 提取特征
    print("\n【3/4】提取特征")
    features = extractor.extract_features(
        processed_signal,
        return_intermediate=True
    )

    if not features.get('valid', False):
        print(f"❌ 特征提取失败")
        print(f"   原因: {features.get('reason', 'Unknown')}")
        return

    print(f"✅ 特征提取完成")
    print(f"   耗时: {features['extraction_time_ms']:.2f} ms")

    # 显示特征
    print("\n【4/4】特征详情")

    # RoR特征
    print(f"\n📊 RoR特征:")
    ror = features['ror_features']
    for key, value in ror.items():
        print(f"   {key}: {value}")

    # 心率特征
    print(f"\n💓 心率特征:")
    hr = features['hr_features']
    if hr.get('available', False):
        for key, value in hr.items():
            if key != 'available':
                print(f"   {key}: {value}")
    else:
        print(f"   ⚠️  心率特征不可用")

    # HRV特征
    print(f"\n📈 HRV特征:")
    hrv = features['hrv_features']
    if hrv.get('available', False):
        for key, value in hrv.items():
            if key != 'available':
                print(f"   {key}: {value}")
    else:
        print(f"   ⚠️  HRV特征不可用（需要至少3个R峰）")

    # 信号质量特征
    print(f"\n🔍 信号质量特征:")
    quality = features['quality_features']
    for key, value in quality.items():
        print(f"   {key}: {value}")

    # 频域特征
    print(f"\n🌊 频域特征:")
    freq = features['frequency_features']
    if freq.get('available', False):
        for key, value in freq.items():
            if key != 'available':
                print(f"   {key}: {value}")
    else:
        print(f"   ⚠️  频域特征不可用")

    # 保存特征
    output_dir = "../../test_output/features"
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        f"{output_dir}/extracted_features.npz",
        features=features,
        signal=processed_signal,
        fps=fps
    )
    print(f"\n✅ 特征已保存: {output_dir}/extracted_features.npz")

    # 性能统计
    stats = extractor.get_performance_stats()
    print(f"\n📈 性能统计:")
    print(f"   提取次数: {stats['total_extractions']}")
    print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_feature_extractor()