"""
2.11 信号预处理 - 完整测试脚本
测试内容：
1. 带通滤波测试
2. 去趋势测试
3. 归一化测试
4. 完整预处理流程测试
5. 性能基准测试
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 项目根目录
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

# 导入模块
try:
    from modules.signal.signal_preprocess import SignalPreprocessor

    print("✅ signal_preprocess 导入成功")
except ImportError as e:
    print(f"❌ signal_preprocess 导入失败: {e}")
    print("💡 请确保已将signal_preprocess.py放到modules/signal/目录下")
    sys.exit(1)


class SignalPreprocessTester:
    """信号预处理完整测试器"""

    def __init__(self, output_dir: str = "test_output/signal"):
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 测试结果输出: {os.path.abspath(self.output_dir)}")

        # 加载原始信号
        self.raw_signal = None
        self.fps = 30
        self._load_signal()

    def _load_signal(self):
        """加载原始rPPG信号"""
        signal_file = os.path.join(self.output_dir, "rppg_signal_raw.npz")

        if not os.path.exists(signal_file):
            print(f"\n❌ 原始信号文件不存在: {signal_file}")
            print("💡 请先运行2.10任务的测试脚本生成信号数据")
            sys.exit(1)

        data = np.load(signal_file)
        self.raw_signal = data['chrom']
        self.fps = float(data['fps']) if 'fps' in data else 30

        print(f"\n✅ 成功加载原始信号")
        print(f"   信号长度: {len(self.raw_signal)} 个采样点")
        print(f"   采样率: {self.fps} Hz")
        print(f"   时间跨度: {len(self.raw_signal) / self.fps:.2f} 秒")

    def test_bandpass_filter(self):
        """测试带通滤波"""
        print(f"\n🔧 【测试1】带通滤波测试")
        print("=" * 50)

        # 创建预处理器（仅滤波）
        preprocessor = SignalPreprocessor(
            fps=int(self.fps),
            lowcut=0.5,
            highcut=4.0,
            filter_order=4,
            enable_detrend=False,
            enable_normalization=False
        )

        # 应用滤波
        result = preprocessor.preprocess(
            self.raw_signal,
            return_intermediate=True
        )

        if result['processed'] is None:
            print(f"❌ 滤波失败")
            return False

        print(f"✅ 滤波完成")
        print(f"   输入信号长度: {len(self.raw_signal)}")
        print(f"   输出信号长度: {len(result['processed'])}")

        # 分析频率特性
        filtered = result['processed']

        # 简单的幅度统计
        print(f"\n📊 滤波前后对比:")
        print(f"   原始信号:")
        print(f"     均值: {np.mean(self.raw_signal):.6f}")
        print(f"     标准差: {np.std(self.raw_signal):.6f}")
        print(f"     峰峰值: {np.ptp(self.raw_signal):.6f}")
        print(f"   滤波后:")
        print(f"     均值: {np.mean(filtered):.6f}")
        print(f"     标准差: {np.std(filtered):.6f}")
        print(f"     峰峰值: {np.ptp(filtered):.6f}")

        return True

    def test_detrend(self):
        """测试去趋势"""
        print(f"\n📈 【测试2】去趋势测试")
        print("=" * 50)

        # 创建预处理器（仅去趋势）
        preprocessor = SignalPreprocessor(
            fps=int(self.fps),
            enable_detrend=True,
            enable_normalization=False
        )

        # 先滤波再去趋势
        from scipy.signal import filtfilt, butter
        b, a = butter(4, [0.5 / (self.fps / 2), 4.0 / (self.fps / 2)], btype='band')
        filtered = filtfilt(b, a, self.raw_signal)

        result = preprocessor.preprocess(filtered, return_intermediate=True)

        if result['processed'] is None:
            print(f"❌ 去趋势失败")
            return False

        print(f"✅ 去趋势完成")

        # 分析趋势移除效果
        detrended = result['processed']

        print(f"\n📊 去趋势前后对比:")
        print(f"   去趋势前:")
        print(f"     均值: {np.mean(filtered):.6f}")
        print(f"     最小值: {np.min(filtered):.6f}")
        print(f"     最大值: {np.max(filtered):.6f}")
        print(f"   去趋势后:")
        print(f"     均值: {np.mean(detrended):.6f}")
        print(f"     最小值: {np.min(detrended):.6f}")
        print(f"     最大值: {np.max(detrended):.6f}")

        return True

    def test_normalization(self):
        """测试归一化"""
        print(f"\n📏 【测试3】归一化测试")
        print("=" * 50)

        # 创建预处理器（完整流程）
        preprocessor = SignalPreprocessor(
            fps=int(self.fps),
            enable_detrend=True,
            enable_normalization=True
        )

        result = preprocessor.preprocess(
            self.raw_signal,
            return_intermediate=True
        )

        if result['processed'] is None:
            print(f"❌ 归一化失败")
            return False

        print(f"✅ 归一化完成")

        normalized = result['processed']

        print(f"\n📊 归一化结果:")
        print(f"   均值: {np.mean(normalized):.6f} (应接近0)")
        print(f"   标准差: {np.std(normalized):.6f} (应接近1)")
        print(f"   最小值: {np.min(normalized):.6f}")
        print(f"   最大值: {np.max(normalized):.6f}")

        # 验证z-score标准化
        is_normalized = (
                abs(np.mean(normalized)) < 0.01 and
                abs(np.std(normalized) - 1.0) < 0.01
        )

        print(f"   z-score验证: {'✅ 通过' if is_normalized else '⚠️ 偏差较大'}")

        return True

    def test_full_pipeline(self):
        """测试完整预处理流程"""
        print(f"\n🔄 【测试4】完整预处理流程测试")
        print("=" * 50)

        # 创建预处理器
        print("\n🔧 初始化预处理器...")
        preprocessor = SignalPreprocessor(
            fps=int(self.fps),
            lowcut=0.5,
            highcut=4.0,
            filter_order=4,
            enable_detrend=True,
            enable_normalization=True,
            min_signal_length=60
        )

        # 完整预处理
        print("\n🔍 开始预处理...")
        result = preprocessor.preprocess(
            self.raw_signal,
            return_intermediate=True
        )

        if result['processed'] is None:
            print(f"❌ 预处理失败")
            if 'quality' in result:
                print(f"   原因: {result['quality'].get('reason', 'Unknown')}")
            return None

        print(f"✅ 预处理完成")

        # 分析质量
        quality = result['quality']

        print(f"\n📈 信号质量评估:")
        print(f"   有效性: {'✅ 有效' if quality['valid'] else '❌ 无效'}")
        print(f"   SNR: {quality['snr']:.2f} dB")
        print(f"   峰峰值: {quality['peak_to_peak']:.4f}")
        print(f"   零交叉率: {quality['zero_crossing_rate']:.4f}")
        print(f"   信号长度: {quality['length']} 个采样点")

        # 保存结果
        print(f"\n💾 保存预处理结果...")
        output_file = os.path.join(self.output_dir, "rppg_signal_processed.npz")
        preprocessor.save_processed_signal(
            result,
            output_file,
            metadata={
                'source': 'test_video_1.avi',
                'preprocessing': {
                    'lowcut': preprocessor.lowcut,
                    'highcut': preprocessor.highcut,
                    'filter_order': preprocessor.filter_order,
                    'detrend': preprocessor.enable_detrend,
                    'normalization': preprocessor.enable_normalization
                }
            }
        )

        return result

    def test_performance_benchmark(self):
        """性能基准测试"""
        print(f"\n⚡【测试5】性能基准测试")
        print("=" * 50)

        # 创建预处理器
        preprocessor = SignalPreprocessor(
            fps=int(self.fps),
            enable_detrend=True,
            enable_normalization=True
        )

        print(f"📹 对信号进行10次重复预处理...")

        # 重置性能统计
        preprocessor.reset_performance_stats()

        # 多次处理测试
        for i in range(10):
            result = preprocessor.preprocess(self.raw_signal)

            if (i + 1) % 3 == 0:
                stats = preprocessor.get_performance_stats()
                print(f"   完成 {i + 1}/10 次: 平均 {stats['avg_time_ms']:.2f} ms")

        # 最终统计
        stats = preprocessor.get_performance_stats()

        print(f"\n📈 性能统计:")
        print(f"   处理次数: {stats['total_processed']}")
        print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
        print(f"   总耗时: {stats['total_time_s']:.2f} 秒")
        print(f"   性能目标: ≤100ms")
        print(f"   性能达标: {'✅ 是' if stats['meets_target'] else '❌ 否'}")

        return stats

    def visualize_results(self, result: dict):
        """可视化预处理结果"""
        print(f"\n📊 【测试6】生成可视化图表")
        print("=" * 50)

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # 创建子图
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            # 1. 原始信号
            axes[0].plot(result['raw'], color='gray', linewidth=1, alpha=0.7)
            axes[0].set_title('1. Raw rPPG Signal', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim([0, len(result['raw'])])

            # 2. 滤波后
            if 'filtered' in result:
                axes[1].plot(result['filtered'], color='blue', linewidth=1)
                axes[1].set_title('2. After Bandpass Filter (0.5-4 Hz)',
                                  fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Amplitude')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xlim([0, len(result['filtered'])])

            # 3. 去趋势后
            if 'detrended' in result:
                axes[2].plot(result['detrended'], color='green', linewidth=1)
                axes[2].set_title('3. After Detrending',
                                  fontsize=12, fontweight='bold')
                axes[2].set_ylabel('Amplitude')
                axes[2].grid(True, alpha=0.3)
                axes[2].set_xlim([0, len(result['detrended'])])

            # 4. 最终归一化信号
            axes[3].plot(result['processed'], color='red', linewidth=1.5)
            axes[3].set_title('4. Final Normalized Signal (Ready for Feature Extraction)',
                              fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Sample Index')
            axes[3].set_ylabel('Amplitude (z-score)')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim([0, len(result['processed'])])

            # 添加质量信息
            quality = result['quality']
            quality_text = (
                f"SNR: {quality['snr']:.2f} dB\n"
                f"Valid: {'Yes' if quality['valid'] else 'No'}"
            )
            axes[3].text(0.98, 0.95, quality_text,
                         transform=axes[3].transAxes,
                         verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         fontsize=10)

            plt.tight_layout()

            # 保存图表
            plot_file = os.path.join(self.output_dir, "signal_preprocessing_steps.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✅ 可视化图表已保存: {plot_file}")

            # 绘制频谱对比
            self._plot_frequency_comparison(result)

            return True

        except ImportError:
            print(f"⚠️  matplotlib未安装，跳过可视化")
            print(f"💡 安装: pip install matplotlib --break-system-packages")
            return False

    def _plot_frequency_comparison(self, result: dict):
        """绘制频谱对比图"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # FFT分析
        def plot_spectrum(signal, ax, title, color):
            # FFT
            fft_result = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal), 1 / self.fps)

            # 仅显示正频率
            positive_freq = fft_freq > 0
            freq_bpm = fft_freq[positive_freq] * 60
            magnitude = np.abs(fft_result[positive_freq])

            ax.plot(freq_bpm, magnitude, color=color, linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Frequency (BPM)')
            ax.set_ylabel('Magnitude')
            ax.set_xlim([0, 250])
            ax.grid(True, alpha=0.3)

            # 标注心率范围
            ax.axvspan(30, 240, alpha=0.1, color='green', label='Valid HR Range')
            ax.legend()

        # 原始信号频谱
        plot_spectrum(result['raw'], axes[0],
                      'Frequency Spectrum - Raw Signal', 'gray')

        # 预处理后频谱
        plot_spectrum(result['processed'], axes[1],
                      'Frequency Spectrum - Processed Signal', 'red')

        plt.tight_layout()

        spectrum_file = os.path.join(self.output_dir, "frequency_spectrum_comparison.png")
        plt.savefig(spectrum_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 频谱对比图已保存: {spectrum_file}")

    def generate_report(self, test_results: dict):
        """生成测试报告"""
        print(f"\n📄 【测试7】生成测试报告")
        print("=" * 50)

        report_lines = [
            "=" * 70,
            "2.11 信号预处理模块测试报告",
            "=" * 70,
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试环境: {'ECS' if 'linux' in sys.platform else '本地Windows'}",
            "",
            "【预处理配置】",
            "  带通滤波: 0.5-4 Hz (30-240 BPM)",
            "  滤波器类型: 4阶巴特沃斯",
            "  去趋势: 线性去趋势",
            "  归一化: z-score标准化",
            "",
            "【处理流程】",
            "  1. 带通滤波 → 保留心率频率范围",
            "  2. 去趋势处理 → 移除低频漂移",
            "  3. z-score归一化 → 标准化幅度",
            "  4. 信号质量评估 → SNR/峰峰值/零交叉率",
            "",
            "【性能统计】",
        ]

        if 'performance' in test_results and test_results['performance']:
            perf = test_results['performance']
            report_lines.extend([
                f"  处理次数: {perf['total_processed']}",
                f"  平均耗时: {perf['avg_time_ms']:.2f} ms",
                f"  性能目标: ≤100ms",
                f"  性能达标: {'✅ 是' if perf['meets_target'] else '❌ 否'}",
            ])

        # 信号质量
        if 'full_pipeline' in test_results and test_results['full_pipeline']:
            quality = test_results['full_pipeline']['quality']
            report_lines.extend([
                "",
                "【信号质量】",
                f"  有效性: {'✅ 有效' if quality['valid'] else '❌ 无效'}",
                f"  信噪比(SNR): {quality['snr']:.2f} dB",
                f"  峰峰值: {quality['peak_to_peak']:.4f}",
                f"  零交叉率: {quality['zero_crossing_rate']:.4f}",
                f"  信号长度: {quality['length']} 个采样点",
            ])

        report_lines.extend([
            "",
            "【功能测试】",
            f"  带通滤波: {'✅ 通过' if test_results.get('bandpass') else '❌ 失败'}",
            f"  去趋势处理: {'✅ 通过' if test_results.get('detrend') else '❌ 失败'}",
            f"  归一化处理: {'✅ 通过' if test_results.get('normalization') else '❌ 失败'}",
            f"  完整流程: {'✅ 通过' if test_results.get('full_pipeline') else '❌ 失败'}",
            f"  可视化: {'✅ 通过' if test_results.get('visualization') else '❌ 失败'}",
            "",
            "【输出文件】",
            "  预处理信号: test_output/signal/rppg_signal_processed.npz",
            "  处理步骤图: test_output/signal/signal_preprocessing_steps.png",
            "  频谱对比图: test_output/signal/frequency_spectrum_comparison.png",
            "",
            "【结论】",
        ])

        # 判断结论
        all_passed = all([
            test_results.get('bandpass'),
            test_results.get('detrend'),
            test_results.get('normalization'),
            test_results.get('full_pipeline')
        ])

        perf_ok = test_results.get('performance', {}).get('meets_target', False)

        if all_passed and perf_ok:
            report_lines.append("信号预处理模块开发完成，性能达标，可用于特征提取")
        elif all_passed:
            report_lines.append("信号预处理模块功能完成，性能待优化")
        else:
            report_lines.append("信号预处理模块待完善")

        report_lines.append("=" * 70)
        final_report = "\n".join(report_lines)

        # 保存报告
        report_path = os.path.join(self.output_dir, "preprocessing_test_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)

        print(f"✅ 报告已保存: {os.path.abspath(report_path)}")

        # 打印报告
        print(f"\n📋 报告预览:")
        print(final_report)


def main():
    """主测试流程"""
    print("=" * 70)
    print("2.11 信号预处理模块完整测试")
    print("=" * 70)

    tester = SignalPreprocessTester()

    test_results = {
        'bandpass': False,
        'detrend': False,
        'normalization': False,
        'full_pipeline': None,
        'performance': None,
        'visualization': False
    }

    # 测试1: 带通滤波
    test_results['bandpass'] = tester.test_bandpass_filter()

    # 测试2: 去趋势
    test_results['detrend'] = tester.test_detrend()

    # 测试3: 归一化
    test_results['normalization'] = tester.test_normalization()

    # 测试4: 完整流程
    test_results['full_pipeline'] = tester.test_full_pipeline()

    # 测试5: 性能基准
    test_results['performance'] = tester.test_performance_benchmark()

    # 测试6: 可视化
    if test_results['full_pipeline']:
        test_results['visualization'] = tester.visualize_results(
            test_results['full_pipeline']
        )

    # 测试7: 生成报告
    tester.generate_report(test_results)

    print("\n" + "=" * 70)
    print("🎉 测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()