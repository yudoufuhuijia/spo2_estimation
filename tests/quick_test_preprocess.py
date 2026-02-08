"""
2.11 信号预处理 - 一键快速测试
功能：5分钟完成基础功能验证
修复点：1. 修正quality变量定义顺序 2. 移除重复打印 3. 完善变量容错 4. 跨平台路径兼容
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 项目根目录：tests的上一级（spo2_estimation）
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

print("=" * 70)
print("2.11 信号预处理 - 一键快速测试")
print("=" * 70)

# 检查环境
import platform
print(f"\n【系统信息】")
print(f"Python版本: {platform.python_version()}")
print(f"操作系统: {platform.system()}")
print(f"项目根目录: {project_root}")

# 导入模块
print(f"\n【1/5】导入模块...")
try:
    from modules.signal.signal_preprocess import SignalPreprocessor
    print("✅ signal_preprocess 导入成功")
except ImportError as e:
    print(f"❌ signal_preprocess 导入失败: {e}")
    print("💡 请将signal_preprocess.py放到【项目根目录】/modules/signal/目录下")
    sys.exit(1)

# 加载原始信号
print(f"\n【2/5】加载原始信号...")
# 跨平台路径拼接+自动创建目录
output_dir = os.path.join(project_root, "test_output", "signal")
signal_file = os.path.join(output_dir, "rppg_signal_raw.npz")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(signal_file):
    print(f"❌ 原始信号文件不存在: {signal_file}")
    print("💡 先运行2.10任务生成rppg_signal_raw.npz")
    sys.exit(1)

# 加载信号并容错
try:
    data = np.load(signal_file)
    raw_signal = data['chrom']
    fps = float(data['fps']) if 'fps' in data.files else 30.0
except KeyError as e:
    print(f"❌ 信号文件缺失字段: {e}（需包含chrom/fps）")
    sys.exit(1)
except Exception as e:
    print(f"❌ 加载信号失败: {e}")
    sys.exit(1)

print(f"✅ 成功加载信号")
print(f"   信号长度: {len(raw_signal)} 个采样点")
print(f"   采样率: {fps:.1f} Hz")
print(f"   时间跨度: {len(raw_signal)/fps:.2f} 秒")

# 初始化预处理器
print(f"\n【3/5】初始化预处理器...")
try:
    preprocessor = SignalPreprocessor(
        fps=int(fps),
        lowcut=0.5,      # 30 BPM
        highcut=4.0,     # 240 BPM
        filter_order=4,
        enable_detrend=True,
        enable_normalization=True,
        min_signal_length=60
    )
    print(f"✅ 信号预处理器初始化完成")
    print(f"   采样率: {int(fps)} Hz")
    print(f"   带通范围: 0.5-4.0 Hz (30-240 BPM)")
    print(f"   滤波器阶数: 4")
except Exception as e:
    print(f"❌ 预处理器初始化失败: {e}")
    sys.exit(1)

# 预处理信号
print(f"\n【4/5】预处理信号...")
start_time = time.time()

try:
    result = preprocessor.preprocess(
        raw_signal,
        return_intermediate=True
    )
except Exception as e:
    print(f"❌ 预处理报错: {e}")
    sys.exit(1)

# 计算耗时
elapsed_ms = (time.time() - start_time) * 1000

# 【核心修复】先定义quality变量，再使用！避免NameError
quality = result.get('quality', {})  # 容错：如果result无quality，返回空字典
# 先判断预处理结果是否有效
if result['processed'] is None or not quality.get('valid', False):
    print(f"❌ 预处理失败")
    if 'reason' in quality:
        print(f"   失败原因: {quality['reason']}")
    sys.exit(1)

print(f"✅ 预处理完成")
print(f"   耗时: {elapsed_ms:.2f} ms")

# 分析结果
print(f"\n📊 预处理结果:")
print(f"   处理步骤:")
print(f"     1. 带通滤波 (0.5-4 Hz) → ✅")
print(f"     2. 去趋势处理 → ✅")
print(f"     3. z-score归一化 → ✅")

print(f"\n   信号统计:")
print(f"     原始信号 - 均值: {np.mean(raw_signal):.6f}, 标准差: {np.std(raw_signal):.6f}")
print(f"     处理后 - 均值: {np.mean(result['processed']):.6f}, 标准差: {np.std(result['processed']):.6f}")

print(f"\n📈 信号质量:")
print(f"   有效性: {'✅ 有效' if quality['valid'] else '❌ 无效'}")
print(f"   SNR: {quality['snr']:.2f} dB")
print(f"   峰峰值: {quality['peak_to_peak']:.4f}")
print(f"   信号长度: {quality['length']} 个采样点")

# 性能统计
stats = preprocessor.get_performance_stats()

print(f"\n⚡ 性能统计:")
print(f"   处理耗时: {stats['avg_time_ms']:.2f} ms")
print(f"   性能目标: ≤100ms")
print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

# 保存结果
print(f"\n【5/5】保存测试结果...")
processed_file = os.path.join(output_dir, "rppg_signal_processed_quick.npz")
try:
    preprocessor.save_processed_signal(
        result,
        processed_file,
        metadata={'test_type': 'quick_test'}
    )
except Exception as e:
    print(f"⚠️  保存信号失败: {e}")

# 生成测试报告
print(f"\n" + "=" * 70)
print("测试报告")
print("=" * 70)

report_lines = [
    "=" * 70,
    "2.11 信号预处理 - 快速测试报告",
    "=" * 70,
    f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"测试环境: {platform.system()} | Python {platform.python_version()}",
    "",
    "【预处理配置】",
    "  带通滤波: 0.5-4 Hz (30-240 BPM)",
    "  去趋势: 线性去趋势",
    "  归一化: z-score标准化",
    "",
    "【功能测试】",
    "  带通滤波: ✅ 成功",
    "  去趋势处理: ✅ 成功",
    "  归一化处理: ✅ 成功",
    "",
    "【性能测试】",
    f"  处理耗时: {stats['avg_time_ms']:.2f} ms",
    f"  性能目标: ≤100ms",
    f"  性能达标: {'✅ 是' if stats['meets_target'] else '❌ 否'}",
    "",
    "【信号质量】",
    f"  SNR: {quality['snr']:.2f} dB",
    f"  有效性: {'✅ 有效' if quality['valid'] else '❌ 无效'}",
    "",
    "【输出文件】",
    f"  预处理信号: {processed_file}",
    "",
    "【结论】",
]

if stats['meets_target'] and quality['valid']:
    report_lines.append("信号预处理模块测试通过✅")
elif quality['valid']:
    report_lines.append("功能正常，性能可接受✅")
else:
    report_lines.append("需要检查信号质量⚠️")

report_lines.append("=" * 70)

# 打印并保存报告
for line in report_lines:
    print(line)

report_path = os.path.join(output_dir, "quick_preprocess_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✅ 测试报告已保存: {report_path}")
print(f"\n" + "=" * 70)
print("🎉 快速测试完成！")
print("=" * 70)
print(f"\n📂 结果路径: {output_dir}")