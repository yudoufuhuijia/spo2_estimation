"""
2.12 特征提取 - 一键快速测试
功能：5分钟完成基础功能验证
修复点：1. 修正项目根目录路径 2. 补充信号不存在的解决步骤 3. 优化跨平台路径兼容
"""
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# 核心修复：项目根目录 = tests目录的上级目录（spo2_estimation）
project_root = str(Path(__file__).parent.parent.resolve())  # 原代码是parent，改为parent.parent
sys.path.insert(0, project_root)

print("=" * 70)
print("2.12 特征提取 - 一键快速测试")
print("=" * 70)

# 导入模块
print(f"\n【1/4】导入模块...")
try:
    from modules.features.feature_extractor import FeatureExtractor
    print("✅ feature_extractor 导入成功")
except ImportError as e:
    print(f"❌ feature_extractor 导入失败: {e}")
    print(f"💡 解决：检查modules/features下是否有feature_extractor.py文件")
    sys.exit(1)

# 加载预处理信号（优化路径拼接，跨平台兼容）
print(f"\n【2/4】加载预处理信号...")
# 正确路径：spo2_estimation/test_output/signal/...
signal_dir = os.path.join(project_root, "test_output", "signal")
signal_file = os.path.join(signal_dir, "rppg_signal_processed.npz")

# 信号不存在时给出详细解决步骤（原代码仅提示不存在，无解决方法）
if not os.path.exists(signal_file):
    print(f"❌ 预处理信号不存在: {signal_file}")
    print(f"\n💡 快速解决步骤：")
    print(f"   1. 进入项目根目录：cd {project_root}")
    print(f"   2. 激活环境：conda activate spo2_env")
    print(f"   3. 运行预处理脚本：python tests/quick_test_preprocess.py")
    sys.exit(1)

# 加载信号并做异常捕获
try:
    data = np.load(signal_file, allow_pickle=True)
    processed_signal = data['processed_signal']
    # 兼容fps字段存在/不存在的情况
    fps = float(data['fps']) if 'fps' in data and data['fps'] else 30.0
    print(f"✅ 成功加载信号")
    print(f"   信号长度: {len(processed_signal)} 个采样点")
    print(f"   采样率: {fps:.1f} Hz")
except Exception as e:
    print(f"❌ 信号文件加载失败: {e}")
    print(f"💡 解决：重新运行预处理脚本生成有效信号")
    sys.exit(1)

# 初始化特征提取器并提取特征
print(f"\n【3/4】提取特征...")
try:
    extractor = FeatureExtractor(fps=int(fps))
    features = extractor.extract_features(processed_signal, return_intermediate=True)
except Exception as e:
    print(f"❌ 特征提取初始化/执行失败: {e}")
    sys.exit(1)

if not features.get('valid', False):
    print(f"❌ 特征提取失败（特征有效性验证不通过）")
    sys.exit(1)

print(f"✅ 特征提取完成")
print(f"   耗时: {features['extraction_time_ms']:.2f} ms {'✅' if features['extraction_time_ms'] <= 200 else '⚠️'}")

# 显示特征摘要
print(f"\n【4/4】特征摘要:")

print(f"\n📊 RoR特征:")
ror = features['ror_features']
print(f"   RoR比值: {ror['ror']:.6f}")
print(f"   AC分量: {ror['ac_component']:.6f}")
print(f"   DC分量: {ror['dc_component']:.6f}")
print(f"   检测到R峰: {ror['n_peaks']} 个")

print(f"\n💓 心率特征:")
hr = features['hr_features']
if hr.get('available'):
    print(f"   平均心率: {hr['mean_hr']:.2f} BPM")
    print(f"   心率标准差: {hr['std_hr']:.2f} BPM")
else:
    print(f"   ⚠️  心率特征不可用（R峰数量<2，需更长/质量更好的信号）")

print(f"\n🔍 信号质量:")
quality = features['quality_features']
print(f"   SNR: {quality['snr']:.2f} dB")
print(f"   偏度: {quality['skewness']:.4f}")
print(f"   峰度: {quality['kurtosis']:.4f}")

print(f"\n🌊 频域特征:")
freq = features['frequency_features']
if freq.get('available'):
    print(f"   主频率: {freq['dominant_freq_bpm']:.2f} BPM")
    print(f"   功率占比: {freq['dominant_power_ratio']:.4f}")
else:
    print(f"   ⚠️  频域特征不可用（信号质量差/长度不足）")

# 保存特征（自动创建features目录，避免目录不存在报错）
features_dir = os.path.join(project_root, "test_output", "features")
os.makedirs(features_dir, exist_ok=True)  # 新增：目录不存在则自动创建
feature_save_file = os.path.join(features_dir, "features_quick_test.npz")

# 保存特征并做异常捕获
try:
    np.savez(
        feature_save_file,
        features=features,
        signal=processed_signal,
        save_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
except Exception as e:
    print(f"\n⚠️  特征保存失败: {e}，不影响测试结果但无法留存特征")
else:
    print(f"\n📁 特征已保存至: {feature_save_file}")

# 性能统计
stats = extractor.get_performance_stats()
print(f"\n⚡ 性能统计:")
print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'} (目标≤200ms)")

# 测试报告
print(f"\n" + "=" * 70)
print("测试报告")
print("=" * 70)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"项目根目录: {project_root}")
print(f"\n【功能测试】")
print(f"  RoR特征: ✅ 成功")
print(f"  心率特征: {'✅ 成功' if hr.get('available') else '⚠️  需更多R峰数据'}")
print(f"  质量特征: ✅ 成功")
print(f"  频域特征: {'✅ 成功' if freq.get('available') else '⚠️  信号不满足要求'}")
print(f"\n【性能测试】")
print(f"  单次提取耗时: {stats['avg_time_ms']:.2f} ms")
print(f"  性能目标: ≤200ms")
print(f"  性能达标: {'✅ 是' if stats['meets_target'] else '❌ 否'}")
print(f"\n【结论】")
if stats['meets_target']:
    print(f"特征提取模块测试通过✅")
else:
    print(f"功能正常，性能待优化⚠️")
print("=" * 70)

print(f"\n🎉 快速测试完成！可继续执行2.13模型搭建步骤")