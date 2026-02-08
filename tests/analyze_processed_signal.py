import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 核心：获取项目根目录（tests的上一级），拼接信号文件的正确路径
project_root = Path(__file__).parent.parent  # 对应D:\bishe\spo2_estimation
signal_file = os.path.join(project_root, "test_output", "signal", "rppg_signal_processed.npz")
# 兼容快速测试生成的文件（如果需要切换，解开下面注释即可）
# signal_file = os.path.join(project_root, "test_output", "signal", "rppg_signal_processed_quick.npz")

# 检查文件是否存在，友好提示
if not os.path.exists(signal_file):
    print(f"❌ 未找到预处理信号文件，路径：{signal_file}")
    print("💡 请先运行2.11的基础测试生成该文件，执行命令：")
    # 修复转义序列：用原始字符串r""包裹路径，避免\s被识别为转义字符
    print(r"   cd D:\bishe\spo2_estimation && python modules/signal/signal_preprocess.py")
    exit(1)

# 核心修复：开启allow_pickle=True，支持加载字典/object类型的quality指标
# 编码指定latin1，兼容npz中不同类型数据的存储
data = np.load(signal_file, allow_pickle=True, encoding='latin1')

print("预处理信号文件包含：")
# 修复遍历报错：逐个处理key，对object类型做容错，避免直接取值触发报错
for key in data.files:
    try:
        # 尝试获取形状，失败则显示为scalar
        shape = data[key].shape if hasattr(data[key], 'shape') else 'scalar'
        print(f"  {key}: shape = {shape}")
    except:
        print(f"  {key}: type = object (dict/quality info)")

# 提取信号和质量指标（单独处理，避免遍历报错）
processed_signal = data['processed_signal']
# 直接提取quality，因已开启allow_pickle，可正常解析字典
quality = data['quality'].item() if 'quality' in data.files else {}

# 打印信号质量（做容错，避免key缺失）
print(f"\n信号质量:")
print(f"  有效性: {quality.get('valid', '未知')}")
print(f"  SNR: {quality.get('snr', 0.0):.2f} dB")
print(f"  峰峰值: {quality.get('peak_to_peak', 0.0):.4f}")
if 'zero_crossing_rate' in quality:
    print(f"  零交叉率: {quality['zero_crossing_rate']:.4f}")

# 绘图并保存到test_output/signal目录（方便归类）
save_img_path = os.path.join(project_root, "test_output", "signal", "processed_signal_check.png")
plt.figure(figsize=(10, 4))
plt.plot(processed_signal, 'r-', linewidth=1.5)
plt.title('Processed rPPG Signal (z-score)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (z-score)')
plt.grid(True, alpha=0.3)
plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
plt.close()  # 关闭画布，避免matplotlib警告

print(f"\n✅ 信号检查图已保存至: {save_img_path}")
print(f"\n📌 预处理信号基本信息:")
print(f"   信号长度: {len(processed_signal)} 个采样点")
print(f"   信号均值: {np.mean(processed_signal):.6f} (归一化后应接近0)")
print(f"   信号标准差: {np.std(processed_signal):.6f} (归一化后应接近1)")