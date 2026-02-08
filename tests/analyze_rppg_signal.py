import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  # 新增：适配跨平台路径

# 关键：按文档规范获取信号文件路径（项目根目录→test_output→signal）
project_root = str(Path(__file__).parent.parent.resolve())  # 定位项目根目录（tests的上一级）
signal_file_path = Path(project_root) / "test_output" / "signal" / "rppg_signal_raw.npz"  # 拼接信号文件路径

# 加载信号数据（先检查文件是否存在，符合文档“验证文件”的习惯）
if not signal_file_path.exists():
    print(f"❌ 信号文件不存在，请先运行完整测试生成！")
    print(f"   运行命令：python tests/test_signal_extraction.py")
    exit(1)

data = np.load(signal_file_path)

# 以下为原脚本内容，无需修改
print("信号文件包含：")
for key in data.files:
    print(f"  {key}: shape = {data[key].shape}")

# 提取CHROM信号
chrom_signal = data['chrom']
timestamps = data['timestamps']
fps = data['fps']

# 简单分析
print(f"\n信号统计:")
print(f"  采样点数: {len(chrom_signal)}")
print(f"  时间跨度: {timestamps[-1] - timestamps[0]:.2f} 秒")
print(f"  均值: {np.mean(chrom_signal):.4f}")
print(f"  标准差: {np.std(chrom_signal):.4f}")

# 估算心率（简单方法）
if len(chrom_signal) > 60:
    # FFT
    fft_result = np.fft.fft(chrom_signal)
    fft_freq = np.fft.fftfreq(len(chrom_signal), 1 / fps)

    # 找到主频率（50-150 BPM范围，符合文档“正常心率范围”）
    valid_mask = (fft_freq * 60 >= 50) & (fft_freq * 60 <= 150)
    valid_freq_bpm = fft_freq[valid_mask] * 60
    valid_magnitude = np.abs(fft_result[valid_mask])

    peak_idx = np.argmax(valid_magnitude)
    estimated_hr = valid_freq_bpm[peak_idx]

    print(f"\n估算心率: {estimated_hr:.1f} BPM")
else:
    print(f"\n⚠️  信号长度不足60个采样点，无法估算心率（需运行完整测试处理200+帧）")