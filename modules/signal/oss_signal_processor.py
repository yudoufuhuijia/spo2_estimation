"""
OSS视频流rPPG信号提取 - oss_signal_processor.py
功能：逐帧处理OSS视频流，提取rPPG信号
特点：
1. 支持OSS视频流式读取
2. 自动缓存管理
3. 实时信号提取
4. 结果存储到OSS
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# 项目根目录
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

try:
    from data_process.VideoReader import VideoReader
    from modules.detection.face_detector import FaceDetector
    from modules.roi.roi_extractor import ROIExtractor
    from modules.signal.chrom_extractor import CHROMExtractor

    HAS_ALL_MODULES = True
except ImportError as e:
    print(f"⚠️  模块导入失败: {e}")
    HAS_ALL_MODULES = False


class OSSSignalProcessor:
    """
    OSS视频流rPPG信号处理器

    完整流程：
    1. 从OSS读取视频流
    2. 逐帧人脸检测
    3. 提取ROI区域
    4. 计算rPPG信号
    5. 保存结果到OSS
    """

    def __init__(
            self,
            output_dir: str = "test_output/signal",
            enable_oss: bool = False,  # 是否启用OSS（本地测试设为False）
            save_interval: int = 100  # 每处理N帧保存一次
    ):
        """
        初始化处理器

        Args:
            output_dir: 输出目录
            enable_oss: 是否启用OSS
            save_interval: 保存间隔（帧数）
        """
        self.output_dir = output_dir
        self.enable_oss = enable_oss
        self.save_interval = save_interval

        os.makedirs(output_dir, exist_ok=True)

        # 初始化模块
        print("🔧 初始化处理模块...")
        self.face_detector = FaceDetector(method='mtcnn')
        self.roi_extractor = ROIExtractor()
        self.chrom_extractor = CHROMExtractor(
            fps=30,
            window_size=300,
            use_forehead_only=True
        )
        print("✅ 所有模块初始化完成")

        # 统计信息
        self.stats = {
            'total_frames': 0,
            'face_detected': 0,
            'roi_extracted': 0,
            'signal_extracted': 0,
            'processing_time': 0.0
        }

    def process_video(
            self,
            video_path: str,
            max_frames: Optional[int] = None,
            verbose: bool = True
    ) -> Dict:
        """
        处理单个视频

        Args:
            video_path: 视频路径（OSS或本地）
            max_frames: 最大处理帧数（None=全部）
            verbose: 是否打印详细信息

        Returns:
            处理结果字典
        """
        if verbose:
            print(f"\n📹 开始处理视频: {video_path}")

        # 打开视频
        if self.enable_oss:
            # 使用VideoReader读取OSS视频
            video_reader = VideoReader(video_path)
            cap = cv2.VideoCapture(video_path)  # 需要适配
        else:
            # 本地视频
            if not os.path.exists(video_path):
                print(f"❌ 视频不存在: {video_path}")
                return {}

            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ 无法打开视频")
            return {}

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if verbose:
            print(f"   帧率: {fps} FPS")
            print(f"   总帧数: {total_frames}")
            if max_frames:
                print(f"   处理帧数: {max_frames}")

        # 重置提取器
        self.chrom_extractor.reset()

        # 逐帧处理
        frame_count = 0
        start_time = time.time()

        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break

            # 检查是否达到最大帧数
            if max_frames and frame_count >= max_frames:
                break

            # 处理单帧
            self._process_frame(frame, frame_count / fps)

            frame_count += 1

            # 定期打印进度
            if verbose and frame_count % 50 == 0:
                self._print_progress(frame_count, fps)

            # 定期保存中间结果
            if frame_count % self.save_interval == 0:
                self._save_intermediate(frame_count)

        cap.release()

        # 最终保存
        elapsed = time.time() - start_time
        self.stats['processing_time'] = elapsed

        result = self._save_final_results(video_path, fps)

        if verbose:
            self._print_final_stats(elapsed, fps)

        return result

    def _process_frame(self, frame: np.ndarray, timestamp: float):
        """
        处理单帧

        Args:
            frame: 视频帧
            timestamp: 时间戳（秒）
        """
        self.stats['total_frames'] += 1

        # 人脸检测
        detections = self.face_detector.detect(frame)
        if not detections:
            return

        self.stats['face_detected'] += 1

        # ROI提取
        rois = self.roi_extractor.extract_rois(frame, detections[0])
        if not rois or 'forehead' not in rois:
            return

        self.stats['roi_extracted'] += 1

        # rPPG信号提取
        signal_value = self.chrom_extractor.extract_from_rois(rois, timestamp)
        if signal_value is not None:
            self.stats['signal_extracted'] += 1

    def _print_progress(self, frame_count: int, fps: float):
        """打印处理进度"""
        elapsed = time.time() - time.time()  # 临时
        processing_fps = frame_count / max(elapsed, 0.001)

        print(f"   处理进度: {frame_count} 帧 | "
              f"检测: {self.stats['face_detected']} | "
              f"信号: {self.stats['signal_extracted']} | "
              f"速度: {processing_fps:.1f} FPS")

    def _save_intermediate(self, frame_count: int):
        """保存中间结果（可选）"""
        # 可以实现增量保存逻辑
        pass

    def _save_final_results(
            self,
            video_path: str,
            fps: float
    ) -> Dict:
        """
        保存最终结果

        Returns:
            结果字典
        """
        # 获取信号数据
        signals = self.chrom_extractor.get_signal_buffer()
        quality = self.chrom_extractor.get_signal_quality()

        # 生成输出文件名
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_prefix = f"{self.output_dir}/{video_name}_{timestamp}"

        # 保存信号数据（.npz格式）
        signal_file = f"{output_prefix}_signal.npz"
        np.savez(
            signal_file,
            raw_R=signals['raw_R'],
            raw_G=signals['raw_G'],
            raw_B=signals['raw_B'],
            chrom=signals['chrom'],
            timestamps=signals['timestamps'],
            fps=fps,
            stats=self.stats,
            quality=quality
        )

        print(f"\n✅ 信号数据已保存: {signal_file}")

        # 保存统计信息（.txt格式）
        stats_file = f"{output_prefix}_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("rPPG信号提取统计\n")
            f.write("=" * 60 + "\n")
            f.write(f"视频文件: {video_path}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n【处理统计】\n")
            f.write(f"总帧数: {self.stats['total_frames']}\n")
            f.write(f"检测到人脸: {self.stats['face_detected']}\n")
            f.write(f"提取ROI: {self.stats['roi_extracted']}\n")
            f.write(f"提取信号: {self.stats['signal_extracted']}\n")
            f.write(f"处理时长: {self.stats['processing_time']:.2f}秒\n")
            f.write("\n【信号质量】\n")
            f.write(f"信噪比: {quality['snr']:.2f} dB\n")
            f.write(f"信号长度: {quality['signal_length']}\n")
            f.write(f"信号有效: {'是' if quality['is_valid'] else '否'}\n")
            f.write("=" * 60 + "\n")

        print(f"✅ 统计信息已保存: {stats_file}")

        return {
            'signal_file': signal_file,
            'stats_file': stats_file,
            'quality': quality,
            'stats': self.stats
        }

    def _print_final_stats(self, elapsed: float, fps: float):
        """打印最终统计"""
        print(f"\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"检测到人脸: {self.stats['face_detected']} "
              f"({self.stats['face_detected'] / self.stats['total_frames'] * 100:.1f}%)")
        print(f"提取ROI: {self.stats['roi_extracted']}")
        print(f"提取信号: {self.stats['signal_extracted']}")
        print(f"\n处理时长: {elapsed:.2f} 秒")
        print(f"平均处理速度: {self.stats['total_frames'] / elapsed:.2f} FPS")

        quality = self.chrom_extractor.get_signal_quality()
        print(f"\n信号质量:")
        print(f"  SNR: {quality['snr']:.2f} dB")
        print(f"  长度: {quality['signal_length']} 个采样点")
        print(f"  有效: {'✅' if quality['is_valid'] else '❌'}")
        print("=" * 60)


# ===================== 测试代码 =====================
def test_oss_signal_processor():
    """测试OSS信号处理器"""
    print("=" * 70)
    print("📝 OSS视频流rPPG信号提取测试")
    print("=" * 70)

    # 初始化处理器
    processor = OSSSignalProcessor(
        output_dir="../../test_output/signal",
        enable_oss=False,  # 本地测试
        save_interval=100
    )

    # 处理测试视频
    test_video = "../../test_videos/test_video_1.avi"

    result = processor.process_video(
        video_path=test_video,
        max_frames=200,  # 处理200帧
        verbose=True
    )

    if result:
        print(f"\n✅ 处理完成！")
        print(f"\n输出文件:")
        for key, value in result.items():
            if isinstance(value, str):
                print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_oss_signal_processor()