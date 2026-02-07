import psutil
import time
import os
import sys
from datetime import datetime
from typing import List, Dict
import threading

# 解决Windows/ECS路径编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

class ResourceMonitor:
    """系统资源监控器"""

    def __init__(self, interval: float = 0.1):
        """
        Args:
            interval: 采样间隔（秒）
        """
        self.interval = interval
        self.is_monitoring = False
        self.monitor_thread = None

        # 资源记录
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []

    def start(self):
        """开始监控"""
        if self.is_monitoring:
            print("⚠️  监控已在运行")
            return

        self.is_monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []

        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        print("✅ 资源监控已启动")

    def stop(self):
        """停止监控"""
        self.is_monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        print("✅ 资源监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            # 记录CPU使用率（无间隔采样，提升效率）
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)

            # 记录内存使用率
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)

            # 记录时间戳
            self.timestamps.append(time.time())

            # 等待下一次采样
            time.sleep(self.interval)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.cpu_usage:
            return {
                'cpu_avg': 0,
                'cpu_max': 0,
                'cpu_min': 0,
                'memory_avg': 0,
                'memory_max': 0,
                'memory_min': 0,
                'duration': 0,
                'samples': 0
            }

        # 增加numpy依赖异常处理（文档要求已安装，友好提示）
        try:
            import numpy as np
        except ImportError:
            print("❌ 未安装numpy，请执行：pip install numpy --break-system-packages")
            sys.exit(1)

        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0

        return {
            'cpu_avg': np.mean(self.cpu_usage),
            'cpu_max': np.max(self.cpu_usage),
            'cpu_min': np.min(self.cpu_usage),
            'memory_avg': np.mean(self.memory_usage),
            'memory_max': np.max(self.memory_usage),
            'memory_min': np.min(self.memory_usage),
            'duration': duration,
            'samples': len(self.cpu_usage)
        }

    def print_stats(self):
        """打印统计信息（严格遵循文档步骤5的输出格式）"""
        stats = self.get_stats()

        print(f"\n📊 资源使用统计:")
        print(f"  监控时长: {stats['duration']:.2f} 秒")
        print(f"  采样次数: {stats['samples']}")
        print(f"\n  CPU使用率:")
        print(f"    平均: {stats['cpu_avg']:.1f}%")
        print(f"    最大: {stats['cpu_max']:.1f}%")
        print(f"    最小: {stats['cpu_min']:.1f}%")
        print(f"\n  内存使用率:")
        print(f"    平均: {stats['memory_avg']:.1f}%")
        print(f"    最大: {stats['memory_max']:.1f}%")
        print(f"    最小: {stats['memory_min']:.1f}%")

    def save_report(self, output_path: str = "test_output/detection/resource_monitor.txt"):
        """保存监控报告（自动创建目录，兼容跨环境）"""
        stats = self.get_stats()

        report_lines = [
            "=" * 60,
            "系统资源监控报告 - 人脸检测模块",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"运行环境: {sys.platform} (Python {sys.version.split()[0]})",
            "",
            "【监控配置】",
            f"采样间隔: {self.interval} 秒",
            f"监控时长: {stats['duration']:.2f} 秒",
            f"采样次数: {stats['samples']}",
            "",
            "【CPU使用率】",
            f"平均: {stats['cpu_avg']:.1f}%",
            f"最大: {stats['cpu_max']:.1f}%",
            f"最小: {stats['cpu_min']:.1f}%",
            "",
            "【内存使用率】",
            f"平均: {stats['memory_avg']:.1f}%",
            f"最大: {stats['memory_max']:.1f}%",
            f"最小: {stats['memory_min']:.1f}%",
            "",
            "【系统信息】",
            f"CPU核心数: {psutil.cpu_count(logical=False)} (逻辑{psutil.cpu_count()})",
            f"总内存: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
            f"可用内存: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB",
            "=" * 60
        ]

        report_text = "\n".join(report_lines)
        # 自动创建输出目录，避免不存在报错
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n✅ 监控报告已保存: {os.path.abspath(output_path)}")

        return output_path


def monitor_face_detection_performance():
    """
    监控人脸检测性能（核心修复：导入路径+视频路径）
    严格遵循文档目录结构：modules/ 与 tests/ 同级
    """
    print("=" * 70)
    print("人脸检测性能监控")
    print("=" * 70)

    # ✅ 核心修复1：添加项目根目录到Python搜索路径
    # 获取tests目录的上级目录（即项目根目录），兼容Windows/ECS
    TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(TESTS_DIR)
    sys.path.insert(0, PROJECT_ROOT)

    # ✅ 核心修复2：从modules.detection导入FaceDetector（文档标准路径）
    try:
        from modules.detection.face_detector import FaceDetector
    except ImportError as e:
        print(f"❌ 导入人脸检测器失败：{str(e)}")
        print(f"💡 检查项1：项目根目录是否有 modules/detection/face_detector.py")
        print(f"💡 检查项2：face_detector.py中是否有FaceDetector类")
        print(f"💡 项目根目录：{PROJECT_ROOT}")
        return
    except Exception as e:
        print(f"❌ 检测器初始化异常：{str(e)}")
        return

    # 导入cv2（添加异常处理）
    try:
        import cv2
    except ImportError:
        print("❌ 未安装OpenCV，请执行：pip install opencv-python --break-system-packages")
        sys.exit(1)

    # ✅ 核心修复3：基于项目根目录拼接视频路径，兼容跨环境
    test_video = os.path.join(PROJECT_ROOT, "test_videos", "test_video_1.avi")
    if not os.path.exists(test_video):
        print(f"❌ 测试视频不存在: {os.path.abspath(test_video)}")
        print(f"💡 请按文档要求，在项目根目录创建test_videos并放入test_video_1.avi")
        return

    # 初始化检测器和监控器
    try:
        detector = FaceDetector()
        monitor = ResourceMonitor(interval=0.1)  # 文档默认采样间隔0.1秒
    except Exception as e:
        print(f"❌ 初始化失败：{str(e)}")
        return

    # 读取测试帧（最多100帧，文档步骤5要求）
    cap = cv2.VideoCapture(test_video, cv2.CAP_FFMPEG)  # 硬解码，提升速度
    test_frames = []

    for _ in range(100):
        ret, frame = cap.read()
        if ret:
            test_frames.append(frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()  # 释放cv2窗口，避免内存泄漏

    if not test_frames:
        print("❌ 未读取到任何视频帧，请检查视频文件是否损坏")
        return
    print(f"\n✅ 准备了 {len(test_frames)} 帧测试数据")

    # 开始监控+人脸检测
    print(f"\n🔍 开始监控人脸检测性能...")
    monitor.start()
    detection_count = 0
    start_time = time.time()

    # 执行人脸检测（按文档要求处理100帧）
    for i, frame in enumerate(test_frames):
        detections = detector.detect(frame)
        detection_count += len(detections)
        # 按文档步骤5输出进度（每20帧打印一次）
        if (i + 1) % 20 == 0:
            print(f"  处理进度: {i + 1}/{len(test_frames)}")

    # 计算耗时
    elapsed = time.time() - start_time

    # 停止监控（等待0.5秒，确保最后采样完成）
    time.sleep(0.5)
    monitor.stop()

    # 打印检测结果（严格遵循文档步骤5的输出格式）
    print(f"\n✅ 检测完成")
    print(f"  处理帧数: {len(test_frames)}")
    print(f"  检测人脸数: {detection_count}")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均帧率: {len(test_frames) / elapsed:.2f} FPS")

    # 打印+保存资源统计
    monitor.print_stats()
    monitor.save_report(os.path.join(PROJECT_ROOT, "test_output", "detection", "resource_monitor.txt"))

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 运行前强制内存回收，避免初始内存占用过高
    import gc
    gc.collect()
    # 执行监控
    monitor_face_detection_performance()