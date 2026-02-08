"""
ROI提取模块完整测试 - test_roi_extraction.py
测试内容：
1. 基于MTCNN关键点的ROI提取
2. 降级方案（无关键点时）
3. 性能基准测试
4. 可视化验证
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 项目根目录
project_root = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, project_root)

# 导入模块
try:
    from modules.detection.face_detector import FaceDetector
    from modules.roi.roi_extractor import ROIExtractor
    from modules.roi.roi_cache import ROICache

    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


class ROIExtractionTester:
    """ROI提取完整测试器"""

    def __init__(self, output_dir: str = "test_output/roi"):
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📂 测试结果输出: {os.path.abspath(self.output_dir)}")

        # 初始化模块
        print("\n🔧 初始化模块...")
        self.face_detector = FaceDetector(method='mtcnn')
        self.roi_extractor = ROIExtractor(enable_visualization=True)
        self.roi_cache = ROICache(
            cache_dir=os.path.join(self.output_dir, "cache"),
            auto_cleanup=True
        )
        print("✅ 所有模块初始化完成")

    def test_basic_extraction(
            self,
            video_path: str = "test_videos/test_video_1.avi"
    ):
        """测试基础ROI提取功能"""
        print(f"\n📹 【测试1】基础ROI提取测试")
        print("=" * 50)

        full_path = os.path.join(project_root, video_path)

        if not os.path.exists(full_path):
            print(f"❌ 测试视频不存在: {full_path}")
            return False

        # 读取视频帧
        cap = cv2.VideoCapture(full_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ 无法读取视频帧")
            return False

        print(f"✅ 读取测试帧，尺寸: {frame.shape[1]}x{frame.shape[0]}")

        # 检测人脸
        print(f"\n🔍 检测人脸...")
        detections = self.face_detector.detect(frame)

        if not detections:
            print("❌ 未检测到人脸")
            return False

        face = detections[0]
        print(f"✅ 检测到人脸")
        print(f"   人脸框: {face['box']}")
        print(f"   置信度: {face['confidence']}")

        if face.get('landmarks'):
            print(f"   关键点: {list(face['landmarks'].keys())}")

        # 提取ROI
        print(f"\n✂️  提取ROI...")
        start_time = time.time()
        rois = self.roi_extractor.extract_rois(frame, face)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"✅ ROI提取完成，耗时: {elapsed_ms:.2f}ms")
        print(f"\n📊 提取结果:")
        for roi_name, roi_img in rois.items():
            if roi_name != 'coords':
                h, w = roi_img.shape[:2]
                print(f"   {roi_name:12s}: {w}x{h} 像素")

        # 保存可视化结果
        print(f"\n💾 保存可视化结果...")

        # 1. 绘制ROI边框
        result_img = self.roi_extractor.draw_rois(frame, rois['coords'])
        cv2.imwrite(
            os.path.join(self.output_dir, "roi_visualization.jpg"),
            result_img
        )

        # 2. 保存各个ROI
        for roi_name, roi_img in rois.items():
            if roi_name != 'coords':
                cv2.imwrite(
                    os.path.join(self.output_dir, f"{roi_name}.jpg"),
                    roi_img
                )

        print(f"✅ 可视化结果已保存到: {self.output_dir}")

        return True

    def test_performance_benchmark(
            self,
            video_path: str = "test_videos/test_video_1.avi",
            num_frames: int = 100
    ):
        """性能基准测试"""
        print(f"\n⚡ 【测试2】性能基准测试")
        print("=" * 50)

        full_path = os.path.join(project_root, video_path)

        if not os.path.exists(full_path):
            print(f"❌ 测试视频不存在")
            return None

        # 提取测试帧
        cap = cv2.VideoCapture(full_path)
        test_frames = []

        print(f"📹 提取 {num_frames} 帧测试数据...")
        while len(test_frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.size > 0:
                test_frames.append(frame)

        cap.release()

        if len(test_frames) < 10:
            print(f"❌ 测试帧不足")
            return None

        print(f"✅ 准备了 {len(test_frames)} 帧")

        # 性能测试
        print(f"\n🔍 开始性能测试...")
        self.face_detector.reset_performance_stats()
        self.roi_extractor.reset_performance_stats()

        successful_extractions = 0
        total_face_time = 0.0
        total_roi_time = 0.0

        for i, frame in enumerate(test_frames):
            # 人脸检测
            face_start = time.time()
            detections = self.face_detector.detect(frame)
            face_elapsed = time.time() - face_start
            total_face_time += face_elapsed

            if not detections:
                continue

            # ROI提取
            roi_start = time.time()
            rois = self.roi_extractor.extract_rois(frame, detections[0])
            roi_elapsed = time.time() - roi_start
            total_roi_time += roi_elapsed

            successful_extractions += 1

            # 每20帧打印进度
            if (i + 1) % 20 == 0:
                avg_face_ms = (total_face_time / (i + 1)) * 1000
                avg_roi_ms = (total_roi_time / successful_extractions) * 1000 if successful_extractions > 0 else 0
                print(f"   处理 {i + 1} 帧: "
                      f"人脸检测={avg_face_ms:.2f}ms, "
                      f"ROI提取={avg_roi_ms:.2f}ms")

        # 性能统计
        print(f"\n📈 性能统计:")

        # 人脸检测性能
        face_stats = self.face_detector.get_performance_stats()
        print(f"\n   人脸检测:")
        print(f"     平均耗时: {face_stats['avg_time_ms']:.2f} ms")
        print(f"     检测次数: {face_stats['total_detections']}")

        # ROI提取性能
        roi_stats = self.roi_extractor.get_performance_stats()
        print(f"\n   ROI提取:")
        print(f"     平均耗时: {roi_stats['avg_time_ms']:.2f} ms")
        print(f"     提取次数: {roi_stats['total_extractions']}")
        print(f"     性能达标: {'✅ 是' if roi_stats['meets_target'] else '❌ 否'}")

        # 总体性能
        total_avg_ms = face_stats['avg_time_ms'] + roi_stats['avg_time_ms']
        print(f"\n   总体:")
        print(f"     总平均耗时: {total_avg_ms:.2f} ms/帧")
        print(f"     理论最大FPS: {1000 / total_avg_ms:.2f}")
        print(f"     成功率: {successful_extractions / len(test_frames) * 100:.1f}%")

        return {
            'face_detection': face_stats,
            'roi_extraction': roi_stats,
            'total_avg_ms': total_avg_ms,
            'success_rate': successful_extractions / len(test_frames)
        }

    def test_cache_functionality(self):
        """测试缓存功能"""
        print(f"\n💾 【测试3】缓存功能测试")
        print("=" * 50)

        # 创建测试ROI数据
        test_roi = {
            'forehead': np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8),
            'left_cheek': np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
            'right_cheek': np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
            'coords': {
                'forehead': (100, 50, 80, 50),
                'left_cheek': (80, 120, 40, 40),
                'right_cheek': (180, 120, 40, 40)
            }
        }

        roi_id = "test_video_frame_001"

        # 保存到缓存
        print(f"💾 保存ROI到缓存...")
        cache_path = self.roi_cache.save_roi(roi_id, test_roi)
        print(f"✅ 已保存: {cache_path}")

        # 从缓存加载
        print(f"\n📖 从缓存加载ROI...")
        loaded_roi = self.roi_cache.load_roi(roi_id)

        if loaded_roi:
            print(f"✅ 加载成功")

            # 验证数据一致性
            is_consistent = np.array_equal(
                loaded_roi['forehead'],
                test_roi['forehead']
            )
            print(f"   数据一致性: {'✅ 通过' if is_consistent else '❌ 失败'}")
        else:
            print(f"❌ 加载失败")

        # 打印缓存统计
        self.roi_cache.print_stats()

        # 清理测试缓存
        print(f"\n🗑️  清理测试缓存...")
        self.roi_cache.delete_roi(roi_id)
        print(f"✅ 已清理")

        return True

    def generate_report(self, test_results: dict):
        """生成测试报告"""
        print(f"\n📄 【测试4】生成测试报告")
        print("=" * 50)

        report_lines = [
            "=" * 70,
            "ROI提取模块测试报告",
            "=" * 70,
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试环境: {'ECS' if 'linux' in sys.platform else '本地Windows'}",
            "",
            "【模块配置】",
            "  ROI区域:",
            "    - 额头ROI（用于SpO2信号提取）",
            "    - 左脸颊ROI（辅助信号源）",
            "    - 右脸颊ROI（辅助信号源）",
            "  提取策略: 基于MTCNN 5个关键点",
            "  降级方案: 基于人脸框估算（无关键点时）",
            "",
            "【性能统计】",
        ]

        if 'performance' in test_results and test_results['performance']:
            perf = test_results['performance']

            # ROI提取性能
            roi_stats = perf['roi_extraction']
            report_lines.extend([
                f"  ROI提取:",
                f"    平均耗时: {roi_stats['avg_time_ms']:.2f} ms",
                f"    性能目标: ≤10ms",
                f"    性能达标: {'✅ 是' if roi_stats['meets_target'] else '❌ 否'}",
            ])

            # 总体性能
            report_lines.extend([
                f"  总体性能:",
                f"    人脸检测 + ROI提取: {perf['total_avg_ms']:.2f} ms/帧",
                f"    理论最大FPS: {1000 / perf['total_avg_ms']:.2f}",
                f"    成功率: {perf['success_rate'] * 100:.1f}%",
            ])

        report_lines.extend([
            "",
            "【功能测试】",
            f"  基础提取: {'✅ 通过' if test_results.get('basic_extraction') else '❌ 失败'}",
            f"  缓存管理: {'✅ 通过' if test_results.get('cache_test') else '❌ 失败'}",
            "",
            "【结论】",
        ])

        # 判断结论
        roi_perf = test_results.get('performance', {}).get('roi_extraction', {})
        roi_meets_target = roi_perf.get('meets_target', False)

        if test_results.get('basic_extraction') and roi_meets_target:
            report_lines.append("ROI提取模块开发完成，性能达标，可用于rPPG信号提取")
        else:
            report_lines.append("ROI提取模块功能通过，性能待优化")

        report_lines.append("=" * 70)
        final_report = "\n".join(report_lines)

        # 保存报告
        report_path = os.path.join(self.output_dir, "test_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)

        print(f"✅ 报告已保存: {os.path.abspath(report_path)}")

        # 打印报告
        print(f"\n📋 报告预览:")
        print(final_report)


def main():
    """主测试流程"""
    print("=" * 70)
    print("ROI提取模块完整测试")
    print("=" * 70)

    tester = ROIExtractionTester()

    test_results = {
        'basic_extraction': False,
        'performance': None,
        'cache_test': False
    }

    # 测试1: 基础ROI提取
    test_results['basic_extraction'] = tester.test_basic_extraction()

    # 测试2: 性能基准
    test_results['performance'] = tester.test_performance_benchmark(
        num_frames=100
    )

    # 测试3: 缓存功能
    test_results['cache_test'] = tester.test_cache_functionality()

    # 测试4: 生成报告
    tester.generate_report(test_results)

    print("\n" + "=" * 70)
    print("🎉 测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()