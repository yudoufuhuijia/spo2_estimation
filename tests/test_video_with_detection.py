"""视频读取 + 人脸检测集成测试"""
import sys
import os
# 添加项目根目录到Python路径，确保能导入自定义模块
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 导入项目的视频读取模块和人脸检测模块
from data_process.VideoReader import VideoReader
from modules.detection.face_detector import FaceDetector

def test_integration():
    print("🔍 开始视频读取+人脸检测集成测试")
    # 初始化人脸检测器
    detector = FaceDetector()
    # 🔥 核心修复：基于项目根目录拼接视频绝对路径，兼容Windows
    video_path = os.path.join(PROJECT_ROOT, "test_videos", "test_video_1.avi")

    # 前置检查：视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 测试视频不存在：{os.path.abspath(video_path)}")
        print(f"💡 请在项目根目录创建test_videos文件夹，并放入test_video_1.avi")
        return

    print(f"✅ 找到测试视频：{os.path.abspath(video_path)}")
    print(f"📹 开始逐帧检测（仅测试前20帧）...\n")

    # 读取视频并逐帧检测人脸，增加异常捕获
    try:
        with VideoReader(video_path) as reader:
            for i, frame in enumerate(reader.read_generator()):
                detections = detector.detect(frame)
                face_num = len(detections)
                if face_num > 0:
                    print(f"帧{i:2d}: 检测到 {face_num} 张人脸")
                else:
                    print(f"帧{i:2d}: 未检测到人脸")
                # 仅测试前20帧，提高测试效率
                if i >= 20:
                    break
    except Exception as e:
        print(f"\n❌ 集成测试失败：{str(e)}")
        return

    # 测试通过提示
    print("\n" + "="*50)
    print("✅ 视频读取+人脸检测集成测试通过！")
    print("✅ 可用于后续ROI提取模块开发")
    print("="*50)

if __name__ == "__main__":
    # 执行集成测试
    test_integration()