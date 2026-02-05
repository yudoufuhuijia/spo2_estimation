"""从OSS下载测试视频到项目根目录的test_videos"""
import os
import sys

# 定位项目根目录（当前是tests目录，上级为项目根）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入OSS工具
from utils.oss_file_reader import oss_read_file_stream
from config.oss_config import bucket

def download_test_videos():
    # 要下载的VIPL测试视频（OSS路径）
    test_videos = [
        "datasets/vipl/train/1/video1.mp4.avi",
        "datasets/vipl/train/2/video1.mp4.avi",
        "datasets/vipl/train/3/video1.mp4.avi"
    ]

    # ========== 核心修复：用项目根目录拼接绝对路径 ==========
    # 项目根目录下的test_videos（统一存储测试视频）
    test_videos_dir = os.path.join(project_root, "test_videos")
    os.makedirs(test_videos_dir, exist_ok=True)  # 在根目录创建test_videos
    print(f"📂 测试视频将保存到：{test_videos_dir}")

    # 批量下载
    for i, oss_path in enumerate(test_videos, 1):
        # 拼接绝对本地路径（根目录/test_videos/xxx.avi）
        local_path = os.path.join(test_videos_dir, f"test_video_{i}.avi")
        print(f"\n【下载视频 {i}/{len(test_videos)}】OSS路径：{oss_path}")

        try:
            # 从OSS流式下载并保存
            with open(local_path, "wb") as f:
                for chunk in oss_read_file_stream(oss_path):
                    f.write(chunk)
            print(f"✅ 保存成功：{local_path}")
        except Exception as e:
            print(f"❌ 下载失败：{str(e)}")

    print(f"\n🎉 测试视频下载完成（共{len(test_videos)}个）")

if __name__ == "__main__":
    download_test_videos()