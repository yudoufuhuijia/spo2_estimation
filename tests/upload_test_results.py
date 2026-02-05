"""上传测试结果到OSS（修复文件路径+前置检查）"""
import os
import sys

# ================= 核心修复1：定位项目根目录 =================
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是tests目录
project_root = os.path.dirname(current_dir)  # 项目根目录（spo2_estimation）
sys.path.insert(0, project_root)

# 导入OSS配置（确保从根目录正确导入）
from config.oss_config import bucket


def upload_test_results():
    # ================= 核心修复2：用绝对路径定位文件 =================
    # 测试报告、帧文件的绝对路径（基于项目根目录）
    files_to_upload = {
        os.path.join(project_root, "test_output", "video_module_test_report.txt"):
        "test_results/2.7_video_module_test_report.txt",

        os.path.join(project_root, "test_output", "frames", "frame_000.jpg"):
        "test_results/2.7_sample_frame.jpg"
    }

    # ================= 核心修复3：前置检查（避免文件不存在） =================
    # 检查test_output目录是否存在（没运行test_video_module.py的话会不存在）
    test_output_dir = os.path.join(project_root, "test_output")
    if not os.path.exists(test_output_dir):
        print(f"⚠️  未找到test_output目录！请先运行：python tests/test_video_module.py 生成测试结果")
        return

    # 检查frames子目录是否存在
    frames_dir = os.path.join(test_output_dir, "frames")
    if not os.path.exists(frames_dir):
        print(f"⚠️  未找到帧文件目录！请先运行：python tests/test_video_module.py 完成帧提取测试")
        return

    # 批量上传文件
    print("开始上传测试结果到OSS...")
    success_count = 0
    for local_path, oss_path in files_to_upload.items():
        if os.path.exists(local_path):
            bucket.put_object_from_file(oss_path, local_path)
            print(f"✅ 上传成功: {os.path.basename(local_path)} → {oss_path}")
            success_count += 1
        else:
            print(f"❌ 文件不存在: {local_path}（请确保test_video_module.py已正常运行）")

    # 上传结果汇总
    print(f"\n📊 上传完成：成功{success_count}/{len(files_to_upload)}个文件")


if __name__ == "__main__":
    upload_test_results()