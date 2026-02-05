"""
视频读取模块完整测试脚本 - test_video_module.py
用于在ECS上验证视频读取功能
"""

import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime

# 确保能导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_process.video_capture import VideoCapture, get_video_info
from data_process.VideoReader import VideoReader, clear_video_cache


def test_local_video_reading():
    """测试本地视频读取性能"""
    print("\n" + "=" * 70)
    print("【测试1】本地视频读取性能测试")
    print("=" * 70)

    test_dir = "../test_videos"
    if not os.path.exists(test_dir):
        print(f"❌ 测试目录不存在: {test_dir}")
        return False

    # 查找测试视频
    videos = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith(('.mp4', '.avi', '.mkv'))
    ]

    if not videos:
        print(f"❌ 未找到测试视频")
        return False

    test_video = videos[0]
    print(f"\n测试视频: {test_video}")

    # 获取视频信息
    info = get_video_info(test_video)
    print(f"\n视频信息:")
    print(f"  文件大小: {info['size_mb']:.2f} MB")
    print(f"  时长: {info['duration_sec']:.2f} 秒")
    print(f"  帧率: {info['fps']:.2f} FPS")
    print(f"  分辨率: {info['resolution'][0]}x{info['resolution'][1]}")
    print(f"  总帧数: {info['total_frames']}")

    # 性能测试
    print(f"\n开始读取性能测试...")
    with VideoCapture(test_video) as cap:
        frame_count = 0
        start_time = time.time()

        for frame in cap.read_generator():
            frame_count += 1

            # 读取300帧或全部
            if frame_count >= min(300, info['total_frames']):
                break

        elapsed = time.time() - start_time

        print(f"\n✅ 读取完成")
        print(f"  读取帧数: {frame_count}")
        print(f"  耗时: {elapsed:.2f} 秒")
        print(f"  实际帧率: {frame_count/elapsed:.2f} FPS")
        print(f"  理论帧率: {info['fps']:.2f} FPS")
        print(f"  处理效率: {(frame_count/elapsed)/info['fps']*100:.1f}%")

    return True


def test_oss_video_reading():
    """测试OSS视频读取"""
    print("\n" + "=" * 70)
    print("【测试2】OSS视频读取测试")
    print("=" * 70)

    try:
        from config.oss_config import bucket
        from utils.oss_file_reader import oss_read_file_stream
    except ImportError:
        print("❌ OSS模块未配置，跳过测试")
        return False

    # 使用之前清洗后的有效VIPL视频
    test_oss_path = "datasets/vipl/train/1/video1.mp4.avi"

    print(f"\n测试OSS视频: {test_oss_path}")

    try:
        # 测试缓存模式
        print(f"\n【模式1】缓存模式（推荐）")
        start_time = time.time()

        with VideoReader(test_oss_path, source_type='oss', cache_enabled=True) as reader:
            download_time = time.time() - start_time

            print(f"  下载耗时: {download_time:.2f} 秒")
            print(f"  帧率: {reader.get_fps():.2f} FPS")
            print(f"  分辨率: {reader.get_resolution()}")

            # 读取前100帧
            frame_count = 0
            read_start = time.time()

            for frame in reader.read_generator():
                frame_count += 1
                if frame_count >= 100:
                    break

            read_time = time.time() - read_start

            print(f"  读取100帧耗时: {read_time:.2f} 秒")
            print(f"  读取帧率: {frame_count/read_time:.2f} FPS")

        print(f"\n✅ OSS视频读取测试通过")
        return True

    except Exception as e:
        print(f"❌ OSS视频读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_interface():
    """测试统一接口"""
    print("\n" + "=" * 70)
    print("【测试3】统一接口测试")
    print("=" * 70)

    # 准备测试源
    test_sources = []

    # 本地视频
    if os.path.exists("../test_videos"):
        local_videos = [
            os.path.join("../test_videos", f)
            for f in os.listdir("../test_videos")
            if f.endswith(('.mp4', '.avi'))
        ][:2]
        test_sources.extend(local_videos)

    # OSS视频
    try:
        from config.oss_config import bucket
        test_sources.append("datasets/vipl/train/1/video1.mp4.avi")
    except:
        pass

    if not test_sources:
        print("❌ 未找到测试源")
        return False

    print(f"\n发现 {len(test_sources)} 个测试源")

    for i, source in enumerate(test_sources, 1):
        print(f"\n--- 测试源 {i}/{len(test_sources)} ---")
        print(f"路径: {source}")

        try:
            with VideoReader(source) as reader:
                print(f"  类型: {reader.source_type}")
                print(f"  帧率: {reader.get_fps():.2f} FPS")
                print(f"  分辨率: {reader.get_resolution()}")

                # 读取5帧
                for j, frame in enumerate(reader.read_generator()):
                    if j >= 5:
                        break
                    print(f"  帧{j+1}: {frame.shape}")

                print(f"  ✅ 读取成功")

        except Exception as e:
            print(f"  ❌ 读取失败: {e}")

    return True


def test_frame_extraction():
    """测试帧提取功能"""
    print("\n" + "=" * 70)
    print("【测试4】帧提取与保存测试")
    print("=" * 70)

    test_video = None

    # 查找测试视频
    if os.path.exists("../test_videos"):
        videos = [
            os.path.join("../test_videos", f)
            for f in os.listdir("../test_videos")
            if f.endswith(('.mp4', '.avi'))
        ]
        if videos:
            test_video = videos[0]

    if not test_video:
        print("❌ 未找到测试视频")
        return False

    print(f"\n测试视频: {test_video}")

    # 创建输出目录
    output_dir = "../test_output/frames"
    os.makedirs(output_dir, exist_ok=True)

    # 提取关键帧
    print(f"\n提取关键帧...")
    with VideoReader(test_video) as reader:
        total_frames = reader.get_total_frames()

        # 提取均匀分布的10帧
        extract_indices = np.linspace(0, total_frames-1, 10, dtype=int)

        extracted = 0
        for i, frame in enumerate(reader.read_generator()):
            if i in extract_indices:
                output_path = os.path.join(output_dir, f"frame_{extracted:03d}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"  保存帧 {i}: {output_path}")
                extracted += 1

            if extracted >= 10:
                break

    print(f"\n✅ 成功提取 {extracted} 帧")
    print(f"输出目录: {output_dir}")

    return True


def generate_test_report():
    """生成测试报告"""
    print("\n" + "=" * 70)
    print("生成测试报告")
    print("=" * 70)

    report_lines = [
        "=" * 70,
        "视频读取模块测试报告",
        "=" * 70,
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"测试环境: ECS (无图形界面)",
        "",
        "【测试结果】"
    ]

    # 运行所有测试
    tests = {
        "本地视频读取": test_local_video_reading,
        "OSS视频读取": test_oss_video_reading,
        "统一接口": test_unified_interface,
        "帧提取功能": test_frame_extraction
    }

    results = {}
    for name, test_func in tests.items():
        try:
            results[name] = test_func()
        except Exception as e:
            results[name] = False
            print(f"\n❌ {name} 测试异常: {e}")

    # 添加到报告
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        report_lines.append(f"{name:20s} {status}")

    report_lines.extend([
        "",
        "【环境信息】",
        f"Python版本: {sys.version.split()[0]}",
        f"OpenCV版本: {cv2.__version__}",
        f"当前目录: {os.getcwd()}",
        "",
        "【结论】",
        f"通过率: {sum(results.values())}/{len(results)} ({sum(results.values())/len(results)*100:.0f}%)",
        "=" * 70
    ])

    report_text = "\n".join(report_lines)

    # 保存报告
    os.makedirs("../test_output", exist_ok=True)
    report_path = "../test_output/video_module_test_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n📄 报告已保存: {report_path}")

    # 上传到OSS
    try:
        from config.oss_config import bucket
        oss_report_path = "test_results/video_module_test_report.txt"
        bucket.put_object_from_file(oss_report_path, report_path)
        print(f"✅ 报告已上传OSS: {oss_report_path}")
    except:
        print("⚠️  OSS上传跳过（模块未配置）")


def main():
    """主测试流程"""
    print("\n" + "=" * 70)
    print("视频读取模块完整测试")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 生成测试报告
    generate_test_report()

    # 清理缓存
    print("\n清理缓存...")
    clear_video_cache()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()