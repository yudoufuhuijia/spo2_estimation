"""视频读取性能基准测试"""
import time
from data_process.VideoReader import VideoReader

def benchmark_local_video():
    """本地视频读取基准"""
    video_path = "../test_videos/test_video_1.avi"

    print("【本地视频读取基准】")

    # 测试1: 顺序读取
    with VideoReader(video_path) as reader:
        start = time.time()
        count = 0
        for frame in reader.read_generator():
            count += 1
            if count >= 300:
                break
        elapsed = time.time() - start

        print(f"顺序读取300帧: {elapsed:.2f}秒 ({count/elapsed:.2f} FPS)")

    # 测试2: 随机访问
    with VideoReader(video_path) as reader:
        import random
        positions = [random.randint(0, 300) for _ in range(50)]

        start = time.time()
        for pos in positions:
            reader.set_position(pos)
            reader.read()
        elapsed = time.time() - start

        print(f"随机访问50次: {elapsed:.2f}秒 ({50/elapsed:.2f} 次/秒)")

def benchmark_oss_video():
    """OSS视频读取基准"""
    oss_path = "datasets/vipl/train/1/video1.mp4.avi"

    print("\n【OSS视频读取基准】")

    # 测试1: 缓存模式
    start = time.time()
    with VideoReader(oss_path, cache_enabled=True) as reader:
        download_time = time.time() - start

        read_start = time.time()
        count = 0
        for frame in reader.read_generator():
            count += 1
            if count >= 100:
                break
        read_time = time.time() - read_start

        print(f"下载耗时: {download_time:.2f}秒")
        print(f"读取100帧: {read_time:.2f}秒 ({count/read_time:.2f} FPS)")

if __name__ == "__main__":
    benchmark_local_video()

    try:
        benchmark_oss_video()
    except:
        print("\n⚠️  OSS基准测试跳过")