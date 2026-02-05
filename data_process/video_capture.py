"""
视频采集适配器 - video_capture.py
功能：
1. 支持本地视频文件读取
2. 预留实时流接口（RTSP/RTMP/HTTP）
3. 无需图形界面，适配云环境
4. 统一的帧读取接口
"""

import cv2
import os
import sys
import time
from typing import Optional, Tuple, Generator
import numpy as np
from datetime import datetime


class VideoCapture:
    """
    视频采集适配器
    支持多种视频源：本地文件、RTSP流、HTTP流等
    """

    def __init__(self, source: str, backend: str = 'auto'):
        self.source = source
        self.backend = backend
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0

        self._init_capture()

    def _init_capture(self):
        try:
            if self.backend == 'ffmpeg':
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            elif self.backend == 'gstreamer':
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频源: {self.source}")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.is_opened = True

            print(f"✅ 视频源已打开: {self.source}")
            print(f"   分辨率: {self.width}x{self.height}")
            print(f"   帧率: {self.fps:.2f} FPS")
            if self.total_frames > 0:
                print(f"   总帧数: {self.total_frames}")

        except Exception as e:
            raise RuntimeError(f"视频源初始化失败: {str(e)}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened:
            return False, None

        ret, frame = self.cap.read()

        if ret:
            self.frame_count += 1

        return ret, frame

    def read_generator(self) -> Generator[np.ndarray, None, None]:
        while self.is_opened:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def set_position(self, frame_number: int) -> bool:
        if not self.is_opened:
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_current_position(self) -> int:
        if not self.is_opened:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_fps(self) -> float:
        return self.fps

    def get_resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_frame_count(self) -> int:
        return self.frame_count

    def get_total_frames(self) -> int:
        return self.total_frames

    def is_stream(self) -> bool:
        return self.total_frames <= 0

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print(f"✅ 视频源已释放: {self.source}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


class RTSPCapture(VideoCapture):
    def __init__(self, rtsp_url: str, reconnect_interval: int = 5):
        self.reconnect_interval = reconnect_interval
        super().__init__(rtsp_url, backend='ffmpeg')

    def read_with_reconnect(self) -> Tuple[bool, Optional[np.ndarray]]:
        max_retry = 3
        retry_count = 0

        while retry_count < max_retry:
            ret, frame = self.read()
            if ret:
                return True, frame

            print(f"⚠️  RTSP流断开，{self.reconnect_interval}秒后重连...")
            time.sleep(self.reconnect_interval)

            try:
                self.release()
                self._init_capture()
                retry_count += 1
            except Exception as e:
                print(f"❌ 重连失败: {e}")
                retry_count += 1

        return False, None


def test_video_source(source: str) -> dict:
    result = {
        'success': False,
        'source': source,
        'fps': 0,
        'resolution': (0, 0),
        'total_frames': 0,
        'error': None
    }

    try:
        with VideoCapture(source) as cap:
            for i in range(5):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"无法读取第{i + 1}帧")

            result['success'] = True
            result['fps'] = cap.get_fps()
            result['resolution'] = cap.get_resolution()
            result['total_frames'] = cap.get_total_frames()

    except Exception as e:
        result['error'] = str(e)

    return result


def get_video_info(video_path: str) -> dict:
    info = {
        'path': video_path,
        'exists': os.path.exists(video_path),
        'size_mb': 0,
        'duration_sec': 0,
        'fps': 0,
        'resolution': (0, 0),
        'total_frames': 0,
        'codec': 'unknown'
    }

    if not info['exists']:
        return info

    try:
        info['size_mb'] = os.path.getsize(video_path) / (1024 * 1024)

        with VideoCapture(video_path) as cap:
            info['fps'] = cap.get_fps()
            info['resolution'] = cap.get_resolution()
            info['total_frames'] = cap.get_total_frames()

            if info['fps'] > 0:
                info['duration_sec'] = info['total_frames'] / info['fps']

        cap_temp = cv2.VideoCapture(video_path)
        fourcc = int(cap_temp.get(cv2.CAP_PROP_FOURCC))
        info['codec'] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap_temp.release()

    except Exception as e:
        print(f"⚠️  获取视频信息失败: {e}")

    return info


if __name__ == "__main__":
    print("=" * 60)
    print("视频采集模块测试")
    print("=" * 60)

    # 适配data_process目录，自动查找项目根目录
    import pathlib
    def find_project_root(marker_dirs=("test_videos", "utils")):
        cur = pathlib.Path(__file__).resolve().parent  # 当前是data_process目录
        for _ in range(6):  # 最多回溯6层
            if any((cur / d).exists() for d in marker_dirs):
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return pathlib.Path(__file__).resolve().parent

    project_root = find_project_root()
    test_videos_dir = str(project_root / "test_videos")
    supported_exts = ('.mp4', '.avi', '.mkv')

    available_videos = []

    if os.path.exists(test_videos_dir):
        for f in os.listdir(test_videos_dir):
            if f.lower().endswith(supported_exts):
                full_path = os.path.join(test_videos_dir, f)
                if os.path.isfile(full_path):
                    available_videos.append(f)

    if available_videos:
        print(f"\n发现的视频文件 (在 {test_videos_dir}):")
        for v in available_videos:
            print(f"  - {v}")

        # 优先选择 .mp4，如果没有就选择列表第一个
        mp4s = [f for f in available_videos if f.lower().endswith('.mp4')]
        if mp4s:
            test_video = os.path.join(test_videos_dir, mp4s[0])
        else:
            test_video = os.path.join(test_videos_dir, available_videos[0])

        if test_video and os.path.exists(test_video):
            print(f"\n📹 测试视频: {test_video}")

            info = get_video_info(test_video)
            print(f"\n视频信息:")
            print(f"  文件大小: {info['size_mb']:.2f} MB")
            print(f"  时长: {info['duration_sec']:.2f} 秒")
            print(f"  帧率: {info['fps']:.2f} FPS")
            print(f"  分辨率: {info['resolution'][0]}x{info['resolution'][1]}")
            print(f"  总帧数: {info['total_frames']}")
            print(f"  编码: {info['codec']}")

            print(f"\n开始读取测试...")
            with VideoCapture(test_video) as cap:
                frame_count = 0
                start_time = time.time()

                for frame in cap.read_generator():
                    frame_count += 1
                    if frame_count >= 100:
                        break

                elapsed = time.time() - start_time
                print(f"✅ 读取完成")
                print(f"  读取帧数: {frame_count}")
                print(f"  耗时: {elapsed:.2f} 秒")
                print(f"  平均帧率: {frame_count / elapsed:.2f} FPS")

    else:
        print(f"\n⚠️  未发现可用的测试视频文件！")
        print(f"请将测试视频放置在 test_videos 目录下")
        # 打印真实工作目录和推断的项目根目录方便定位
        print(f"\n当前工作目录: {os.getcwd()}")
        print(f"项目根目录推断为: {project_root}")
        print(f"尝试查找的 test_videos 路径: {test_videos_dir}")

    print("\n" + "=" * 60)