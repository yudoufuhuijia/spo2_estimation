"""
统一视频读取接口 - VideoReader.py
功能：
1. 统一读取 OSS 视频流 / 本地测试视频
2. 自动识别视频源类型
3. 提供一致的API接口
4. 支持缓存和流式处理
【修改说明】：适配Windows/ECS跨平台缓存目录，解决WinError 5权限问题
"""

import os
import cv2
import tempfile
import platform  # 新增：用于判断操作系统
import numpy as np
from typing import Optional, Tuple, Generator, Union
from pathlib import Path
import time
import sys

# 适配data_process目录，添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是data_process目录
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.insert(0, project_root)

# 导入自定义模块
try:
    from utils.oss_file_reader import oss_read_file_stream
    from config.oss_config import bucket
    HAS_OSS = True
except ImportError:
    HAS_OSS = False
    print("⚠️  OSS模块未导入，仅支持本地视频")

# ================= 核心修复：替换相对导入为绝对导入 =================
from data_process.video_capture import VideoCapture  # 改为绝对导入，避免相对导入报错


# 以下代码保持不变...
class VideoReader:
    def __init__(
        self,
        source: Union[str, Path],
        source_type: str = 'auto',
        cache_enabled: bool = True,
        cache_dir: str = None  # 修改：默认值改为None，自动适配系统
    ):
        """
        初始化视频读取器

        Args:
            source: 视频源
                - OSS路径: 'oss://bucket/path/to/video.mp4'
                - 本地路径: '/path/to/video.mp4'
                - RTSP流: 'rtsp://...'
            source_type: 源类型
                - 'auto': 自动识别
                - 'oss': OSS存储
                - 'local': 本地文件
                - 'rtsp': RTSP流
            cache_enabled: 是否启用缓存（OSS视频）
            cache_dir: 缓存目录（None则自动适配系统）
        """
        self.source = str(source)
        self.source_type = source_type
        self.cache_enabled = cache_enabled

        # ================= 核心修改1：跨平台缓存目录适配 =================
        # 优先使用传入的cache_dir，否则根据系统自动选择
        if cache_dir is not None:
            self.cache_dir = cache_dir
        else:
            # Windows：使用项目内的utils/tmp（有写入权限）
            if platform.system() == "Windows":
                self.cache_dir = os.path.join(project_root, "utils", "tmp")
            # Linux/ECS：保留原有/tmp逻辑（兼容服务器）
            else:
                self.cache_dir = '/tmp/video_cache'

        self.video_capture = None
        self.cached_path = None
        self.is_oss_source = False

        # 自动识别源类型
        if self.source_type == 'auto':
            self.source_type = self._detect_source_type()

        # 初始化视频读取
        self._init_reader()

    def _detect_source_type(self) -> str:
        """自动检测视频源类型"""
        if self.source.startswith('oss://'):
            return 'oss'
        elif self.source.startswith('rtsp://'):
            return 'rtsp'
        elif self.source.startswith('http://') or self.source.startswith('https://'):
            return 'http'
        elif os.path.exists(self.source):
            return 'local'
        else:
            # 可能是OSS路径（不带oss://前缀）
            if HAS_OSS:
                try:
                    bucket.head_object(self.source)
                    return 'oss'
                except:
                    pass
            raise ValueError(f"无法识别视频源类型: {self.source}")

    def _init_reader(self):
        """初始化视频读取器"""
        print(f"📹 初始化视频源: {self.source}")
        print(f"   类型: {self.source_type}")
        # 新增：打印缓存目录，便于调试
        print(f"   缓存目录: {self.cache_dir}")

        if self.source_type == 'oss':
            self._init_oss_reader()
        elif self.source_type == 'local':
            self._init_local_reader()
        elif self.source_type in ['rtsp', 'http']:
            self._init_stream_reader()
        else:
            raise ValueError(f"不支持的源类型: {self.source_type}")

    def _init_oss_reader(self):
        """初始化OSS视频读取"""
        if not HAS_OSS:
            raise RuntimeError("OSS模块未安装，无法读取OSS视频")

        self.is_oss_source = True

        # 处理 oss:// 前缀
        oss_path = self.source.replace('oss://', '').lstrip('/')

        if self.cache_enabled:
            # 下载到本地缓存（已适配跨平台目录）
            print(f"   正在从OSS下载视频...")
            self.cached_path = self._download_from_oss(oss_path)
            self.video_capture = VideoCapture(self.cached_path)
        else:
            # 流式处理（不推荐，效率低）
            print(f"   使用OSS流式读取（可能较慢）...")
            self.cached_path = self._create_temp_video(oss_path)
            self.video_capture = VideoCapture(self.cached_path)

    def _download_from_oss(self, oss_path: str) -> str:
        """从OSS下载视频到本地（修复Windows权限）"""
        # 创建缓存目录（自动创建，确保有写入权限）
        os.makedirs(self.cache_dir, exist_ok=True)

        # 生成缓存文件名（跨平台路径拼接）
        filename = os.path.basename(oss_path)
        cache_path = os.path.join(self.cache_dir, filename)

        # 检查缓存是否存在
        if os.path.exists(cache_path):
            print(f"   ✅ 使用缓存文件: {cache_path}")
            return cache_path

        # 下载文件（原有逻辑不变，路径已适配）
        start_time = time.time()
        with open(cache_path, 'wb') as f:
            for chunk in oss_read_file_stream(oss_path):
                f.write(chunk)

        elapsed = time.time() - start_time
        file_size = os.path.getsize(cache_path) / (1024 * 1024)

        print(f"   ✅ 下载完成: {file_size:.2f} MB, 耗时 {elapsed:.2f}秒")

        return cache_path

    def _create_temp_video(self, oss_path: str) -> str:
        """创建临时视频文件（适配Windows临时目录）"""
        # 修改：Windows下使用系统临时目录（而非/tmp）
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(oss_path)[1],
            dir=self.cache_dir  # 临时文件放到自定义缓存目录
        )
        temp_path = temp_file.name
        temp_file.close()

        with open(temp_path, 'wb') as f:
            for chunk in oss_read_file_stream(oss_path):
                f.write(chunk)

        return temp_path

    def _init_local_reader(self):
        """初始化本地视频读取"""
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"视频文件不存在: {self.source}")

        self.video_capture = VideoCapture(self.source)

    def _init_stream_reader(self):
        """初始化流式读取"""
        self.video_capture = VideoCapture(self.source)

    # ===================== 统一接口 =====================

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        return self.video_capture.read()

    def read_generator(self) -> Generator[np.ndarray, None, None]:
        """生成器模式读取"""
        return self.video_capture.read_generator()

    def get_fps(self) -> float:
        """获取帧率"""
        return self.video_capture.get_fps()

    def get_resolution(self) -> Tuple[int, int]:
        """获取分辨率"""
        return self.video_capture.get_resolution()

    def get_total_frames(self) -> int:
        """获取总帧数"""
        return self.video_capture.get_total_frames()

    def set_position(self, frame_number: int) -> bool:
        """设置读取位置"""
        return self.video_capture.set_position(frame_number)

    def get_current_position(self) -> int:
        """获取当前位置"""
        return self.video_capture.get_current_position()

    def is_opened(self) -> bool:
        """检查是否打开"""
        return self.video_capture.is_opened

    def release(self):
        """释放资源"""
        if self.video_capture is not None:
            self.video_capture.release()

        # 清理临时文件（非缓存文件）
        if self.cached_path and not self.cache_enabled:
            if os.path.exists(self.cached_path):
                try:
                    os.remove(self.cached_path)
                    print(f"✅ 临时文件已删除: {self.cached_path}")
                except Exception as e:
                    print(f"⚠️  删除临时文件失败: {e}")

    def clear_cache(self):
        """清理缓存文件"""
        if self.cached_path and os.path.exists(self.cached_path):
            try:
                os.remove(self.cached_path)
                print(f"✅ 缓存已清理: {self.cached_path}")
            except Exception as e:
                print(f"⚠️  清理缓存失败: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        self.release()


# ===================== 批量读取工具 =====================

class BatchVideoReader:
    """
    批量视频读取器
    用于处理多个视频文件
    """

    def __init__(self, video_sources: list, **kwargs):
        """
        Args:
            video_sources: 视频源列表
            **kwargs: VideoReader 的参数
        """
        self.video_sources = video_sources
        self.reader_kwargs = kwargs
        self.current_reader = None
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> VideoReader:
        if self.current_index >= len(self.video_sources):
            raise StopIteration

        source = self.video_sources[self.current_index]
        self.current_index += 1

        return VideoReader(source, **self.reader_kwargs)


# ===================== 辅助函数 =====================

def clear_video_cache(cache_dir: str = None):
    """清理视频缓存目录（适配跨平台）"""
    # 修改：自动适配缓存目录
    if cache_dir is None:
        if platform.system() == "Windows":
            cache_dir = os.path.join(project_root, "utils", "tmp")
        else:
            cache_dir = '/tmp/video_cache'

    if not os.path.exists(cache_dir):
        print(f"⚠️  缓存目录不存在: {cache_dir}")
        return

    import shutil
    try:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✅ 缓存目录已清理: {cache_dir}")
    except Exception as e:
        print(f"⚠️  清理缓存失败: {e}")


# ===================== 测试代码 =====================

if __name__ == "__main__":
    print("=" * 60)
    print("统一视频读取器测试")
    print("=" * 60)

    # 测试1: 本地视频（适配项目根目录）
    print("\n【测试1】本地视频读取")
    local_video = os.path.join(project_root, "test_videos", "test_video_1.mp4")

    if os.path.exists(local_video):
        with VideoReader(local_video) as reader:
            print(f"帧率: {reader.get_fps():.2f} FPS")
            print(f"分辨率: {reader.get_resolution()}")
            print(f"总帧数: {reader.get_total_frames()}")

            # 读取前10帧
            for i, frame in enumerate(reader.read_generator()):
                if i >= 10:
                    break
                print(f"  读取第 {i+1} 帧, shape: {frame.shape}")

            print("✅ 本地视频读取测试通过")
    else:
        print(f"⚠️  测试视频不存在: {local_video}")
        print(f"项目根目录: {project_root}")
        print(f"请检查 test_videos 目录是否存在视频文件")

    # 测试2: OSS视频（如果可用）
    if HAS_OSS:
        print("\n【测试2】OSS视频读取")
        oss_video = "datasets/vipl/train/1/video1.mp4.avi"

        try:
            # 传入自定义缓存目录（可选，代码已自动适配）
            with VideoReader(oss_video, source_type='oss') as reader:
                print(f"帧率: {reader.get_fps():.2f} FPS")
                print(f"分辨率: {reader.get_resolution()}")

                # 读取前5帧
                for i, frame in enumerate(reader.read_generator()):
                    if i >= 5:
                        break
                    print(f"  读取第 {i+1} 帧")

                print("✅ OSS视频读取测试通过")
        except Exception as e:
            print(f"❌ OSS视频读取失败: {e}")
            import traceback
            traceback.print_exc()

    # 测试3: 批量读取
    print("\n【测试3】批量视频读取")
    test_videos_dir = os.path.join(project_root, "test_videos")
    test_videos = []

    if os.path.exists(test_videos_dir):
        test_videos = [
            os.path.join(test_videos_dir, f)
            for f in os.listdir(test_videos_dir)
            if f.endswith(('.mp4', '.avi'))
        ][:3]  # 最多测试3个

    if test_videos:
        batch_reader = BatchVideoReader(test_videos)

        for i, reader in enumerate(batch_reader, 1):
            print(f"\n视频 {i}: {reader.source}")
            print(f"  帧率: {reader.get_fps():.2f}")
            print(f"  分辨率: {reader.get_resolution()}")
            reader.release()

        print("\n✅ 批量读取测试通过")
    else:
        print("⚠️  未找到测试视频")

    print("\n" + "=" * 60)