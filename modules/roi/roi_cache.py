"""
ROI缓存管理模块 - roi_cache.py
功能：管理OSS临时缓存，处理后立即删除
"""

import os
import shutil
import time
from typing import Optional, Dict
from pathlib import Path


class ROICache:
    """
    ROI缓存管理器

    功能：
    1. 管理/tmp/oss_cache/roi临时目录
    2. 自动清理过期缓存
    3. 记录缓存统计信息
    """

    def __init__(
            self,
            cache_dir: str = "/tmp/oss_cache/roi",
            max_cache_size_mb: int = 500,  # 最大缓存500MB
            auto_cleanup: bool = True  # 是否自动清理
    ):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
            max_cache_size_mb: 最大缓存大小（MB）
            auto_cleanup: 是否启用自动清理
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # 转换为字节
        self.auto_cleanup = auto_cleanup

        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 缓存统计
        self.hits = 0
        self.misses = 0
        self.total_saved = 0
        self.total_deleted = 0

    def save_roi(
            self,
            roi_id: str,
            roi_data: Dict,
            metadata: Optional[Dict] = None
    ) -> str:
        """
        保存ROI到缓存

        Args:
            roi_id: ROI唯一标识（如：video_id_frame_123）
            roi_data: ROI数据字典
            metadata: 元数据（可选）

        Returns:
            缓存文件路径
        """
        import pickle

        # 生成缓存文件路径
        cache_file = self.cache_dir / f"{roi_id}.pkl"

        # 保存数据
        data_to_save = {
            'roi_data': roi_data,
            'metadata': metadata or {},
            'timestamp': time.time()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data_to_save, f)

        self.total_saved += 1

        # 自动清理检查
        if self.auto_cleanup:
            self._check_and_cleanup()

        return str(cache_file)

    def load_roi(self, roi_id: str) -> Optional[Dict]:
        """
        从缓存加载ROI

        Args:
            roi_id: ROI唯一标识

        Returns:
            ROI数据字典，或None（缓存未命中）
        """
        import pickle

        cache_file = self.cache_dir / f"{roi_id}.pkl"

        if not cache_file.exists():
            self.misses += 1
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            self.hits += 1
            return data['roi_data']

        except Exception as e:
            print(f"⚠️  缓存加载失败: {e}")
            self.misses += 1
            return None

    def delete_roi(self, roi_id: str) -> bool:
        """
        删除指定ROI缓存

        Args:
            roi_id: ROI唯一标识

        Returns:
            是否删除成功
        """
        cache_file = self.cache_dir / f"{roi_id}.pkl"

        if cache_file.exists():
            try:
                cache_file.unlink()
                self.total_deleted += 1
                return True
            except Exception as e:
                print(f"⚠️  删除缓存失败: {e}")
                return False

        return False

    def clear_all(self) -> int:
        """
        清空所有缓存

        Returns:
            删除的文件数
        """
        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"⚠️  删除文件失败 {cache_file}: {e}")

        self.total_deleted += deleted_count
        return deleted_count

    def _check_and_cleanup(self):
        """
        检查并清理过期缓存

        策略：
        1. 超过最大缓存大小时，删除最旧的文件
        2. 删除1小时前的缓存
        """
        current_time = time.time()
        one_hour_ago = current_time - 3600

        # 获取所有缓存文件
        cache_files = list(self.cache_dir.glob("*.pkl"))

        # 计算总大小
        total_size = sum(f.stat().st_size for f in cache_files)

        # 超过大小限制，删除最旧的文件
        if total_size > self.max_cache_size:
            # 按修改时间排序
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            # 删除直到低于限制
            for old_file in cache_files:
                if total_size <= self.max_cache_size * 0.8:  # 保留20%余量
                    break

                file_size = old_file.stat().st_size
                try:
                    old_file.unlink()
                    total_size -= file_size
                    self.total_deleted += 1
                except Exception as e:
                    print(f"⚠️  删除旧缓存失败: {e}")

        # 删除1小时前的缓存
        for cache_file in cache_files:
            if cache_file.stat().st_mtime < one_hour_ago:
                try:
                    cache_file.unlink()
                    self.total_deleted += 1
                except Exception as e:
                    print(f"⚠️  删除过期缓存失败: {e}")

    def get_cache_size(self) -> float:
        """
        获取当前缓存大小（MB）

        Returns:
            缓存大小（MB）
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        return total_size_bytes / (1024 * 1024)

    def get_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            统计字典
        """
        cache_size_mb = self.get_cache_size()
        cache_count = len(list(self.cache_dir.glob("*.pkl")))
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            'cache_dir': str(self.cache_dir),
            'cache_count': cache_count,
            'cache_size_mb': round(cache_size_mb, 2),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate * 100, 2),
            'total_saved': self.total_saved,
            'total_deleted': self.total_deleted
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()

        print(f"\n📊 ROI缓存统计:")
        print(f"   缓存目录: {stats['cache_dir']}")
        print(f"   缓存文件数: {stats['cache_count']}")
        print(f"   缓存大小: {stats['cache_size_mb']} MB")
        print(f"   缓存命中: {stats['hits']} 次")
        print(f"   缓存未命中: {stats['misses']} 次")
        print(f"   命中率: {stats['hit_rate']}%")
        print(f"   累计保存: {stats['total_saved']} 次")
        print(f"   累计删除: {stats['total_deleted']} 次")


# ===================== 测试代码 =====================
def test_roi_cache():
    """ROI缓存测试函数"""
    import numpy as np

    print("=" * 70)
    print("📝 ROI缓存管理测试")
    print("=" * 70)

    # 初始化缓存管理器
    print("\n【1/3】初始化缓存管理器")
    cache = ROICache(
        cache_dir="../../test_output/cache",  # 本地测试目录
        max_cache_size_mb=50,
        auto_cleanup=True
    )
    print(f"✅ 缓存管理器初始化完成")
    print(f"   缓存目录: {cache.cache_dir}")

    # 保存测试ROI
    print("\n【2/3】保存测试ROI")
    test_roi_data = {
        'forehead': np.random.randint(0, 255, (50, 80, 3), dtype=np.uint8),
        'left_cheek': np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
        'right_cheek': np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8),
        'coords': {
            'forehead': (100, 50, 80, 50),
            'left_cheek': (80, 120, 40, 40),
            'right_cheek': (180, 120, 40, 40)
        }
    }

    test_metadata = {
        'video_id': 'test_video_1',
        'frame_id': 123,
        'timestamp': time.time()
    }

    roi_id = "test_video_1_frame_123"
    cache_path = cache.save_roi(roi_id, test_roi_data, test_metadata)
    print(f"✅ ROI已保存到缓存")
    print(f"   ROI ID: {roi_id}")
    print(f"   缓存路径: {cache_path}")

    # 加载测试ROI
    print("\n【3/3】加载测试ROI")
    loaded_roi = cache.load_roi(roi_id)

    if loaded_roi:
        print(f"✅ ROI加载成功")
        print(f"   包含区域: {list(loaded_roi.keys())}")

        # 验证数据一致性
        if np.array_equal(loaded_roi['forehead'], test_roi_data['forehead']):
            print(f"   ✅ 数据一致性验证通过")
        else:
            print(f"   ❌ 数据不一致")
    else:
        print(f"❌ ROI加载失败")

    # 打印缓存统计
    cache.print_stats()

    # 清理测试缓存
    print(f"\n🗑️  清理测试缓存...")
    deleted_count = cache.delete_roi(roi_id)
    if deleted_count:
        print(f"✅ 已删除测试缓存")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_roi_cache()