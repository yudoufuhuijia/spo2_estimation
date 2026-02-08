"""
特征分块写入器 - feature_block_writer.py
功能：将提取的特征分块存储到OSS
特点：
1. 分块管理（每块≤100MB）
2. 增量写入（避免内存溢出）
3. 索引管理（快速查找）
4. 元数据记录
"""

import os
import numpy as np
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime


class FeatureBlockWriter:
    """
    特征分块写入器

    核心功能：
    1. 分块存储：将特征分割为小块存储
    2. 索引管理：维护特征索引文件
    3. 元数据：记录特征来源、时间戳等
    4. 批量写入：提高IO效率
    """

    def __init__(
            self,
            output_dir: str = "test_output/features",
            block_size_mb: int = 100,  # 每块最大大小（MB）
            enable_compression: bool = True  # 是否启用压缩
    ):
        """
        初始化特征写入器

        Args:
            output_dir: 输出目录
            block_size_mb: 每块最大大小（MB）
            enable_compression: 是否启用numpy压缩
        """
        self.output_dir = Path(output_dir)
        self.block_size_mb = block_size_mb
        self.enable_compression = enable_compression

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 特征缓冲区
        self.feature_buffer = []
        self.current_block_id = 0

        # 索引
        self.feature_index = {
            'blocks': [],
            'total_samples': 0,
            'created_at': datetime.now().isoformat()
        }

        print(f"✅ 特征写入器初始化完成")
        print(f"   输出目录: {self.output_dir}")
        print(f"   块大小限制: {block_size_mb} MB")
        print(f"   压缩: {'启用' if enable_compression else '禁用'}")

    def add_features(
            self,
            features: Dict,
            video_id: str,
            metadata: Optional[Dict] = None
    ):
        """
        添加特征到缓冲区

        Args:
            features: 特征字典（feature_extractor输出）
            video_id: 视频标识
            metadata: 额外元数据
        """
        # 创建特征记录
        feature_record = {
            'video_id': video_id,
            'features': features,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.feature_buffer.append(feature_record)

        # 检查是否需要写入
        if self._should_flush():
            self.flush()

    def flush(self):
        """
        将缓冲区特征写入磁盘
        """
        if len(self.feature_buffer) == 0:
            return

        # 生成块文件名
        block_filename = f"features_block_{self.current_block_id:04d}.npz"
        block_path = self.output_dir / block_filename

        # 准备数据
        block_data = {
            'features': self.feature_buffer,
            'block_id': self.current_block_id,
            'n_samples': len(self.feature_buffer),
            'created_at': datetime.now().isoformat()
        }

        # 写入文件
        if self.enable_compression:
            np.savez_compressed(block_path, **block_data)
        else:
            np.savez(block_path, **block_data)

        # 更新索引
        block_info = {
            'block_id': self.current_block_id,
            'filename': block_filename,
            'n_samples': len(self.feature_buffer),
            'file_size_mb': block_path.stat().st_size / (1024 * 1024),
            'created_at': block_data['created_at']
        }

        self.feature_index['blocks'].append(block_info)
        self.feature_index['total_samples'] += len(self.feature_buffer)

        print(f"✅ 写入特征块 {self.current_block_id:04d}")
        print(f"   样本数: {len(self.feature_buffer)}")
        print(f"   文件大小: {block_info['file_size_mb']:.2f} MB")

        # 清空缓冲区，递增块ID
        self.feature_buffer = []
        self.current_block_id += 1

    def _should_flush(self) -> bool:
        """
        判断是否需要刷新缓冲区

        Returns:
            是否需要刷新
        """
        if len(self.feature_buffer) == 0:
            return False

        # 估算缓冲区大小（粗略）
        # 假设每个特征记录约10KB
        estimated_size_mb = len(self.feature_buffer) * 10 / 1024

        return estimated_size_mb >= self.block_size_mb

    def save_index(self, filename: str = "feature_index.json"):
        """
        保存特征索引

        Args:
            filename: 索引文件名
        """
        import json

        index_path = self.output_dir / filename

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_index, f, indent=2, ensure_ascii=False)

        print(f"✅ 特征索引已保存: {index_path}")

    def close(self):
        """
        关闭写入器（刷新缓冲区并保存索引）
        """
        # 写入剩余特征
        if len(self.feature_buffer) > 0:
            self.flush()

        # 保存索引
        self.save_index()

        print(f"\n📊 特征写入统计:")
        print(f"   总块数: {len(self.feature_index['blocks'])}")
        print(f"   总样本数: {self.feature_index['total_samples']}")

        total_size = sum(block['file_size_mb'] for block in self.feature_index['blocks'])
        print(f"   总大小: {total_size:.2f} MB")


class FeatureReader:
    """
    特征读取器（与FeatureBlockWriter配套）

    功能：
    1. 加载特征索引
    2. 按块读取特征
    3. 查询特征
    """

    def __init__(self, feature_dir: str):
        """
        初始化特征读取器

        Args:
            feature_dir: 特征目录
        """
        self.feature_dir = Path(feature_dir)

        # 加载索引
        self.index = self._load_index()

        print(f"✅ 特征读取器初始化完成")
        print(f"   特征目录: {self.feature_dir}")
        print(f"   总块数: {len(self.index['blocks'])}")
        print(f"   总样本数: {self.index['total_samples']}")

    def _load_index(self) -> Dict:
        """加载特征索引"""
        import json

        index_path = self.feature_dir / "feature_index.json"

        if not index_path.exists():
            raise FileNotFoundError(f"特征索引不存在: {index_path}")

        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)

        return index

    def load_block(self, block_id: int) -> List[Dict]:
        """
        加载指定块的特征

        Args:
            block_id: 块ID

        Returns:
            特征列表
        """
        # 查找块信息
        block_info = None
        for block in self.index['blocks']:
            if block['block_id'] == block_id:
                block_info = block
                break

        if block_info is None:
            raise ValueError(f"块ID不存在: {block_id}")

        # 加载块文件
        block_path = self.feature_dir / block_info['filename']
        data = np.load(block_path, allow_pickle=True)

        features = data['features'].tolist()

        return features

    def load_all_features(self) -> List[Dict]:
        """
        加载所有特征

        Returns:
            所有特征列表
        """
        all_features = []

        for block in self.index['blocks']:
            block_features = self.load_block(block['block_id'])
            all_features.extend(block_features)

        return all_features

    def get_feature_by_video_id(self, video_id: str) -> Optional[Dict]:
        """
        根据视频ID查询特征

        Args:
            video_id: 视频ID

        Returns:
            特征字典（如果找到）
        """
        for block in self.index['blocks']:
            block_features = self.load_block(block['block_id'])

            for feature_record in block_features:
                if feature_record['video_id'] == video_id:
                    return feature_record

        return None


# ===================== 测试代码 =====================
def test_feature_block_writer():
    """特征块写入器测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')

    print("=" * 70)
    print("📝 特征分块写入器测试")
    print("=" * 70)

    # 初始化写入器
    print("\n【1/3】初始化写入器")
    writer = FeatureBlockWriter(
        output_dir="../../test_output/features",
        block_size_mb=10,  # 测试用，设置较小
        enable_compression=True
    )

    # 模拟添加特征
    print("\n【2/3】添加测试特征")

    for i in range(5):
        # 模拟特征
        test_features = {
            'valid': True,
            'ror_features': {
                'ror': 1.234 + i * 0.1,
                'ac_component': 0.567,
                'dc_component': 0.456,
                'n_peaks': 10 + i
            },
            'hr_features': {
                'available': True,
                'mean_hr': 75.0 + i * 2,
                'std_hr': 5.2
            }
        }

        writer.add_features(
            features=test_features,
            video_id=f"test_video_{i:03d}",
            metadata={'test': True, 'index': i}
        )

        print(f"   添加特征 {i + 1}/5")

    # 关闭写入器
    print("\n【3/3】关闭写入器")
    writer.close()

    # 测试读取器
    print("\n【验证】测试读取器")
    reader = FeatureReader("../../test_output/features")

    # 读取所有特征
    all_features = reader.load_all_features()
    print(f"✅ 成功读取 {len(all_features)} 个特征")

    # 查询特定视频
    feature = reader.get_feature_by_video_id("test_video_002")
    if feature:
        print(f"✅ 成功查询视频 test_video_002")
        print(f"   心率: {feature['features']['hr_features']['mean_hr']:.1f} BPM")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_feature_block_writer()