"""
数据集划分脚本（最终修复版）
功能：
1. 读取清洗后的数据索引
2. 按7:3比例划分训练集和测试集
3. 支持分层采样（按标签比例划分）
4. 生成训练/测试索引文件并上传OSS
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from config.oss_config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME
import oss2
from datetime import datetime

# ===================== 配置 =====================
class SplitConfig:
    # 输入文件
    VALID_INDEX_PATH = "processed_data/valid_data_index.csv"

    # 输出文件
    TRAIN_INDEX_PATH = "processed_data/train_index.csv"
    TEST_INDEX_PATH = "processed_data/test_index.csv"
    LABEL_MAPPING_PATH = "processed_data/label_mapping.csv"
    SPLIT_REPORT_PATH = "processed_data/split_report.txt"

    # 划分参数
    TEST_SIZE = 0.3  # 测试集比例
    RANDOM_STATE = 42  # 随机种子（确保可复现）
    STRATIFY = True  # 是否分层采样

# ===================== OSS连接（全局复用，提速） =====================
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

# ===================== 工具函数 =====================
def download_from_oss(oss_path, local_path):
    """从OSS下载文件"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket.get_object_to_file(oss_path, local_path)
    print(f"✅ 下载成功: {oss_path} → {local_path}")

def upload_to_oss(local_path, oss_path):
    """上传文件到OSS"""
    bucket.put_object_from_file(oss_path, local_path)
    print(f"✅ 上传成功: {local_path} → {oss_path}")

def load_valid_index():
    """加载有效数据索引"""
    local_path = "tmp/valid_data_index.csv"
    download_from_oss(SplitConfig.VALID_INDEX_PATH, local_path)
    df = pd.read_csv(local_path, encoding='utf-8')
    print(f"📊 加载数据索引: {len(df)} 条记录")
    return df

def create_label_mapping(df):
    """创建标签映射表"""
    unique_labels = sorted(df['label'].unique())
    label_mapping = pd.DataFrame({
        'label_id': range(len(unique_labels)),
        'label_name': unique_labels,
        'count': [len(df[df['label'] == label]) for label in unique_labels]
    })
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label_mapping, label_to_idx

def split_dataset(df):
    """划分数据集"""
    print(f"\n🔀 开始划分数据集 (训练:测试 = {1-SplitConfig.TEST_SIZE}:{SplitConfig.TEST_SIZE})")
    label_mapping, label_to_idx = create_label_mapping(df)
    df['label_id'] = df['label'].map(label_to_idx)
    stratify_column = df['label_id'] if SplitConfig.STRATIFY else None

    train_df, test_df = train_test_split(
        df,
        test_size=SplitConfig.TEST_SIZE,
        random_state=SplitConfig.RANDOM_STATE,
        stratify=stratify_column
    )

    print(f"✅ 划分完成: 训练集={len(train_df)} | 测试集={len(test_df)}")
    return train_df, test_df, label_mapping

def save_splits(train_df, test_df, label_mapping):
    """保存划分结果"""
    os.makedirs("tmp", exist_ok=True)
    # 保存训练集
    train_local = "tmp/train_index.csv"
    train_df.to_csv(train_local, index=False, encoding='utf-8')
    upload_to_oss(train_local, SplitConfig.TRAIN_INDEX_PATH)
    # 保存测试集
    test_local = "tmp/test_index.csv"
    test_df.to_csv(test_local, index=False, encoding='utf-8')
    upload_to_oss(test_local, SplitConfig.TEST_INDEX_PATH)
    # 保存标签映射
    label_local = "tmp/label_mapping.csv"
    label_mapping.to_csv(label_local, index=False, encoding='utf-8')
    upload_to_oss(label_local, SplitConfig.LABEL_MAPPING_PATH)

def generate_split_report(train_df, test_df, label_mapping):
    """生成划分报告"""
    report_lines = [
        "=" * 60,
        "数据集划分报告",
        "=" * 60,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "【划分参数】",
        f"测试集比例: {SplitConfig.TEST_SIZE}",
        f"随机种子: {SplitConfig.RANDOM_STATE}",
        f"分层采样: {'是' if SplitConfig.STRATIFY else '否'}",
        "",
        "【数据统计】",
        f"总数据量: {len(train_df) + len(test_df)}",
        f"训练集: {len(train_df)} ({len(train_df)/(len(train_df)+len(test_df))*100:.2f}%)",
        f"测试集: {len(test_df)} ({len(test_df)/(len(train_df)+len(test_df))*100:.2f}%)",
        f"标签类别数: {len(label_mapping)}",
        "",
        "【标签分布】",
    ]
    for _, row in label_mapping.iterrows():
        label = row['label_name']
        train_count = len(train_df[train_df['label'] == label])
        test_count = len(test_df[test_df['label'] == label])
        report_lines.append(f"  {label}: 总={row['count']}, 训练={train_count}, 测试={test_count}")
    report_lines.extend(["", "【数据类型分布】"])
    for data_type in train_df['data_type'].unique():
        train_count = len(train_df[train_df['data_type'] == data_type])
        test_count = len(test_df[test_df['data_type'] == data_type])
        report_lines.append(f"  {data_type}: 训练={train_count}, 测试={test_count}")
    report_lines.extend(["", "【输出文件】",
        f"训练集索引: {SplitConfig.TRAIN_INDEX_PATH}",
        f"测试集索引: {SplitConfig.TEST_INDEX_PATH}",
        f"标签映射: {SplitConfig.LABEL_MAPPING_PATH}",
        "=" * 60
    ])
    report_text = "\n".join(report_lines)
    # 保存报告
    report_local = "tmp/split_report.txt"
    with open(report_local, 'w', encoding='utf-8') as f:
        f.write(report_text)
    upload_to_oss(report_local, SplitConfig.SPLIT_REPORT_PATH)
    print(f"\n✅ 划分报告已生成:\n{report_text}")

def main():
    """主流程"""
    print("=" * 60)
    print("数据集划分脚本启动")
    print("=" * 60)
    try:
        df = load_valid_index()
        train_df, test_df, label_mapping = split_dataset(df)
        save_splits(train_df, test_df, label_mapping)
        generate_split_report(train_df, test_df, label_mapping)
        print("\n" + "=" * 60)
        print("🎉 数据集划分完成！")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)