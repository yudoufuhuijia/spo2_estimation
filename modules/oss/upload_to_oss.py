import oss2
import os

# ==================== OSS配置（从环境变量读取，禁止硬编码） ====================
# 环境变量配置说明：
# export OSS_ACCESS_KEY_ID="你的AccessKey ID"
# export OSS_ACCESS_KEY_SECRET="你的AccessKey Secret"
# export OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"
# export OSS_BUCKET_NAME="spo2-estimation"
ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
ENDPOINT = os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")
BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "spo2-estimation")

# ==================== 本地文件路径（不用改） ====================
LOCAL_FILE_DIR = "../../model_output_v2"
LOCAL_FILES = [
    "best_model.pth",
    "metrics.txt",
    "predictions.npz",
    "training_trend.png"
]

# ==================== OSS上传路径（不用改） ====================
OSS_TARGET_DIR = "spo2-training/"


# ==================== 核心逻辑 ====================
def init_oss_client():
    """初始化OSS客户端，校验密钥和Bucket可用性"""
    # 前置校验：环境变量是否配置
    if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET:
        print("❌ 错误：OSS密钥未配置！请设置环境变量：")
        print("   export OSS_ACCESS_KEY_ID='你的AccessKey ID'")
        print("   export OSS_ACCESS_KEY_SECRET='你的AccessKey Secret'")
        exit(1)

    try:
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
        bucket.get_bucket_info()  # 校验Bucket是否可访问
        print("✅ OSS客户端初始化成功")
        return bucket
    except oss2.exceptions.AccessDenied:
        print("❌ 错误：OSS密钥无上传权限！请给RAM用户添加【AliyunOSSFullAccess】权限")
        exit(1)
    except oss2.exceptions.NoSuchBucket:
        print(f"❌ 错误：OSS桶不存在！请确认桶名是 {BUCKET_NAME}")
        exit(1)
    except Exception as e:
        print(f"❌ OSS连接失败：{str(e)}")
        exit(1)


def upload_file(bucket, local_file_path, oss_file_path):
    """
    上传单个文件到OSS
    :param bucket: OSS Bucket实例
    :param local_file_path: 本地文件绝对路径
    :param oss_file_path: OSS目标路径
    """
    if not os.path.exists(local_file_path):
        print(f"⚠️ 跳过：本地文件不存在 → {local_file_path}")
        return
    try:
        bucket.put_object_from_file(oss_file_path, local_file_path)
        print(f"✅ 上传成功 → OSS路径：{oss_file_path}")
    except Exception as e:
        print(f"❌ 上传失败 → {local_file_path}：{str(e)}")


if __name__ == "__main__":
    # 初始化OSS客户端
    oss_bucket = init_oss_client()

    # 批量上传文件
    print(f"\n=== 开始上传 {LOCAL_FILE_DIR} 下的文件 ===")
    for file_name in LOCAL_FILES:
        local_full_path = os.path.abspath(os.path.join(LOCAL_FILE_DIR, file_name))  # 转为绝对路径，避免相对路径问题
        oss_full_path = os.path.join(OSS_TARGET_DIR, file_name)  # 兼容不同系统路径分隔符
        upload_file(oss_bucket, local_full_path, oss_full_path)

    print("\n=== 所有文件上传完成！===")