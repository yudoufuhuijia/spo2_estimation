import oss2
import os

# ==================== 修正AccessKey ID（和控制台一致） ====================
ACCESS_KEY_ID = "LTAI5tBXijvuE7F8oV95LCPP"  # 替换成控制台里的正确ID
ACCESS_KEY_SECRET = "Wk7LCwD2BI1GXRFWWLANmOmdKpqqXi"  # 必须是这个ID对应的Secret（创建时显示的那个）
ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"
BUCKET_NAME = "spo2-estimation"


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


# ==================== 核心逻辑（不用改） ====================
def init_oss_client():
    try:
        auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
        bucket.get_bucket_info()
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
    if not os.path.exists(local_file_path):
        print(f"⚠️ 跳过：本地文件不存在 → {local_file_path}")
        return
    try:
        bucket.put_object_from_file(oss_file_path, local_file_path)
        print(f"✅ 上传成功 → OSS路径：{oss_file_path}")
    except Exception as e:
        print(f"❌ 上传失败 → {local_file_path}：{str(e)}")


if __name__ == "__main__":
    oss_bucket = init_oss_client()
    print(f"\n=== 开始上传model_output_v2下的文件 ===")
    for file_name in LOCAL_FILES:
        local_full_path = os.path.join(LOCAL_FILE_DIR, file_name)
        oss_full_path = OSS_TARGET_DIR + file_name
        upload_file(oss_bucket, local_full_path, oss_full_path)
    print("\n=== 所有文件上传完成！===")