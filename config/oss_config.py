import oss2
import os
from dotenv import load_dotenv

# 加载当前目录的.env配置文件
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# 从.env读取配置
ACCESS_KEY_ID = os.getenv("ALIYUN_OSS_ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET")
ENDPOINT = os.getenv("ALIYUN_OSS_ENDPOINT")
BUCKET_NAME = os.getenv("ALIYUN_OSS_BUCKET_NAME")

# 调试打印（确认读取到的密钥，和PyCharm一致）
print("读取到的AccessKey ID:", ACCESS_KEY_ID)
print("读取到的Bucket名称:", BUCKET_NAME)

# 配置校验
if not all([ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME]):
    raise ValueError("配置缺失！请检查.env文件是否完整")

# 初始化OSS连接
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

if __name__ == "__main__":
    try:
        # 测试OSS连接
        bucket.list_objects(max_keys=1)
        print("✅ OSS连接成功！")
    except Exception as e:
        # 标准异常打印，无语法错误
        print("❌ 失败：", str(e))