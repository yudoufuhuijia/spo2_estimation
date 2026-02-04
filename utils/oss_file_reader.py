import oss2
import os
import datetime
from config.oss_config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME

# 初始化OSS连接
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

def oss_download_file(oss_file_path, local_temp_path):
    """
    从OSS下载文件到本地临时目录
    :param oss_file_path: OSS文件路径
    :param local_temp_path: 本地临时路径
    :return: 本地临时文件路径
    """
    os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)
    bucket.get_object_to_file(oss_file_path, local_temp_path)
    print(f"✅ OSS文件下载成功：{oss_file_path} → {local_temp_path}")
    return local_temp_path

def oss_read_file_stream(oss_file_path, chunk_size=1024*1024):
    """
    流式读取OSS文件，低内存占用
    :param oss_file_path: OSS文件路径
    :param chunk_size: 分块大小
    :return: 文件流生成器
    """
    object_stream = bucket.get_object(oss_file_path)
    for chunk in object_stream:
        yield chunk
    object_stream.close()

def oss_write_log(log_content, log_file_name="data_read_log.txt"):
    """
    修复OSS日志写入：删除旧文件后重新上传，彻底解决ObjectNotAppendable错误
    兼容所有OSS存储类型，保留历史日志
    """
    log_oss_path = f"logs/{log_file_name}"
    existing_log = ""
    # 读取已有日志内容
    if bucket.object_exists(log_oss_path):
        try:
            existing_log = bucket.get_object(log_oss_path).read().decode('utf-8')
            # 删除原文件，规避追加写入限制
            bucket.delete_object(log_oss_path)
        except Exception:
            existing_log = ""
    # 拼接时间戳与新日志
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_log = existing_log + f"\n[{time_str}] {log_content}"
    # 全新上传日志文件
    bucket.put_object(log_oss_path, new_log.encode('utf-8'))
    print(f"✅ 日志已写入OSS：{log_oss_path}")

# 模块自测
if __name__ == "__main__":
    test_oss_path = "datasets/arpos/ARPOS/PIS-3252.zip"
    test_temp_path = "tmp/test_download.zip"
    try:
        # 测试流式读取
        for _ in list(oss_read_file_stream(test_oss_path))[:2]:
            pass
        oss_write_log("OSS工具类接口测试成功")
        print("✅ oss_file_reader.py 自检完成！")
    except Exception as e:
        print(f"❌ 自检失败：{str(e)}")