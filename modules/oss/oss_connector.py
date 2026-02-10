import oss2  # 确保文件开头已导入 oss2 库
from typing import List
import os  # 导入os模块读取环境变量


class OSSConnector:
    def __init__(self):
        """
        初始化OSS连接器（安全规范：密钥从环境变量读取，禁止硬编码）
        环境变量配置说明：
        - OSS_ACCESS_KEY_ID: 阿里云OSS AccessKey ID
        - OSS_ACCESS_KEY_SECRET: 阿里云OSS AccessKey Secret
        - OSS_ENDPOINT: OSS地域Endpoint（如华东1：oss-cn-hangzhou.aliyuncs.com）
        - OSS_BUCKET_NAME: OSS Bucket名称
        """
        # 从环境变量读取OSS配置（上传GitHub前必须移除硬编码密钥）
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.endpoint = os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")  # 默认值兜底
        self.bucket_name = os.getenv("OSS_BUCKET_NAME", "spo2-estimation")

        # 校验密钥是否存在
        if not all([self.access_key_id, self.access_key_secret]):
            raise ValueError("❌ 错误：OSS密钥未配置！请设置环境变量 OSS_ACCESS_KEY_ID/OSS_ACCESS_KEY_SECRET")

        # 初始化 auth 和 bucket（关键：确保后续能调用OSS接口）
        self.auth = oss2.Auth(self.access_key_id.strip(), self.access_key_secret.strip())
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    # 第二步：添加 list_objects 方法（解决报错的核心）
    def list_objects(self, bucket_name: str, prefix: str, max_keys: int = 100) -> List[str]:
        """
        列举 OSS 指定 Bucket 下前缀为 prefix 的文件，返回文件 key 列表
        :param bucket_name: Bucket 名称（此处类内已固定，可传入self.bucket_name）
        :param prefix: 文件前缀（如 "datasets/arpos/"，筛选该目录下的文件）
        :param max_keys: 最大返回数量
        :return: 文件 key 列表（如 ["datasets/arpos/PIS-256.zip"]）
        """
        object_keys = []
        # 使用 OSS SDK V1 的 object_iterator_v2 列举文件（稳定且兼容脚本调用）
        for obj in oss2.ObjectIteratorV2(
                self.bucket,
                prefix=prefix,
                max_keys=max_keys,
                fetch_owner=False  # 不获取所有者信息，提速
        ):
            object_keys.append(obj.key)
            # 达到 max_keys 数量时停止
            if len(object_keys) >= max_keys:
                break
        return object_keys


# 示例使用（可选，上传GitHub时可注释）
# if __name__ == "__main__":
#     connector = OSSConnector()
#     files = connector.list_objects("spo2-estimation", "datasets/arpos/")
#     print(f"列举到文件：{files}")