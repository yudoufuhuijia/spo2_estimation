import oss2  # 确保文件开头已导入 oss2 库
from typing import List


class OSSConnector:
    def __init__(self):
        # 第一步：先补充 bucket 初始化（若已有则跳过，参考下方说明）
        # 从环境变量获取 AccessKey（避免硬编码，符合安全规范）
        import os
        # self.access_key_id = "LTAI5tBXijvuE7F8oV95LCPP"  # 直接写真实密钥，去掉os.getenv()
        # self.access_key_secret = "Wk7LCwD2BI1GXRFWWLANmOmdKpqqXi"  # 直接写真实密钥
        # self.endpoint = "oss-cn-hangzhou.aliyuncs.com"  # 替换为你的 OSS 地域Endpoint（如华东1为cn-hangzhou）
        # self.bucket_name = "spo2-estimation"  # 固定Bucket名

        # 初始化 auth 和 bucket（关键：确保后续能调用OSS接口）
        self.auth = oss2.Auth(self.access_key_id.strip(), self.access_key_secret.strip())
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    # 第二步：添加 list_objects 方法（解决报错的核心）
    def list_objects(self, bucket_name: str, prefix: str, max_keys: int = 100) -> List[str]:
        """
        列举 OSS 指定 Bucket 下前缀为 prefix 的文件，返回文件 key 列表
        :param bucket_name: Bucket 名称（此处固定为 spo2-estimation，可忽略）
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