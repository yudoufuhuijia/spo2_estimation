# 阿里云OSS配置信息（安全规范：禁止硬编码密钥，从环境变量读取）
# ==================== 环境变量配置示例 ====================
# Linux/Mac 终端配置：
# export OSS_ACCESS_KEY_ID="你的AccessKey ID"
# export OSS_ACCESS_KEY_SECRET="你的AccessKey Secret"
# export OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"

# Windows PowerShell 配置：
# $env:OSS_ACCESS_KEY_ID="你的AccessKey ID"
# $env:OSS_ACCESS_KEY_SECRET="你的AccessKey Secret"
# $env:OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"

# ==================== 代码中读取方式 ====================
import os

access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
endpoint = os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com")  # 默认值兜底

# 校验配置（可选）
if not all([access_key_id, access_key_secret]):
    raise RuntimeError("❌ OSS配置未完成！请先设置上述环境变量")