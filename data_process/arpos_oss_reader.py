import zipfile
import io
import cv2
import numpy as np
import os
from utils.oss_file_reader import oss_read_file_stream, oss_write_log
from config.oss_config import bucket


# ===================== 固定配置（匹配你的OSS真实路径）=====================
ARPOS_OSS_PATH = "datasets/arpos/ARPOS/PIS-3252.zip"
TEST_IMAGE_DIR = "PIS-3252/AfterExcersizeCropped/Color/cheeksCombined/"
LOCAL_SAVE_DIR = "results"
OSS_SAVE_DIR = "processed_data"
# =========================================================================

# 自动创建本地输出目录
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

def read_arpos_image_seq_from_oss(image_dir):
    """
    不解压缩包，从OSS流式读取图片序列
    """
    # 读取压缩包二进制流
    zip_data = b"".join(oss_read_file_stream(ARPOS_OSS_PATH))
    zip_stream = io.BytesIO(zip_data)

    with zipfile.ZipFile(zip_stream, 'r') as zip_file:
        # 筛选指定目录下的图片文件
        image_files = [
            f for f in zip_file.namelist()
            if f.startswith(image_dir) and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not image_files:
            raise FileNotFoundError(f"路径 {image_dir} 下无图片文件")

        # 排序保证帧顺序正确
        image_files.sort()
        total_images = len(image_files)
        log_info = f"图片序列读取完成 | 路径：{image_dir} | 总数量：{total_images}"
        oss_write_log(log_info)
        print(f"✅ {log_info}")

        # 逐张解析图片
        for img_path in image_files:
            img_binary = zip_file.read(img_path)
            nparr = np.frombuffer(img_binary, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"⚠️  跳过损坏文件：{img_path}")
                continue
            yield img

if __name__ == "__main__":
    try:
        # 生成器读取图片
        image_generator = read_arpos_image_seq_from_oss(TEST_IMAGE_DIR)
        # 提取前3张测试图片
        test_images = list(image_generator)[:3]

        if not test_images:
            raise RuntimeError("未读取到有效图片数据")

        # 本地保存 + 上传OSS
        for idx, img in enumerate(test_images, start=1):
            local_file = os.path.join(LOCAL_SAVE_DIR, f"pis3252_sample_{idx}.png")
            cv2.imwrite(local_file, img)
            oss_file = f"{OSS_SAVE_DIR}/pis3252_sample_{idx}.png"
            bucket.put_object_from_file(oss_file, local_file)
            print(f"✅ 第{idx}张图片处理完成：本地{local_file} | OSS{oss_file}")

        oss_write_log(f"成功处理并上传{len(test_images)}张测试图片")
        print("\n🎉 脚本全流程执行完成，无报错！")

    except Exception as e:
        error_msg = f"执行异常：{str(e)}"
        oss_write_log(error_msg)
        print(f"❌ {error_msg}")