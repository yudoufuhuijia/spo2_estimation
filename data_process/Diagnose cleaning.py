"""
数据清洗问题诊断脚本
用于排查为什么所有图片都被过滤
"""

import oss2
import os
import numpy as np
import cv2
from skimage.filters import laplace
import zipfile
import io
from config.oss_config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME
from utils.oss_file_reader import oss_read_file_stream

# 初始化OSS
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)


def diagnose_pis_zip(zip_oss_path):
    """诊断单个PIS压缩包"""
    print("=" * 80)
    print(f"诊断压缩包: {os.path.basename(zip_oss_path)}")
    print("=" * 80)

    try:
        # 下载压缩包
        print("1️⃣ 下载压缩包...")
        zip_data = b"".join(oss_read_file_stream(zip_oss_path))
        print(f"   ✅ 压缩包大小: {len(zip_data) / (1024 * 1024):.2f} MB")

        zip_stream = io.BytesIO(zip_data)

        with zipfile.ZipFile(zip_stream, 'r') as zip_file:
            # 列出所有文件
            all_files = zip_file.namelist()
            print(f"\n2️⃣ 压缩包内总文件数: {len(all_files)}")

            # 显示前10个文件路径
            print("\n   前10个文件路径示例:")
            for i, f in enumerate(all_files[:10]):
                print(f"   [{i + 1}] {f}")

            # 查找图片文件
            print("\n3️⃣ 查找图片文件...")

            # 方法1: 包含关键词
            method1_files = []
            for f in all_files:
                if any(keyword in f for keyword in ["Color", "cheeksCombined", "AfterExcersizeCropped"]):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        method1_files.append(f)
            print(f"   方法1（关键词匹配）: 找到 {len(method1_files)} 个文件")

            # 方法2: 所有图片
            method2_files = [f for f in all_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            print(f"   方法2（所有图片）: 找到 {len(method2_files)} 个文件")

            if len(method1_files) > 0:
                print("\n   方法1找到的前5个文件:")
                for f in method1_files[:5]:
                    print(f"   - {f}")

            if len(method2_files) > 0:
                print("\n   方法2找到的前5个文件:")
                for f in method2_files[:5]:
                    print(f"   - {f}")

            # 测试读取第一张图片
            test_files = method1_files if len(method1_files) > 0 else method2_files

            if len(test_files) == 0:
                print("\n❌ 未找到任何图片文件！")
                return

            print(f"\n4️⃣ 测试读取前3张图片...")
            for idx, img_path in enumerate(test_files[:3], 1):
                try:
                    img_binary = zip_file.read(img_path)
                    nparr = np.frombuffer(img_binary, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        print(f"\n   ❌ 图片{idx}: {os.path.basename(img_path)}")
                        print(f"      解码失败！")
                        continue

                    # 计算模糊度
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    laplacian_var = laplace(gray).var()

                    # 提取标签
                    parts = os.path.basename(img_path).split("-")
                    label = parts[-2] if len(parts) >= 4 else "unknown"

                    print(f"\n   ✅ 图片{idx}: {os.path.basename(img_path)}")
                    print(f"      尺寸: {img.shape[0]}x{img.shape[1]}")
                    print(f"      模糊度: {laplacian_var:.2f}")
                    print(f"      判定: {'模糊 ❌' if laplacian_var < 30 else '清晰 ✅'}")
                    print(f"      标签: {label}")

                except Exception as e:
                    print(f"\n   ❌ 图片{idx}: 处理失败 - {str(e)}")

            # 统计模糊度分布
            print(f"\n5️⃣ 统计模糊度分布（采样100张）...")
            blur_values = []
            sample_files = test_files[:100]

            for img_path in sample_files:
                try:
                    img_binary = zip_file.read(img_path)
                    nparr = np.frombuffer(img_binary, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blur_values.append(laplace(gray).var())
                except:
                    continue

            if len(blur_values) > 0:
                blur_values = np.array(blur_values)
                print(f"\n   模糊度统计（共{len(blur_values)}张有效样本）:")
                print(f"   最小值: {blur_values.min():.2f}")
                print(f"   最大值: {blur_values.max():.2f}")
                print(f"   平均值: {blur_values.mean():.2f}")
                print(f"   中位数: {np.median(blur_values):.2f}")
                print(f"   25分位: {np.percentile(blur_values, 25):.2f}")
                print(f"   75分位: {np.percentile(blur_values, 75):.2f}")

                # 不同阈值下的通过率
                print(f"\n   不同模糊阈值的通过率:")
                for threshold in [10, 20, 30, 40, 50]:
                    passed = (blur_values >= threshold).sum()
                    rate = passed / len(blur_values) * 100
                    print(f"   阈值={threshold}: {passed}/{len(blur_values)} ({rate:.1f}%)")

            print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ 诊断失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主诊断流程"""
    print("\n数据清洗问题诊断")
    print("=" * 80)

    # 获取第一个PIS压缩包
    print("\n查找PIS压缩包...")
    pis_prefix = "datasets/arpos/ARPOS/"

    test_zip = None
    for obj in oss2.ObjectIterator(bucket, prefix=pis_prefix):
        if obj.key.endswith('.zip') and 'PIS' in os.path.basename(obj.key):
            test_zip = obj.key
            break

    if test_zip is None:
        print("❌ 未找到PIS压缩包")
        return

    print(f"✅ 找到测试压缩包: {test_zip}")

    # 诊断
    diagnose_pis_zip(test_zip)

    print("\n" + "=" * 80)
    print("📋 诊断建议:")
    print("=" * 80)
    print("1. 如果模糊度普遍<30，建议降低BLUR_THRESHOLD（如改为10或15）")
    print("2. 如果方法1找不到文件但方法2能找到，说明路径匹配有问题")
    print("3. 如果解码失败，说明图片文件可能损坏")
    print("4. 根据模糊度分布，选择合适的阈值（建议用25分位数）")
    print("=" * 80)


if __name__ == "__main__":
    main()