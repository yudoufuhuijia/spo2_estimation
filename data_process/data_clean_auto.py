"""
修复版数据清洗脚本 - 解决模糊检测误判问题
改进：
1. 禁用基于Laplacian的模糊检测（不适用于此数据集）
2. 改用简单的图片有效性检测（非全黑、非全白、尺寸正常）
3. 保留其他清洗逻辑
"""

import oss2
import os
import numpy as np
import cv2
from config.oss_config import ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME
from utils.oss_file_reader import oss_read_file_stream, oss_write_log
import zipfile
import io
import tempfile
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import traceback

# ===================== 全局配置 =====================
class Config:
    PIS_OSS_DIR = "datasets/arpos/ARPOS/"
    VIPL_OSS_ROOT = "datasets/vipl/train/"

    OSS_CLEAN_LOG = "logs/clean_log.txt"
    OSS_VALID_INDEX = "processed_data/valid_data_index.csv"
    OSS_REPORT = "processed_data/cleaning_report.txt"

    # 图片质量检测参数
    MIN_IMAGE_SIZE = 50  # 最小图片尺寸（像素）
    MIN_BRIGHTNESS = 5   # 最小平均亮度（0-255）
    MAX_BRIGHTNESS = 250 # 最大平均亮度（0-255）

    MIN_VIDEO_FRAMES = 30

    PIS_ZIP_PREFIX = "PIS"
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
    VIDEO_EXTENSIONS = (".mp4", ".avi")

    MAX_WORKERS = 2
    VIPL_SAMPLE_SIZE = 512 * 1024

# ===================== 工具函数 =====================
def init_oss_bucket():
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

def is_image_valid(image):
    """
    改进的图片有效性检测
    不使用Laplacian模糊检测，改用基础质量检测
    """
    try:
        # 检查1: 尺寸是否正常
        h, w = image.shape[:2]
        if h < Config.MIN_IMAGE_SIZE or w < Config.MIN_IMAGE_SIZE:
            return False, "尺寸过小"

        # 检查2: 是否全黑或全白
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = gray.mean()

        if mean_brightness < Config.MIN_BRIGHTNESS:
            return False, "图片过暗（可能全黑）"

        if mean_brightness > Config.MAX_BRIGHTNESS:
            return False, "图片过亮（可能全白）"

        # 检查3: 是否有内容变化（标准差>0）
        std_brightness = gray.std()
        if std_brightness < 1.0:
            return False, "无内容变化（纯色图片）"

        return True, "有效"

    except Exception as e:
        return False, f"检测异常: {str(e)}"

def extract_label_from_path(file_path, data_type):
    try:
        if data_type == "PIS3252":
            # 示例: cropped-1-13-12-857.png
            # 倒数第二个字段是标签
            parts = os.path.basename(file_path).split("-")
            if len(parts) >= 4:
                return parts[-2]
            return "unknown"
        elif data_type == "VIPL":
            parent_dir = os.path.basename(os.path.dirname(file_path))
            return parent_dir if parent_dir.isdigit() else "unknown"
    except:
        return "unknown"

def log_message(message, log_file=Config.OSS_CLEAN_LOG):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    try:
        oss_write_log(message, log_file)
    except:
        pass

# ===================== PIS数据处理 =====================
def process_single_pis_zip(args):
    zip_oss_path, bucket = args
    zip_basename = os.path.basename(zip_oss_path)
    valid_data = []
    stats = {
        "zip_name": zip_basename,
        "total_images": 0,
        "valid_images": 0,
        "too_small": 0,
        "too_dark": 0,
        "too_bright": 0,
        "no_variation": 0,
        "corrupted_images": 0,
        "errors": []
    }

    try:
        print(f"📦 正在处理: {zip_basename}")
        zip_data = b"".join(oss_read_file_stream(zip_oss_path))
        zip_stream = io.BytesIO(zip_data)

        with zipfile.ZipFile(zip_stream, 'r') as zip_file:
            all_files = zip_file.namelist()

            # 智能查找图片文件
            image_files = []
            for f in all_files:
                if any(keyword in f for keyword in ["Color", "cheeksCombined", "AfterExcersizeCropped"]):
                    if f.lower().endswith(Config.IMAGE_EXTENSIONS):
                        image_files.append(f)

            if not image_files:
                image_files = [f for f in all_files if f.lower().endswith(Config.IMAGE_EXTENSIONS)]

            stats["total_images"] = len(image_files)

            if not image_files:
                stats["errors"].append("未找到图片文件")
                return valid_data, stats

            image_files.sort()

            for img_path in image_files:
                try:
                    img_binary = zip_file.read(img_path)
                    nparr = np.frombuffer(img_binary, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        stats["corrupted_images"] += 1
                        continue

                    # 使用新的有效性检测
                    is_valid, reason = is_image_valid(img)

                    if not is_valid:
                        # 统计具体原因
                        if "尺寸" in reason:
                            stats["too_small"] += 1
                        elif "过暗" in reason:
                            stats["too_dark"] += 1
                        elif "过亮" in reason:
                            stats["too_bright"] += 1
                        elif "无内容" in reason:
                            stats["no_variation"] += 1
                        continue

                    label = extract_label_from_path(img_path, "PIS3252")

                    valid_data.append({
                        "data_type": "PIS3252",
                        "oss_path": zip_oss_path,
                        "inner_path": img_path,
                        "label": label,
                        "source_zip": zip_basename,
                        "image_shape": f"{img.shape[0]}x{img.shape[1]}"
                    })
                    stats["valid_images"] += 1

                except Exception as e:
                    stats["errors"].append(f"{img_path}: {str(e)}")

        print(f"✅ {zip_basename}: {stats['valid_images']}/{stats['total_images']} 张有效")

    except Exception as e:
        error_msg = f"压缩包处理失败: {str(e)}"
        stats["errors"].append(error_msg)
        print(f"❌ {zip_basename}: {error_msg}")

    return valid_data, stats

def clean_pis3252_data_parallel():
    bucket = init_oss_bucket()

    print(f"\n🔍 扫描OSS目录: {Config.PIS_OSS_DIR}")
    zip_files = []
    for obj in oss2.ObjectIterator(bucket, prefix=Config.PIS_OSS_DIR):
        file_name = obj.key
        if file_name.endswith(".zip") and Config.PIS_ZIP_PREFIX in os.path.basename(file_name):
            zip_files.append(file_name)

    total_zips = len(zip_files)
    print(f"📊 发现 {total_zips} 个PIS压缩包")
    log_message(f"开始批量处理PIS数据 | 总压缩包数: {total_zips}")

    if total_zips == 0:
        print("⚠️  未找到PIS压缩包")
        return [], []

    process_args = [(zip_path, bucket) for zip_path in zip_files]

    all_valid_data = []
    all_stats = []

    print("\n开始并行处理...")
    with Pool(processes=Config.MAX_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_pis_zip, process_args),
            total=total_zips,
            desc="处理PIS压缩包",
            unit="个"
        ))

    for valid_data, stats in results:
        all_valid_data.extend(valid_data)
        all_stats.append(stats)

    total_images = sum(s["total_images"] for s in all_stats)
    total_valid = sum(s["valid_images"] for s in all_stats)
    total_corrupted = sum(s["corrupted_images"] for s in all_stats)
    total_too_small = sum(s["too_small"] for s in all_stats)
    total_too_dark = sum(s["too_dark"] for s in all_stats)
    total_too_bright = sum(s["too_bright"] for s in all_stats)
    total_no_variation = sum(s["no_variation"] for s in all_stats)

    print(f"\n📈 PIS数据清洗完成:")
    print(f"   总图片数: {total_images}")
    print(f"   有效图片: {total_valid} ({total_valid/total_images*100 if total_images > 0 else 0:.1f}%)")
    print(f"   过滤详情:")
    print(f"     - 损坏图片: {total_corrupted}")
    print(f"     - 尺寸过小: {total_too_small}")
    print(f"     - 过暗图片: {total_too_dark}")
    print(f"     - 过亮图片: {total_too_bright}")
    print(f"     - 无内容变化: {total_no_variation}")

    log_message(f"PIS清洗完成 | 有效: {total_valid}/{total_images}")

    return all_valid_data, all_stats

# ===================== VIPL数据处理 =====================
def process_single_vipl_video_lightweight(args):
    video_oss_path, bucket = args

    try:
        obj_meta = bucket.head_object(video_oss_path)
        file_size = obj_meta.content_length

        if file_size < 100 * 1024:
            return None, {"error": f"文件过小({file_size} bytes)"}

        partial_data = b""
        for chunk in oss_read_file_stream(video_oss_path, chunk_size=Config.VIPL_SAMPLE_SIZE):
            partial_data += chunk
            if len(partial_data) >= Config.VIPL_SAMPLE_SIZE:
                break

        is_valid_video = (
            partial_data[:4] == b'RIFF' or
            b'ftyp' in partial_data[:32] or
            b'moov' in partial_data[:512]
        )

        if not is_valid_video:
            return None, {"error": "非有效视频格式"}

        label = extract_label_from_path(video_oss_path, "VIPL")

        valid_data = {
            "data_type": "VIPL",
            "oss_path": video_oss_path,
            "inner_path": "",
            "label": label,
            "file_size": f"{file_size/(1024*1024):.2f}MB",
            "validated": "lightweight"
        }

        return valid_data, {"success": True}

    except Exception as e:
        return None, {"error": str(e)}

def clean_vipl_data_lightweight():
    bucket = init_oss_bucket()

    print(f"\n🔍 扫描VIPL视频: {Config.VIPL_OSS_ROOT}")
    video_files = []
    for obj in oss2.ObjectIterator(bucket, prefix=Config.VIPL_OSS_ROOT):
        if obj.key.lower().endswith(Config.VIDEO_EXTENSIONS):
            video_files.append(obj.key)

    total_videos = len(video_files)
    print(f"📊 发现 {total_videos} 个视频文件")
    print(f"⚡ 使用轻量级验证模式")
    log_message(f"开始处理VIPL数据 | 总视频数: {total_videos}")

    if total_videos == 0:
        print("⚠️  未找到VIPL视频")
        return [], []

    process_args = [(video_path, bucket) for video_path in video_files]

    valid_data = []
    error_count = 0

    print("\n开始处理...")
    with Pool(processes=Config.MAX_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_vipl_video_lightweight, process_args),
            total=total_videos,
            desc="验证VIPL视频",
            unit="个"
        ))

    for data, stats in results:
        if data:
            valid_data.append(data)
        else:
            error_count += 1

    print(f"\n📈 VIPL数据清洗完成:")
    print(f"   总视频数: {total_videos}")
    print(f"   有效视频: {len(valid_data)} ({len(valid_data)/total_videos*100 if total_videos > 0 else 0:.1f}%)")
    print(f"   无效视频: {error_count}")

    log_message(f"VIPL清洗完成 | 有效: {len(valid_data)}/{total_videos}")

    return valid_data, []

# ===================== 索引保存 =====================
def save_valid_index(valid_data):
    if not valid_data:
        print("⚠️  没有有效数据")
        return None

    df = pd.DataFrame(valid_data)
    os.makedirs("tmp", exist_ok=True)
    local_csv = "tmp/valid_data_index.csv"
    df.to_csv(local_csv, index=False, encoding="utf-8")

    bucket = init_oss_bucket()
    bucket.put_object_from_file(Config.OSS_VALID_INDEX, local_csv)

    print(f"✅ 索引文件已保存: {Config.OSS_VALID_INDEX}")
    log_message(f"索引文件生成完成 | 总记录数: {len(df)}")

    os.remove(local_csv)
    return df

def generate_report(pis_stats, vipl_stats, total_valid):
    report_lines = [
        "=" * 60,
        "数据清洗报告（修复版）",
        "=" * 60,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "【质量检测方法】",
        "- 禁用Laplacian模糊检测（不适用于此数据集）",
        "- 使用基础质量检测：尺寸、亮度、内容变化",
        "",
        "【PIS数据集统计】",
        f"处理压缩包数: {len(pis_stats)}",
    ]

    if pis_stats:
        total_pis_images = sum(s["total_images"] for s in pis_stats)
        total_pis_valid = sum(s["valid_images"] for s in pis_stats)
        total_corrupted = sum(s["corrupted_images"] for s in pis_stats)
        total_too_small = sum(s["too_small"] for s in pis_stats)
        total_too_dark = sum(s["too_dark"] for s in pis_stats)
        total_too_bright = sum(s["too_bright"] for s in pis_stats)
        total_no_variation = sum(s["no_variation"] for s in pis_stats)

        report_lines.extend([
            f"总图片数: {total_pis_images}",
            f"有效图片: {total_pis_valid} ({total_pis_valid/total_pis_images*100 if total_pis_images > 0 else 0:.2f}%)",
            f"过滤统计:",
            f"  - 损坏: {total_corrupted}",
            f"  - 尺寸过小: {total_too_small}",
            f"  - 过暗: {total_too_dark}",
            f"  - 过亮: {total_too_bright}",
            f"  - 无内容变化: {total_no_variation}",
            "",
            "各压缩包详情:",
        ])
        for s in pis_stats[:10]:  # 只显示前10个
            report_lines.append(
                f"  {s['zip_name']}: {s['valid_images']}/{s['total_images']}"
            )
        if len(pis_stats) > 10:
            report_lines.append(f"  ... 共{len(pis_stats)}个压缩包")

    report_lines.extend([
        "",
        "【最终结果】",
        f"总有效数据量: {total_valid}",
        f"索引文件: {Config.OSS_VALID_INDEX}",
        "=" * 60
    ])

    report_text = "\n".join(report_lines)

    bucket = init_oss_bucket()
    bucket.put_object(Config.OSS_REPORT, report_text.encode('utf-8'))

    print(f"\n✅ 清洗报告已生成: {Config.OSS_REPORT}")
    print(report_text)

    return report_text

# ===================== 主流程 =====================
def main():
    print("=" * 60)
    print("修复版数据清洗脚本启动")
    print("=" * 60)
    print("改进: 禁用不适用的模糊检测，使用基础质量检测")
    print("=" * 60)

    start_time = datetime.now()
    log_message("===== 数据清洗任务开始（修复版）=====")

    try:
        print("\n【阶段 1/3】处理PIS数据集...")
        valid_pis, pis_stats = clean_pis3252_data_parallel()

        print("\n【阶段 2/3】处理VIPL数据集...")
        valid_vipl, vipl_stats = clean_vipl_data_lightweight()

        print("\n【阶段 3/3】生成数据索引...")
        all_valid = valid_pis + valid_vipl
        valid_df = save_valid_index(all_valid)

        total_valid = len(all_valid)
        generate_report(pis_stats, vipl_stats, total_valid)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("🎉 数据清洗全部完成！")
        print("=" * 60)
        print(f"总耗时: {duration:.2f} 秒")
        print(f"PIS有效数据: {len(valid_pis)}")
        print(f"VIPL有效数据: {len(valid_vipl)}")
        print(f"总有效数据: {total_valid}")
        print(f"索引文件: {Config.OSS_VALID_INDEX}")
        print(f"清洗报告: {Config.OSS_REPORT}")
        print("=" * 60)

        log_message(f"===== 任务完成 | 总耗时: {duration:.2f}s | 有效数据: {total_valid} =====")

        return True

    except Exception as e:
        error_msg = f"主流程异常: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg)
        print(f"\n❌ {error_msg}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)