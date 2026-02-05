import cv2
import os
import tempfile
from utils.oss_file_reader import oss_read_file_stream, oss_write_log
from config.oss_config import bucket

# ===================== 核心配置（严格匹配你的OSS真实路径） =====================
# OSS上VIPL数据集的根目录（来自你的截图）
VIPL_OSS_ROOT_PATH = "datasets/vipl/train/1/"
# 测试视频的完整OSS路径（匹配真实文件：video1.mp4.avi）
TEST_VIDEO_OSS_PATH = f"{VIPL_OSS_ROOT_PATH}video1.mp4.avi"
# 本地/OSS输出路径
LOCAL_OUTPUT_DIR = "results"
LOCAL_FRAME_PATH = os.path.join(LOCAL_OUTPUT_DIR, "vipl_test_frame.png")
OSS_FRAME_PATH = "processed_data/vipl_test_frame.png"


def load_vipl_video_from_oss(video_oss_path):
    """
    修复版：正确管理临时文件生命周期，解决write to closed file错误
    流式读取OSS视频→写入临时文件→OpenCV打开，全程文件句柄合法
    """
    # 1. 创建临时视频文件（不自动删除，手动管理）
    temp_video_path = tempfile.mktemp(suffix=".avi")
    try:
        # 2. 打开临时文件，流式写入OSS数据（全程保持文件打开，直到写入完成）
        print(f"🔍 开始流式读取OSS视频：{video_oss_path}")
        with open(temp_video_path, "wb") as temp_video_file:
            for chunk in oss_read_file_stream(video_oss_path, chunk_size=1024 * 1024):
                temp_video_file.write(chunk)

        # 3. OpenCV打开临时文件（此时文件已写入完成，无关闭冲突）
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"视频无法打开，请检查格式/路径：{video_oss_path}")

        # 4. 获取视频基础信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 5. 日志记录
        log_content = (f"VIPL视频读取成功 | 文件名：{os.path.basename(video_oss_path)} | "
                       f"帧率：{fps:.2f} | 总帧数：{frame_count} | 分辨率：{width}x{height}")
        oss_write_log(log_content)
        print(f"✅ {log_content}")

        # 6. 逐帧生成器
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    yield frame
                else:
                    break
        finally:
            # 释放资源
            cap.release()

    finally:
        # 7. 所有操作完成后，删除临时文件
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# ===================== 测试主逻辑 =====================
if __name__ == "__main__":
    # 自动创建本地输出目录
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    try:
        # 加载视频帧生成器
        video_frames = load_vipl_video_from_oss(TEST_VIDEO_OSS_PATH)

        # 读取前5帧
        frame_list = []
        for idx, frame in enumerate(video_frames):
            frame_list.append(frame)
            if idx >= 4:
                break

        if len(frame_list) < 3:
            raise RuntimeError("视频帧数不足，无法读取第3帧")

        # 保存第3帧到本地
        test_frame = frame_list[2]
        cv2.imwrite(LOCAL_FRAME_PATH, test_frame)
        print(f"✅ 测试帧保存至本地：{LOCAL_FRAME_PATH}")

        # 上传测试帧到OSS
        bucket.put_object_from_file(OSS_FRAME_PATH, LOCAL_FRAME_PATH)
        print(f"✅ 测试帧上传至OSS：{OSS_FRAME_PATH}")

        # 最终日志
        oss_write_log(f"VIPL处理完成，测试帧OSS路径：{OSS_FRAME_PATH}")
        print("🎉 VIPL视频测试全流程执行完成！")

    except Exception as e:
        error_msg = f"VIPL视频处理失败：{str(e)}"
        oss_write_log(error_msg)
        print(f"❌ {error_msg}")