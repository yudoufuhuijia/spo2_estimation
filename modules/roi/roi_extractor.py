"""
ROI提取模块 - roi_extractor.py
功能：基于人脸关键点提取额头和脸颊ROI
性能目标：单帧≤10ms
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class ROIExtractor:
    """
    ROI提取器（基于MTCNN关键点）

    核心功能：
    1. 提取额头ROI（用于SpO2信号提取）
    2. 提取双脸颊ROI（辅助信号源）
    3. 自适应ROI尺寸（基于人脸大小）
    """

    def __init__(
            self,
            forehead_height_ratio: float = 0.25,  # 额头高度比例
            cheek_size_ratio: float = 0.15,  # 脸颊ROI尺寸比例
            min_roi_size: int = 20,  # 最小ROI尺寸
            enable_visualization: bool = False  # 是否可视化ROI
    ):
        """
        初始化ROI提取器

        Args:
            forehead_height_ratio: 额头ROI高度占人脸高度比例
            cheek_size_ratio: 脸颊ROI尺寸占人脸宽度比例
            min_roi_size: ROI最小边长（像素）
            enable_visualization: 是否绘制ROI边框
        """
        self.forehead_height_ratio = forehead_height_ratio
        self.cheek_size_ratio = cheek_size_ratio
        self.min_roi_size = min_roi_size
        self.enable_visualization = enable_visualization

        # 性能统计
        self.extraction_count = 0
        self.total_time = 0.0

    def extract_rois(
            self,
            image: np.ndarray,
            face_detection: Dict
    ) -> Dict[str, np.ndarray]:
        """
        提取所有ROI区域

        Args:
            image: 输入图像（BGR格式）
            face_detection: 人脸检测结果（2.8模块输出）
                {
                    'box': [x, y, w, h],
                    'landmarks': {
                        'left_eye': (x, y),
                        'right_eye': (x, y),
                        'nose': (x, y),
                        'mouth_left': (x, y),
                        'mouth_right': (x, y)
                    }
                }

        Returns:
            ROI字典：
            {
                'forehead': np.ndarray,      # 额头ROI图像
                'left_cheek': np.ndarray,    # 左脸颊ROI图像
                'right_cheek': np.ndarray,   # 右脸颊ROI图像
                'coords': {                   # ROI坐标信息
                    'forehead': (x, y, w, h),
                    'left_cheek': (x, y, w, h),
                    'right_cheek': (x, y, w, h)
                }
            }
        """
        start_time = time.time()

        # 验证输入
        if image is None or image.size == 0:
            return {}

        if 'landmarks' not in face_detection:
            # 降级方案：仅基于人脸框提取
            return self._extract_from_box(image, face_detection['box'])

        # 提取关键点
        landmarks = face_detection['landmarks']
        box = face_detection['box']

        # 计算ROI坐标
        forehead_coords = self._calc_forehead_roi(landmarks, box)
        left_cheek_coords = self._calc_left_cheek_roi(landmarks, box)
        right_cheek_coords = self._calc_right_cheek_roi(landmarks, box)

        # 裁剪ROI图像
        h, w = image.shape[:2]

        rois = {
            'forehead': self._crop_roi(image, forehead_coords, (h, w)),
            'left_cheek': self._crop_roi(image, left_cheek_coords, (h, w)),
            'right_cheek': self._crop_roi(image, right_cheek_coords, (h, w)),
            'coords': {
                'forehead': forehead_coords,
                'left_cheek': left_cheek_coords,
                'right_cheek': right_cheek_coords
            }
        }

        # 更新性能统计
        elapsed = time.time() - start_time
        self.extraction_count += 1
        self.total_time += elapsed

        return rois

    def _calc_forehead_roi(
            self,
            landmarks: Dict,
            box: List[int]
    ) -> Tuple[int, int, int, int]:
        """
        计算额头ROI坐标（基于双眼关键点）

        策略：
        1. 中心点：双眼中点
        2. 宽度：双眼间距 × 1.5
        3. 高度：人脸高度 × forehead_height_ratio
        4. Y坐标：双眼上方
        """
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']

        # 双眼中心点
        eye_center_x = (left_eye[0] + right_eye[0]) // 2
        eye_center_y = (left_eye[1] + right_eye[1]) // 2

        # 双眼间距
        eye_distance = abs(right_eye[0] - left_eye[0])

        # ROI尺寸（自适应）
        roi_width = max(int(eye_distance * 1.5), self.min_roi_size)
        roi_height = max(int(box[3] * self.forehead_height_ratio), self.min_roi_size)

        # ROI位置（双眼上方）
        roi_x = eye_center_x - roi_width // 2
        roi_y = eye_center_y - roi_height - int(eye_distance * 0.3)

        return (roi_x, roi_y, roi_width, roi_height)

    def _calc_left_cheek_roi(
            self,
            landmarks: Dict,
            box: List[int]
    ) -> Tuple[int, int, int, int]:
        """
        计算左脸颊ROI坐标

        策略：
        1. 中心点：左眼与左嘴角中点
        2. 尺寸：人脸宽度 × cheek_size_ratio
        """
        left_eye = landmarks['left_eye']
        mouth_left = landmarks['mouth_left']

        # 脸颊中心点
        cheek_center_x = left_eye[0]
        cheek_center_y = (left_eye[1] + mouth_left[1]) // 2

        # ROI尺寸
        roi_size = max(int(box[2] * self.cheek_size_ratio), self.min_roi_size)

        # ROI位置
        roi_x = cheek_center_x - roi_size // 2
        roi_y = cheek_center_y - roi_size // 2

        return (roi_x, roi_y, roi_size, roi_size)

    def _calc_right_cheek_roi(
            self,
            landmarks: Dict,
            box: List[int]
    ) -> Tuple[int, int, int, int]:
        """
        计算右脸颊ROI坐标
        """
        right_eye = landmarks['right_eye']
        mouth_right = landmarks['mouth_right']

        cheek_center_x = right_eye[0]
        cheek_center_y = (right_eye[1] + mouth_right[1]) // 2

        roi_size = max(int(box[2] * self.cheek_size_ratio), self.min_roi_size)

        roi_x = cheek_center_x - roi_size // 2
        roi_y = cheek_center_y - roi_size // 2

        return (roi_x, roi_y, roi_size, roi_size)

    def _crop_roi(
            self,
            image: np.ndarray,
            coords: Tuple[int, int, int, int],
            image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        安全裁剪ROI（处理边界情况）

        Args:
            image: 原始图像
            coords: ROI坐标 (x, y, w, h)
            image_shape: 图像尺寸 (height, width)

        Returns:
            裁剪后的ROI图像
        """
        x, y, w, h = coords
        img_h, img_w = image_shape

        # 边界检查与修正
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        x2 = max(x + 1, min(x + w, img_w))
        y2 = max(y + 1, min(y + h, img_h))

        # 裁剪
        roi = image[y:y2, x:x2]

        # 验证ROI有效性
        if roi.size == 0:
            # 返回最小有效ROI
            return np.zeros((self.min_roi_size, self.min_roi_size, 3), dtype=np.uint8)

        return roi

    def _extract_from_box(
            self,
            image: np.ndarray,
            box: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        降级方案：仅基于人脸框提取ROI（无关键点时）

        策略：
        - 额头：人脸框上1/4区域
        - 左脸颊：人脸框左侧中部
        - 右脸颊：人脸框右侧中部
        """
        x, y, w, h = box

        # 额头ROI（上1/4）
        forehead_coords = (
            x + w // 4,
            y,
            w // 2,
            h // 4
        )

        # 左脸颊ROI
        left_cheek_coords = (
            x,
            y + h // 3,
            w // 3,
            h // 3
        )

        # 右脸颊ROI
        right_cheek_coords = (
            x + 2 * w // 3,
            y + h // 3,
            w // 3,
            h // 3
        )

        img_h, img_w = image.shape[:2]

        return {
            'forehead': self._crop_roi(image, forehead_coords, (img_h, img_w)),
            'left_cheek': self._crop_roi(image, left_cheek_coords, (img_h, img_w)),
            'right_cheek': self._crop_roi(image, right_cheek_coords, (img_h, img_w)),
            'coords': {
                'forehead': forehead_coords,
                'left_cheek': left_cheek_coords,
                'right_cheek': right_cheek_coords
            }
        }

    def draw_rois(
            self,
            image: np.ndarray,
            roi_coords: Dict[str, Tuple]
    ) -> np.ndarray:
        """
        在图像上绘制ROI边框（用于可视化）

        Args:
            image: 原始图像
            roi_coords: ROI坐标字典

        Returns:
            绘制后的图像
        """
        output = image.copy()

        colors = {
            'forehead': (0, 255, 0),  # 绿色
            'left_cheek': (255, 0, 0),  # 蓝色
            'right_cheek': (0, 0, 255)  # 红色
        }

        labels = {
            'forehead': 'Forehead',
            'left_cheek': 'L-Cheek',
            'right_cheek': 'R-Cheek'
        }

        for roi_name, coords in roi_coords.items():
            if roi_name == 'forehead' or roi_name.endswith('cheek'):
                x, y, w, h = coords
                color = colors.get(roi_name, (255, 255, 255))

                # 绘制矩形
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

                # 添加标签
                label = labels.get(roi_name, roi_name)
                cv2.putText(output, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output

    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if self.extraction_count == 0:
            return {
                'total_extractions': 0,
                'avg_time_ms': 0.0,
                'meets_target': False
            }

        avg_time_ms = (self.total_time / self.extraction_count) * 1000

        return {
            'total_extractions': self.extraction_count,
            'avg_time_ms': round(avg_time_ms, 2),
            'meets_target': avg_time_ms <= 10  # 目标≤10ms
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self.extraction_count = 0
        self.total_time = 0.0


# ===================== 测试代码 =====================
def test_roi_extractor():
    """ROI提取器测试函数"""
    import sys
    import os
    sys.path.insert(0, '../..')
    from modules.detection.face_detector import FaceDetector

    print("=" * 70)
    print("📝 ROI提取模块测试")
    print("=" * 70)

    # 初始化模块
    print("\n【1/4】初始化模块")
    face_detector = FaceDetector(method='mtcnn')
    roi_extractor = ROIExtractor(enable_visualization=True)
    print("✅ 模块初始化完成")

    # 读取测试视频
    print("\n【2/4】读取测试视频")
    test_video = "../../test_videos/test_video_1.avi"

    if not os.path.exists(test_video):
        print(f"❌ 测试视频不存在: {test_video}")
        return

    cap = cv2.VideoCapture(test_video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ 无法读取视频帧")
        return

    print(f"✅ 成功读取测试帧，尺寸: {frame.shape[1]}x{frame.shape[0]}")

    # 检测人脸
    print("\n【3/4】检测人脸")
    detections = face_detector.detect(frame)

    if not detections:
        print("❌ 未检测到人脸")
        return

    face = detections[0]
    print(f"✅ 检测到人脸")
    print(f"   人脸框: {face['box']}")
    if face.get('landmarks'):
        print(f"   关键点数: {len(face['landmarks'])}")

    # 提取ROI
    print("\n【4/4】提取ROI")
    rois = roi_extractor.extract_rois(frame, face)

    print(f"✅ ROI提取完成")
    print(f"\n📊 提取结果:")
    for roi_name, roi_img in rois.items():
        if roi_name != 'coords':
            print(f"   {roi_name}: {roi_img.shape}")

    # 保存可视化结果
    os.makedirs("../../test_output/roi", exist_ok=True)

    # 绘制ROI边框
    result_img = roi_extractor.draw_rois(frame, rois['coords'])
    cv2.imwrite("../../test_output/roi/roi_visualization.jpg", result_img)

    # 保存各个ROI
    for roi_name, roi_img in rois.items():
        if roi_name != 'coords':
            roi_path = f"../../test_output/roi/{roi_name}.jpg"
            cv2.imwrite(roi_path, roi_img)

    print(f"\n✅ 可视化结果已保存: test_output/roi/")

    # 性能统计
    stats = roi_extractor.get_performance_stats()
    print(f"\n📈 性能统计:")
    print(f"   提取次数: {stats['total_extractions']}")
    print(f"   平均耗时: {stats['avg_time_ms']:.2f} ms")
    print(f"   性能达标: {'✅' if stats['meets_target'] else '❌'}")

    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)


if __name__ == "__main__":
    test_roi_extractor()