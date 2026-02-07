import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import warnings
import os

# 优先关闭TensorFlow oneDNN冗余日志（必须在导入MTCNN前执行）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 尝试导入MTCNN（根据诊断结果适配）
try:
    from mtcnn import MTCNN

    HAS_MTCNN = True
except ImportError:
    print("⚠️  MTCNN未安装，将使用OpenCV Haar Cascade备选方案")
    HAS_MTCNN = False

# 忽略无关警告
warnings.filterwarnings('ignore')


class FaceDetector:
    """
    轻量化人脸检测器（完全适配MTCNN诊断结果）
    核心适配：MTCNN仅支持stages/device参数，无min_face_size/scale_factor
    """

    def __init__(
            self,
            method: str = 'mtcnn',
            min_face_size: int = 40,  # 仅用于检测后过滤，不传入MTCNN
            confidence_threshold: float = 0.9,
            mtcnn_device: str = 'CPU:0'  # MTCNN支持的device参数（诊断结果确认）
    ):
        """
        初始化人脸检测器（参数适配诊断版MTCNN）

        Args:
            method: 检测方法 ('mtcnn'优先，'haar'备选)
            min_face_size: 最小人脸尺寸（检测后过滤，像素）
            confidence_threshold: 置信度阈值（MTCNN结果过滤）
            mtcnn_device: MTCNN设备（诊断支持'CPU:0'，无需修改）
        """
        self.method = method
        self.min_face_size = min_face_size  # 仅用于后过滤
        self.confidence_threshold = confidence_threshold
        self.mtcnn_device = mtcnn_device

        self.detector = None
        self.detection_count = 0  # 总检测次数
        self.total_time = 0.0  # 总检测耗时（秒）

        # 初始化检测器（自动处理参数适配）
        self._init_detector()

    def _init_detector(self):
        """初始化检测器（根据方法自动适配，失败降级）"""
        print(f"🔧 初始化人脸检测器 (方法: {self.method})...")

        if self.method == 'mtcnn' and HAS_MTCNN:
            self._init_mtcnn_detector()  # 适配诊断版MTCNN
        else:
            # MTCNN不可用，强制切换到Haar
            print(f"⚠️  MTCNN不可用（未安装/初始化失败），自动切换到Haar Cascade")
            self.method = 'haar'
            self._init_haar_detector()

    def _init_mtcnn_detector(self):
        """初始化MTCNN（严格按诊断结果传参：仅stages/device）"""
        try:
            # 诊断确认：MTCNN.__init__仅支持stages和device参数
            self.detector = MTCNN(
                stages='face_and_landmarks_detection',  # 默认值，保留显式传参
                device=self.mtcnn_device  # 诊断支持的设备参数
            )
            print(f"✅ MTCNN检测器初始化成功")
            print(f"   设备: {self.mtcnn_device}")
            print(f"   关键点支持: ✅（自动返回双眼/鼻子/双嘴角）")

        except TypeError as e:
            # 极端情况：参数仍不兼容，尝试无参数初始化（诊断推荐最安全方式）
            print(f"⚠️  MTCNN参数异常: {str(e)[:100]}")
            print("🔄 尝试无参数初始化MTCNN（诊断推荐安全方案）...")
            try:
                self.detector = MTCNN()  # 无参数初始化（诊断确认可用）
                print(f"✅ MTCNN无参数初始化成功")
            except Exception as e2:
                # MTCNN完全不可用，切换到Haar
                print(f"❌ MTCNN初始化失败: {str(e2)[:100]}")
                self.method = 'haar'
                self._init_haar_detector()

    def _init_haar_detector(self):
        """初始化Haar Cascade备选检测器（确保降级可用）"""
        # 加载OpenCV自带的人脸级联模型（无需额外下载）
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError(f"❌ Haar模型加载失败，路径: {cascade_path}")

        print(f"✅ Haar Cascade检测器初始化成功")
        print(f"   最小人脸尺寸: {self.min_face_size}px")

    def detect(
            self,
            image: np.ndarray,
            return_landmarks: bool = True
    ) -> List[Dict]:
        """
        检测人脸（统一输出格式，兼容MTCNN/Haar）

        Args:
            image: 输入图像（BGR格式，如cv2.imread结果）
            return_landmarks: 是否返回关键点（仅MTCNN生效）

        Returns:
            检测结果列表，每个结果含：
            - box: [x, y, width, height] 人脸框坐标
            - confidence: 置信度（MTCNN: 0~1，Haar: 1.0）
            - landmarks: 关键点字典（仅MTCNN，含left_eye/right_eye/nose/mouth_left/mouth_right）
        """
        # 输入合法性检查
        if image is None or image.size == 0:
            return []

        # 记录检测耗时
        start_time = time.time()

        # 按方法执行检测
        if self.method == 'mtcnn':
            results = self._detect_mtcnn(image, return_landmarks)
        else:
            results = self._detect_haar(image)

        # 更新性能统计
        elapsed = time.time() - start_time
        self.detection_count += 1
        self.total_time += elapsed

        return results

    def _detect_mtcnn(
            self,
            image: np.ndarray,
            return_landmarks: bool
    ) -> List[Dict]:
        """MTCNN检测（适配诊断版输出，增加后过滤逻辑）"""
        # MTCNN要求输入RGB格式，转换BGR→RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行检测（诊断确认detect_faces方法可用）
        raw_detections = self.detector.detect_faces(rgb_image)
        filtered_results = []

        # 结果过滤（弥补MTCNN无初始化参数的问题）
        for det in raw_detections:
            # 1. 置信度过滤（排除低置信结果）
            confidence = det.get('confidence', 0.0)
            if confidence < self.confidence_threshold:
                continue

            # 2. 人脸尺寸过滤（替代MTCNN的min_face_size参数）
            box = det.get('box', [0, 0, 0, 0])  # [x, y, w, h]
            face_width, face_height = box[2], box[3]
            if face_width < self.min_face_size or face_height < self.min_face_size:
                continue

            # 构建统一输出格式
            result = {
                'box': box,
                'confidence': round(confidence, 3)  # 保留3位小数
            }

            # 3. 关键点处理（仅当需要且存在时添加）
            if return_landmarks and 'keypoints' in det:
                raw_kps = det['keypoints']
                # 提取5个核心关键点（与文档需求一致）
                core_kps = {
                    'left_eye': raw_kps.get('left_eye', (0, 0)),
                    'right_eye': raw_kps.get('right_eye', (0, 0)),
                    'nose': raw_kps.get('nose', (0, 0)),
                    'mouth_left': raw_kps.get('mouth_left', (0, 0)),
                    'mouth_right': raw_kps.get('mouth_right', (0, 0))
                }
                result['landmarks'] = core_kps

            filtered_results.append(result)

        return filtered_results

    def _detect_haar(self, image: np.ndarray) -> List[Dict]:
        """Haar Cascade检测（备选方案，统一输出格式）"""
        # Haar要求灰度图输入
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 执行检测（用min_face_size控制最小尺寸）
        raw_faces = self.detector.detectMultiScale(
            gray_image,
            scaleFactor=1.1,  # Haar固定参数，提升检测速度
            minNeighbors=5,  # 过滤误检
            minSize=(self.min_face_size, self.min_face_size)
        )

        # 转换为统一输出格式（无关键点，置信度固定为1.0）
        results = []
        for (x, y, w, h) in raw_faces:
            results.append({
                'box': [x, y, w, h],
                'confidence': 1.0,  # Haar无置信度，固定为1.0
                'landmarks': None  # Haar不支持关键点
            })

        return results

    def draw_detections(
            self,
            image: np.ndarray,
            detections: List[Dict],
            draw_landmarks: bool = True
    ) -> np.ndarray:
        """
        在图像上绘制检测结果（可视化验证）

        Args:
            image: 原始图像（BGR格式）
            detections: detect()返回的检测结果
            draw_landmarks: 是否绘制关键点（仅MTCNN结果生效）

        Returns:
            绘制后的图像（不修改原图，返回新图）
        """
        output_image = image.copy()
        landmark_color = (255, 0, 0)  # 关键点颜色：蓝色
        box_color = (0, 255, 0)  # 人脸框颜色：绿色
        text_color = (0, 255, 0)  # 置信度文字颜色：绿色

        for det in detections:
            x, y, w, h = det['box']
            confidence = det['confidence']

            # 1. 绘制人脸框
            cv2.rectangle(
                output_image,
                (x, y),  # 左上角坐标
                (x + w, y + h),  # 右下角坐标
                box_color,  # 颜色
                2  # 线宽
            )

            # 2. 绘制置信度文字（位于框上方）
            text = f"Conf: {confidence:.2f}"
            cv2.putText(
                output_image,
                text,
                (x, max(0, y - 10)),  # 文字位置（避免超出图像顶部）
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # 字体大小
                text_color,
                2  # 文字线宽
            )

            # 3. 绘制关键点（仅MTCNN结果）
            if draw_landmarks and det.get('landmarks'):
                landmarks = det['landmarks']
                for kp_name, (kp_x, kp_y) in landmarks.items():
                    # 绘制实心圆关键点（半径3，填充）
                    cv2.circle(
                        output_image,
                        (kp_x, kp_y),
                        3,
                        landmark_color,
                        -1  # -1表示填充圆
                    )

        return output_image

    def get_largest_face(self, detections: List[Dict]) -> Optional[Dict]:
        """获取检测结果中面积最大的人脸（单人场景专用）"""
        if not detections:
            return None
        # 按人脸面积（宽×高）排序，取最大
        return max(detections, key=lambda det: det['box'][2] * det['box'][3])

    def get_performance_stats(self) -> Dict:
        """获取检测性能统计（符合文档性能监控需求）"""
        if self.detection_count == 0:
            return {
                'total_detections': 0,
                'avg_time_ms': 0.0,
                'avg_fps': 0.0,
                'total_time_s': 0.0
            }

        avg_time_ms = (self.total_time / self.detection_count) * 1000  # 平均耗时（毫秒）
        avg_fps = self.detection_count / self.total_time  # 平均帧率（FPS）

        return {
            'total_detections': self.detection_count,
            'avg_time_ms': round(avg_time_ms, 2),
            'avg_fps': round(avg_fps, 2),
            'total_time_s': round(self.total_time, 2),
            'meets_target': avg_time_ms <= 50  # 是否满足单帧≤50ms目标
        }

    def reset_performance_stats(self):
        """重置性能统计（用于多轮测试）"""
        self.detection_count = 0
        self.total_time = 0.0


# ===================== 辅助工具函数（人脸对齐，可选扩展） =====================
def align_face(
        image: np.ndarray,
        landmarks: Dict,
        output_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    根据关键点对齐人脸（用于后续ROI提取，符合文档下一步需求）

    Args:
        image: 原始图像（BGR格式）
        landmarks: MTCNN返回的关键点字典
        output_size: 对齐后人脸尺寸

    Returns:
        对齐后的人脸图像（RGB格式），失败返回None
    """
    # 验证关键点完整性
    required_kps = ['left_eye', 'right_eye', 'nose']
    if not all(kp in landmarks for kp in required_kps):
        print("⚠️  关键点不完整，无法对齐人脸")
        return None

    left_eye = np.array(landmarks['left_eye'], dtype=np.float32)
    right_eye = np.array(landmarks['right_eye'], dtype=np.float32)
    nose = np.array(landmarks['nose'], dtype=np.float32)

    # 1. 计算双眼中心点和旋转角度（纠正人脸倾斜）
    eye_center = (left_eye + right_eye) / 2  # 双眼中心点
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # 2. 构建旋转矩阵（以双眼中心为旋转点，纠正角度）
    rotation_matrix = cv2.getRotationMatrix2D(
        center=(int(eye_center[0]), int(eye_center[1])),
        angle=eye_angle,
        scale=1.0
    )

    # 3. 旋转图像（纠正人脸倾斜）
    h, w = image.shape[:2]
    aligned_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC  # 高质量插值
    )

    # 4. 裁剪人脸区域（基于鼻子和双眼距离）
    eye_distance = np.linalg.norm(right_eye - left_eye)  # 双眼间距
    face_width = int(eye_distance * 2.5)  # 人脸宽度（双眼间距的2.5倍）
    face_height = int(face_width * 1.3)  # 人脸高度（宽高比1:1.3）

    # 计算裁剪坐标（以鼻子为中心）
    x1 = max(0, int(nose[0] - face_width / 2))
    y1 = max(0, int(nose[1] - face_height / 2))
    x2 = min(w, x1 + face_width)
    y2 = min(h, y1 + face_height)

    # 裁剪并缩放至目标尺寸
    face_crop = aligned_image[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    # 转换为RGB格式（适配后续模型输入）
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(face_rgb, output_size, interpolation=cv2.INTER_CUBIC)


# ===================== 测试代码（本地验证用，与文档步骤3一致） =====================
def test_face_detector_full(
        test_video_path: str = "../../test_videos/test_video_1.avi",  # 适配本地路径
        test_frame_count: int = 10  # 测试帧数（文档提取10帧）
):
    """
    完整测试函数（与文档步骤3基础功能测试一致）

    Args:
        test_video_path: 测试视频路径（本地相对路径）
        test_frame_count: 提取的测试帧数
    """
    print("=" * 70)
    print("📝 人脸检测器完整测试（适配MTCNN诊断版）")
    print("=" * 70)

    # 1. 初始化检测器（优先MTCNN）
    print("\n【1/4】初始化检测器")
    try:
        detector = FaceDetector(
            method='mtcnn',
            min_face_size=40,
            confidence_threshold=0.9
        )
        print(f"✅ 检测器初始化完成，当前方法: {detector.method}")
    except Exception as e:
        print(f"❌ 检测器初始化失败: {str(e)}")
        return

    # 2. 提取测试视频帧（文档步骤3.1提取10帧）
    print(f"\n【2/4】提取测试视频帧（共{test_frame_count}帧）")
    if not os.path.exists(test_video_path):
        print(f"⚠️  测试视频不存在: {test_video_path}")
        print("💡 请在项目根目录创建'test_videos'，放入test_video_1.avi")
        return

    # 读取视频并提取帧
    cap = cv2.VideoCapture(test_video_path)
    test_frames = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < test_frame_count:
        ret, frame = cap.read()
        if ret:
            test_frames.append(frame)
            frame_idx += 1
        else:
            break
    cap.release()

    if not test_frames:
        print(f"❌ 无法提取视频帧（视频损坏或格式不支持）")
        return
    print(f"✅ 成功提取 {len(test_frames)} 帧测试数据")

    # 3. 执行基础检测（文档步骤3.1性能测试）
    print(f"\n【3/4】执行人脸检测（性能统计）")
    detector.reset_performance_stats()  # 重置统计
    test_image_count = 3  # 文档测试3张图
    test_iterations = 3  # 文档每张重复3次

    for img_idx in range(min(test_image_count, len(test_frames))):
        frame = test_frames[img_idx]
        print(f"\n📷 测试图片 {img_idx + 1}/{test_image_count}")

        for iter_idx in range(test_iterations):
            detections = detector.detect(frame)
            # 获取单次检测耗时（总耗时差）
            stats = detector.get_performance_stats()
            single_time_ms = stats['avg_time_ms'] if stats['total_detections'] > 0 else 0.0

            print(f"   迭代 {iter_idx + 1}/{test_iterations}: "
                  f"人脸数={len(detections)}, "
                  f"耗时={single_time_ms:.2f}ms")

    # 4. 输出性能统计（文档步骤3.1预期输出）
    print(f"\n【4/4】性能统计汇总（目标：单帧≤50ms）")
    final_stats = detector.get_performance_stats()
    print(f"📈 性能结果:")
    print(f"   总检测次数: {final_stats['total_detections']}")
    print(f"   平均耗时: {final_stats['avg_time_ms']} ms/帧")
    print(f"   平均帧率: {final_stats['avg_fps']} FPS")
    print(f"   性能达标: {'✅' if final_stats['meets_target'] else '❌'}")

    # 5. 保存检测结果图片（文档步骤3.2）
    if test_frames:
        first_frame = test_frames[0]
        first_detections = detector.detect(first_frame)
        if first_detections:
            # 创建输出目录（文档路径）
            output_dir = "../../test_output/detection"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/face_detection_demo.jpg"

            # 绘制并保存结果
            result_img = detector.draw_detections(first_frame, first_detections)
            cv2.imwrite(output_path, result_img)
            print(f"\n✅ 检测结果已保存: {output_path}")

            # 打印检测详情（文档步骤3.1预期输出）
            print(f"\n📋 第一帧检测详情:")
            largest_face = detector.get_largest_face(first_detections)
            if largest_face:
                print(f"   最大人脸位置: {largest_face['box']}")
                print(f"   置信度: {largest_face['confidence']}")
                if largest_face.get('landmarks'):
                    print(f"   核心关键点:")
                    for kp_name, (x, y) in largest_face['landmarks'].items():
                        print(f"     {kp_name}: ({x}, {y})")

    print("\n" + "=" * 70)
    print("✅ 测试完成（符合文档步骤3基础功能测试要求）")
    print("=" * 70)


# 本地运行测试（直接执行脚本时触发）
if __name__ == "__main__":
    # 执行完整测试（与文档步骤3一致）
    test_face_detector_full(
        test_video_path="../../test_videos/test_video_1.avi",  # 本地视频路径
        test_frame_count=10  # 提取10帧测试（文档要求）
    )
