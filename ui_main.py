"""
PyQt5主界面 - ui_main.py (原布局最终版)
100%保留原始布局/样式/功能，仅修复QLabel创建报错，与核心稳速版完美匹配
界面：4大区域
1. 视频显示（左上）
2. rPPG波形（左下）
3. SpO2/HR结果（右上）
4. 控制面板（右下）
日期：2026-02-11
"""
import os
# 解决OMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 先导入numpy再导入cv2，解决NumPy重加载警告
import numpy as np
import sys, cv2, time
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

sys.path.insert(0, str(Path(__file__).parent))
from system_core import SystemCore

class SpO2MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.core = None
        self.signal_buffer = []
        self.init_ui()
        self.init_core()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)  # 30fps刷新

    def init_ui(self):
        # 窗口基础设置，保留原始尺寸
        self.setWindowTitle("SpO2血氧估计系统")
        self.setGeometry(100, 100, 1400, 900)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # ========== 左侧区域：视频+波形 ==========
        left_layout = QVBoxLayout()
        # 视频显示模块
        video_group = QGroupBox("📹 实时视频")
        video_group.setFont(QFont("Arial", 12, QFont.Bold))
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; background: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("等待视频流...")
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)

        # 波形显示模块
        signal_group = QGroupBox("📊 rPPG血氧信号")
        signal_group.setFont(QFont("Arial", 12, QFont.Bold))
        signal_layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('left', 'CHROM幅度')
        self.plot_widget.setLabel('bottom', '帧数')
        self.plot_widget.setTitle('实时CHROM信号', color='k', size='12pt')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setFixedHeight(250)
        self.signal_curve = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2))
        signal_layout.addWidget(self.plot_widget)
        signal_group.setLayout(signal_layout)
        left_layout.addWidget(signal_group)
        main_layout.addLayout(left_layout, 2)

        # ========== 右侧区域：结果+控制 ==========
        right_layout = QVBoxLayout()
        # 检测结果模块
        result_group = QGroupBox("📈 检测结果")
        result_group.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout = QVBoxLayout()
        result_layout.setSpacing(20)

        # SpO2显示（修复QLabel创建报错，原样式）
        spo2_container = QVBoxLayout()
        spo2_title = QLabel("血氧饱和度 (SpO₂)")
        spo2_title.setFont(QFont("Arial", 14))
        spo2_title.setAlignment(Qt.AlignCenter)
        spo2_container.addWidget(spo2_title)

        self.spo2_value = QLabel("--.--%")
        self.spo2_value.setFont(QFont("Arial", 48, QFont.Bold))
        self.spo2_value.setAlignment(Qt.AlignCenter)
        self.spo2_value.setStyleSheet("color: #2196F3; padding: 20px; background: #E3F2FD; border-radius: 10px;")
        spo2_container.addWidget(self.spo2_value)

        spo2_range = QLabel("正常: 95-100%")
        spo2_range.setFont(QFont("Arial", 10))
        spo2_range.setAlignment(Qt.AlignCenter)
        spo2_range.setStyleSheet("color: gray;")
        spo2_container.addWidget(spo2_range)
        result_layout.addLayout(spo2_container)

        # HR显示（原样式，与稳速核心联动）
        hr_container = QVBoxLayout()
        hr_title = QLabel("心率 (HR)")
        hr_title.setFont(QFont("Arial", 14))
        hr_title.setAlignment(Qt.AlignCenter)
        hr_container.addWidget(hr_title)

        self.hr_value = QLabel("-- BPM")
        self.hr_value.setFont(QFont("Arial", 48, QFont.Bold))
        self.hr_value.setAlignment(Qt.AlignCenter)
        self.hr_value.setStyleSheet("color: #F44336; padding: 20px; background: #FFEBEE; border-radius: 10px;")
        hr_container.addWidget(self.hr_value)

        hr_range = QLabel("正常: 60-100 BPM")
        hr_range.setFont(QFont("Arial", 10))
        hr_range.setAlignment(Qt.AlignCenter)
        hr_range.setStyleSheet("color: gray;")
        hr_container.addWidget(hr_range)
        result_layout.addLayout(hr_container)

        # FPS显示
        self.fps_label = QLabel("FPS: 0.0", font=QFont("Arial", 10), alignment=Qt.AlignCenter)
        result_layout.addWidget(self.fps_label)
        result_layout.addStretch()
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # 控制按钮模块（原样式+原功能）
        control_group = QGroupBox("🎮 系统控制")
        control_group.setFont(QFont("Arial", 12, QFont.Bold))
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)

        # 数据源选择
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("数据源:", font=QFont("Arial", 10)))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头", "视频文件"])
        self.source_combo.setFont(QFont("Arial", 10))
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        source_layout.addWidget(self.source_combo)
        control_layout.addLayout(source_layout)

        # 开始/暂停按钮
        self.start_btn = QPushButton("▶ 开始检测")
        self.start_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.start_btn.setStyleSheet("""
            QPushButton {background: #4CAF50; color: white; padding: 15px; border-radius: 5px;}
            QPushButton:hover {background: #45a049;}
        """)
        self.start_btn.clicked.connect(self.on_start_clicked)
        control_layout.addWidget(self.start_btn)

        # 保存结果按钮
        save_btn = QPushButton("💾 保存结果")
        save_btn.setFont(QFont("Arial", 11))
        save_btn.setStyleSheet("""
            QPushButton {background: #2196F3; color: white; padding: 12px; border-radius: 5px;}
            QPushButton:hover {background: #0b7dda;}
        """)
        save_btn.clicked.connect(self.on_save_clicked)
        control_layout.addWidget(save_btn)

        # 重置系统按钮
        reset_btn = QPushButton("🔄 重置系统")
        reset_btn.setFont(QFont("Arial", 11))
        reset_btn.setStyleSheet("""
            QPushButton {background: #FF9800; color: white; padding: 12px; border-radius: 5px;}
            QPushButton:hover {background: #e68900;}
        """)
        reset_btn.clicked.connect(self.on_reset_clicked)
        control_layout.addWidget(reset_btn)

        # 状态显示
        self.status_label = QLabel("状态: 就绪")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: green; padding: 10px; background: #F0F0F0; border-radius: 5px;")
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        main_layout.addLayout(right_layout, 1)

    def init_core(self):
        try:
            self.core = SystemCore("model_quantized/quantized_model_int8.pth")
        except Exception as e:
            QMessageBox.warning(self, "核心加载警告", f"系统核心加载失败：{str(e)}")

    def update_ui(self):
        if not self.core: return
        result = self.core.get_result()
        if not result: return

        # 更新视频流
        if result['frame'] is not None:
            self.update_video(result['frame'])

        # 更新CHROM波形
        if result['signal'] is not None:
            self.signal_buffer = list(result['signal'][-150:])
            self.signal_curve.setData(self.signal_buffer)

        # 更新SpO2+颜色智能标注（正常绿色/异常红色）
        if result['spo2'] is not None:
            self.spo2_value.setText(f"{result['spo2']:.2f}%")
            if result['spo2'] < 95:
                self.spo2_value.setStyleSheet("color: #F44336; padding: 20px; background: #FFEBEE; border-radius: 10px;")
            elif 95 <= result['spo2'] <= 100:
                self.spo2_value.setStyleSheet("color: #4CAF50; padding: 20px; background: #E8F5E9; border-radius: 10px;")

        # 更新HR+颜色智能标注（正常绿色/异常红色），联动稳速核心
        if result['hr'] is not None:
            self.hr_value.setText(f"{int(result['hr'])} BPM")
            if result['hr'] < 60 or result['hr'] > 100:
                self.hr_value.setStyleSheet("color: #F44336; padding: 20px; background: #FFEBEE; border-radius: 10px;")
            else:
                self.hr_value.setStyleSheet("color: #4CAF50; padding: 20px; background: #E8F5E9; border-radius: 10px;")

        # 更新实时FPS
        self.fps_label.setText(f"FPS: {result['fps']:.1f}")

    def update_video(self, frame):
        try:
            # 视频帧格式转换，适配PyQt5显示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            tw, th = 640, 480
            # 等比例缩放，防止视频拉伸
            if w/h > tw/th:
                nw, nh = tw, int(h * tw / w)
            else:
                nh, nw = th, int(w * th / h)
            resized = cv2.resize(rgb_frame, (nw, nh))
            # 转换为QImage
            q_img = QImage(resized.data, nw, nh, nw*3, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
        except: pass

    def on_source_changed(self, idx):
        if not self.core:
            QMessageBox.warning(self, "警告", "系统核心未初始化，请重启程序")
            return
        if idx == 1:
            # 选择视频文件
            file_path, _ = QFileDialog.getOpenFileName(self, "选择检测视频", "", "视频文件 (*.mp4 *.avi *.mov)")
            if file_path:
                if self.core.set_video_source(file_path):
                    self.status_label.setText(f"状态: 已加载视频 - {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "警告", "视频文件加载失败，请检查文件格式")
        else:
            # 选择摄像头
            if self.core.set_video_source(0):
                self.status_label.setText("状态: 已加载摄像头")
            else:
                QMessageBox.warning(self, "警告", "摄像头加载失败，请检查设备连接")

    def on_start_clicked(self):
        if not self.core:
            QMessageBox.warning(self, "警告", "系统核心未初始化，请重启程序")
            return
        if self.start_btn.text() == "▶ 开始检测":
            self.core.start()
            self.start_btn.setText("⏸ 暂停")
            self.start_btn.setStyleSheet("""
                QPushButton {background: #FF9800; color: white; padding: 15px; border-radius: 5px;}
                QPushButton:hover {background: #e68900;}
            """)
            self.status_label.setText("状态: 检测中")
        else:
            self.core.pause()
            self.start_btn.setText("▶ 开始检测")
            self.start_btn.setStyleSheet("""
                QPushButton {background: #4CAF50; color: white; padding: 15px; border-radius: 5px;}
                QPushButton:hover {background: #45a049;}
            """)
            self.status_label.setText("状态: 已暂停")

    def on_save_clicked(self):
        if not self.core:
            QMessageBox.warning(self, "警告", "系统核心未初始化，请重启程序")
            return
        # 保存检测结果到results目录
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path("results")
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"result_{ts}.txt"
            # 写入结果
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(f"SpO2血氧检测结果\n")
                f.write(f"{'='*40}\n")
                f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"血氧饱和度(SpO₂): {self.spo2_value.text()}\n")
                f.write(f"心率(HR): {self.hr_value.text()}\n")
                f.write(f"实时帧率(FPS): {self.fps_label.text().replace('FPS: ', '')}\n")
                f.write(f"数据源: {self.status_label.text().replace('状态: ', '')}\n")
                f.write(f"{'='*40}\n")
            QMessageBox.information(self, "保存成功", f"检测结果已保存至：\n{save_path.resolve()}")
            self.status_label.setText("状态: 结果已保存")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"结果保存出错：{str(e)}")

    def on_reset_clicked(self):
        if not self.core:
            QMessageBox.warning(self, "警告", "系统核心未初始化，请重启程序")
            return
        if QMessageBox.question(self, "确认重置", "是否确定重置系统？所有检测数据将被清空", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            self.core.reset()
            # 清空界面显示
            self.spo2_value.setText("--.--%")
            self.hr_value.setText("-- BPM")
            self.fps_label.setText("FPS: 0.0")
            self.signal_buffer = []
            self.signal_curve.setData([])
            self.video_label.setText("等待视频流...")
            # 重置按钮状态
            self.start_btn.setText("▶ 开始检测")
            self.start_btn.setStyleSheet("""
                QPushButton {background: #4CAF50; color: white; padding: 15px; border-radius: 5px;}
                QPushButton:hover {background: #45a049;}
            """)
            self.status_label.setText("状态: 已重置")

    def closeEvent(self, event):
        # 窗口关闭时停止核心线程
        if self.core:
            self.core.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 统一跨平台样式
    window = SpO2MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()