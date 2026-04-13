import sys
import cv2
import time
import torch
import traceback
import threading
import queue
import dxcam
import ctypes
import os
import glob
import numpy as np
from control_utils import PID, calculate_fov_movement, start_mouse_move
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QGroupBox,
                             QTextEdit, QFileDialog, QMessageBox, QSizePolicy, QSplitter, QDialog, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QIcon, QKeyEvent, QMouseEvent
from PyQt6.QtSvg import QSvgRenderer
from PIL import Image
from ultralytics import YOLO
from pynput import mouse


class ScreenDetector:
	def __init__(self, config_path):
		# 解析配置文件
		self._parse_config(config_path)
		
		# 设备检测与模型加载
		self.device = self._determine_device()
		self.model = YOLO(self.model_path).to(self.device)
		
		# 屏幕信息初始化
		self._init_screen_info()
		
		# 控制参数初始化
		self._init_control_params()
		
		# 状态管理
		self.stop_event = threading.Event()
		self.camera_lock = threading.Lock()
		self.target_lock = threading.Lock()
		self.offset_lock = threading.Lock()
		self.button_lock = threading.Lock()
		
		# 推理状态控制
		self.inference_active = False
		self.inference_lock = threading.Lock()
		
		# 初始化相机
		self._init_camera()
		
		# 初始化鼠标监听器
		self._init_mouse_listener()
		
		# 初始化PID控制器
		self._init_pid_controllers()
	
	def _parse_config(self, config_path):
		"""解析并存储配置参数"""
		self.cfg = self._parse_txt_config(config_path)
		
		# 获取应用程序根目录
		if getattr(sys, 'frozen', False):
			base_path = os.path.dirname(sys._MEIPASS)
		else:
			base_path = os.path.dirname(os.path.abspath(__file__))
		
		# 处理模型路径 - 如果是相对路径则转换为绝对路径
		model_path = self.cfg['model_path']
		if not os.path.isabs(model_path):
			model_path = os.path.normpath(os.path.join(base_path, model_path))
		self.model_path = model_path
		
		# 存储常用参数
		self.model_device = self.cfg['model_device']
		self.screen_target_size = int(self.cfg['screen_target_size'])
		self.detection_conf_thres = float(self.cfg['detection_conf_thres'])
		self.detection_iou_thres = float(self.cfg['detection_iou_thres'])
		self.detection_classes = [int(x) for x in self.cfg['detection_classes'].split(',')]
		self.visualization_color = tuple(map(int, self.cfg['visualization_color'].split(',')))
		self.visualization_line_width = int(self.cfg['visualization_line_width'])
		self.visualization_font_scale = float(self.cfg['visualization_font_scale'])
		self.visualization_show_conf = bool(self.cfg['visualization_show_conf'])
		
		# FOV参数
		self.fov_horizontal = float(self.cfg.get('move_fov_horizontal', '90'))
		self.mouse_dpi = int(self.cfg.get('move_mouse_dpi', '400'))
		
		# 目标偏移量参数
		self.target_offset_x_percent = float(self.cfg.get('target_offset_x', '50'))
		self.target_offset_y_percent = 100 - float(self.cfg.get('target_offset_y', '50'))
		
		# PID参数
		self.pid_kp = float(self.cfg.get('pid_kp', '1.0'))
		self.pid_ki = float(self.cfg.get('pid_ki', '0.05'))
		self.pid_kd = float(self.cfg.get('pid_kd', '0.2'))
		
		# 贝塞尔曲线参数
		self.bezier_steps = int(self.cfg.get('bezier_steps', '100'))
		self.bezier_duration = float(self.cfg.get('bezier_duration', '0.1'))
		self.bezier_curve = float(self.cfg.get('bezier_curve', '0.3'))
		
		# 快捷键设置
		self.inference_hotkey = self.cfg.get('inference_hotkey', 'F1')
	
	def update_config(self, config_path):
		"""动态更新配置"""
		try:
			# 重新解析配置文件
			self._parse_config(config_path)
			
			# 更新瞄准按键
			self.aim_button = self._get_aim_button_from_config()
			
			# 更新可以直接修改的参数
			self.detection_conf_thres = float(self.cfg['detection_conf_thres'])
			self.detection_iou_thres = float(self.cfg['detection_iou_thres'])
			self.target_offset_x_percent = float(self.cfg.get('target_offset_x', '50'))
			self.target_offset_y_percent = 100 - float(self.cfg.get('target_offset_y', '50'))
			
			# PID参数更新
			self.pid_kp = float(self.cfg.get('pid_kp', '1.0'))
			self.pid_ki = float(self.cfg.get('pid_ki', '0.05'))
			self.pid_kd = float(self.cfg.get('pid_kd', '0.2'))
			
			# 更新PID控制器
			self.pid_x = PID(self.pid_kp, self.pid_ki, self.pid_kd)
			self.pid_y = PID(self.pid_kp, self.pid_ki, self.pid_kd)
			
			# FOV和DPI更新
			self.fov_horizontal = float(self.cfg.get('move_fov_horizontal', '90'))
			self.mouse_dpi = int(self.cfg.get('move_mouse_dpi', '400'))
			
			# 更新贝塞尔曲线参数
			self.bezier_steps = int(self.cfg.get('bezier_steps', '100'))
			self.bezier_duration = float(self.cfg.get('bezier_duration', '0.1'))
			self.bezier_curve = float(self.cfg.get('bezier_curve', '0.3'))
			
			print("配置已动态更新")
			return True
		except Exception as e:
			print(f"更新配置失败: {str(e)}")
			traceback.print_exc()
			return False
	
	def _parse_txt_config(self, path):
		"""解析TXT格式的配置文件"""
		config = {}
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if not line or line.startswith('#'):
					continue
				if '=' in line:
					key, value = line.split('=', 1)
					config[key.strip()] = value.strip()
		return config
	
	def _init_pid_controllers(self):
		"""初始化PID控制器"""
		# 创建XY方向的PID控制器
		self.pid_x = PID(self.pid_kp, self.pid_ki, self.pid_kd)
		self.pid_y = PID(self.pid_kp, self.pid_ki, self.pid_kd)
	
	def start_inference(self):
		"""启动推理"""
		with self.inference_lock:
			self.inference_active = True
	
	def stop_inference(self):
		"""停止推理"""
		with self.inference_lock:
			self.inference_active = False
	
	def _determine_device(self):
		"""确定运行设备"""
		if self.model_device == 'auto':
			return 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
		return self.model_device
	
	def _init_screen_info(self):
		"""初始化屏幕信息"""
		user32 = ctypes.windll.user32
		self.screen_width, self.screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
		self.screen_center = (self.screen_width // 2, self.screen_height // 2)
		
		# 计算截图区域
		left = (self.screen_width - self.screen_target_size) // 2
		top = (self.screen_height - self.screen_target_size) // 2
		self.region = (
			max(0, int(left)),
			max(0, int(top)),
			min(self.screen_width, int(left + self.screen_target_size)),
			min(self.screen_height, int(top + self.screen_target_size))
		)
	
	def _init_control_params(self):
		"""初始化控制参数"""
		self.previous_target_info = None
		self.closest_target_absolute = None
		self.target_offset = None
		self.aim_button_pressed = False  # 改为通用的瞄准按键状态
	
	def _init_camera(self):
		"""初始化相机"""
		try:
			with self.camera_lock:
				self.camera = dxcam.create(
					output_idx=0,
					output_color="BGR",
					region=self.region
				)
				self.camera.start(target_fps=120, video_mode=True)
		except Exception as e:
			print(f"相机初始化失败: {str(e)}")
			try:
				# 降级模式
				with self.camera_lock:
					self.camera = dxcam.create()
					self.camera.start(target_fps=60, video_mode=True)
			except Exception as fallback_e:
				print(f"降级模式初始化失败: {str(fallback_e)}")
				self.camera = None
	
	def _init_mouse_listener(self):
		"""初始化鼠标监听器"""
		# 读取配置的瞄准按键
		self.aim_button = self._get_aim_button_from_config()
		
		self.mouse_listener = mouse.Listener(
			on_click=self.on_mouse_click
		)
		self.mouse_listener.daemon = True
		self.mouse_listener.start()
	
	def _get_aim_button_from_config(self):
		"""从配置中获取瞄准按键"""
		button_mapping = {
			"鼠标左键": mouse.Button.left,
			"鼠标右键": mouse.Button.right,
			"鼠标中键": mouse.Button.middle,
			"鼠标侧键1": mouse.Button.x1,
			"鼠标侧键2": mouse.Button.x2
		}
		
		aim_button_name = self.cfg.get('aim_button', '鼠标右键')
		return button_mapping.get(aim_button_name, mouse.Button.right)
	
	def on_mouse_click(self, x, y, button, pressed):
		"""处理鼠标点击事件"""
		try:
			# 使用配置的瞄准按键，而不是硬编码的右键
			if button == self.aim_button:
				with self.button_lock:
					self.aim_button_pressed = pressed  # 重命名变量以反映通用性
					# 当瞄准按键释放时重置PID控制器
					if not pressed:
						self.pid_x.reset()
						self.pid_y.reset()
		except Exception as e:
			print(f"鼠标事件处理错误: {str(e)}")
	
	def calculate_fov_movement(self, dx, dy):
		"""初始化FOV"""
		return calculate_fov_movement(
			dx, dy,
			self.screen_width,
			self.screen_height,
			self.fov_horizontal,
			self.mouse_dpi
		)
	
	def move_mouse_to_target(self):
		"""移动鼠标对准目标点"""
		if not self.target_offset:
			return
		
		try:
			# 获取目标点与屏幕中心的偏移量
			with self.offset_lock:
				dx, dy = self.target_offset
			
			# 使用FOV算法将像素偏移转换为鼠标移动量
			move_x, move_y = self.calculate_fov_movement(dx, dy)
			
			# 使用PID计算的移动量
			pid_move_x = self.pid_x.pidPosition(0, -move_x)  # 将dx取反
			pid_move_y = self.pid_y.pidPosition(0, -move_y)  # 将dy取反
			
			# 移动鼠标
			if pid_move_x != 0 or pid_move_y != 0:
				start_mouse_move(int(pid_move_x), int(pid_move_y), self.bezier_steps, self.bezier_duration,
				                 self.bezier_curve)
		except Exception as e:
			print(f"移动鼠标时出错: {str(e)}")
	
	def run(self, frame_queue):
		"""主检测循环"""
		while not self.stop_event.is_set():
			try:
				# 检查推理状态
				with self.inference_lock:
					if not self.inference_active:
						time.sleep(0.01)
						continue
				
				# 截图
				grab_start = time.perf_counter()
				screenshot = self._grab_screenshot()
				grab_time = (time.perf_counter() - grab_start) * 1000  # ms
				
				if screenshot is None:
					time.sleep(0.001)
					continue
				
				# 推理
				inference_start = time.perf_counter()
				results = self._inference(screenshot)
				inference_time = (time.perf_counter() - inference_start) * 1000  # ms
				
				# 处理检测结果
				target_info, closest_target_relative, closest_offset = self._process_detection_results(results)
				
				# 更新目标信息
				self._update_target_info(target_info, closest_offset)
				
				# 移动鼠标
				self._move_mouse_if_needed()
				
				# 可视化处理
				annotated_frame = self._visualize_results(results, closest_target_relative) if frame_queue else None
				
				# 放入队列
				if frame_queue:
					try:
						frame_queue.put(
							(annotated_frame, len(target_info), inference_time, grab_time, target_info),
							timeout=0.01
						)
					except queue.Full:
						pass
			
			except Exception as e:
				print(f"检测循环异常: {str(e)}")
				traceback.print_exc()
				self._reset_camera()
				time.sleep(0.5)
	
	def _grab_screenshot(self):
		"""安全获取截图"""
		with self.camera_lock:
			if self.camera:
				return self.camera.grab()
		return None
	
	def _inference(self, screenshot):
		"""执行模型推理"""
		return self.model.predict(
			screenshot,
			conf=self.detection_conf_thres,
			iou=self.detection_iou_thres,
			classes=self.detection_classes,
			device=self.device,
			verbose=False
		)
	
	def _process_detection_results(self, results):
		"""处理检测结果"""
		target_info = []
		min_distance = float('inf')
		closest_target_relative = None
		closest_target_absolute = None
		closest_offset = None
		
		for box in results[0].boxes:
			# 获取边界框坐标
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			
			# 计算绝对坐标
			x1_abs = x1 + self.region[0]
			y1_abs = y1 + self.region[1]
			x2_abs = x2 + self.region[0]
			y2_abs = y2 + self.region[1]
			
			# 计算边界框尺寸
			width = x2_abs - x1_abs
			height = y2_abs - y1_abs
			
			# 应用偏移百分比计算目标点
			target_x = x1_abs + int(width * (self.target_offset_x_percent / 100))
			target_y = y1_abs + int(height * (self.target_offset_y_percent / 100))
			
			# 计算偏移量
			dx = target_x - self.screen_center[0]
			dy = target_y - self.screen_center[1]
			distance = (dx ** 2 + dy ** 2) ** 0.5
			
			# 更新最近目标
			if distance < min_distance:
				min_distance = distance
				# 计算相对坐标（用于可视化）
				closest_target_relative = (
					x1 + int(width * (self.target_offset_x_percent / 100)),
					y1 + int(height * (self.target_offset_y_percent / 100))
				)
				closest_target_absolute = (target_x, target_y)
				closest_offset = (dx, dy)
			
			# 保存目标信息
			class_id = int(box.cls)
			class_name = self.model.names[class_id]
			target_info.append(f"{class_name}:{x1_abs},{y1_abs},{x2_abs},{y2_abs}")
		
		return target_info, closest_target_relative, closest_offset
	
	def _update_target_info(self, target_info, closest_offset):
		"""更新目标信息"""
		# 检查目标信息是否有变化
		if target_info != self.previous_target_info:
			self.previous_target_info = target_info.copy()
			print(f"{len(target_info)}|{'|'.join(target_info)}")
		
		# 更新目标偏移量
		with self.offset_lock:
			self.target_offset = closest_offset
	
	def _visualize_results(self, results, closest_target):
		"""可视化处理结果"""
		frame = results[0].plot(
			line_width=self.visualization_line_width,
			font_size=self.visualization_font_scale,
			conf=self.visualization_show_conf
		)
		
		# 绘制最近目标
		if closest_target:
			# 绘制目标中心点
			cv2.circle(
				frame,
				(int(closest_target[0]), int(closest_target[1])),
				3, (0, 0, 255), -1
			)
			
			# 计算屏幕中心在截图区域内的相对坐标
			screen_center_x = self.screen_center[0] - self.region[0]
			screen_center_y = self.screen_center[1] - self.region[1]
			
			# 绘制中心到目标的连线
			cv2.line(
				frame,
				(int(screen_center_x), int(screen_center_y)),
				(int(closest_target[0]), int(closest_target[1])),
				(0, 255, 0), 1
			)
		
		return frame
	
	def _move_mouse_if_needed(self):
		"""如果需要则移动鼠标"""
		with self.button_lock:
			if self.aim_button_pressed and self.target_offset:  # 使用通用变量名
				self.move_mouse_to_target()
	
	def _reset_camera(self):
		"""重置相机"""
		print("正在重置相机...")
		try:
			self._init_camera()
		except Exception as e:
			print(f"相机重置失败: {str(e)}")
			traceback.print_exc()
	
	def stop(self):
		"""安全停止检测器"""
		self.stop_event.set()
		self._safe_stop()
		if hasattr(self, 'mouse_listener') and self.mouse_listener.running:  # 改为停止鼠标监听器
			self.mouse_listener.stop()
	
	def _safe_stop(self):
		"""同步释放资源"""
		print("正在安全停止相机...")
		try:
			with self.camera_lock:
				if self.camera:
					self.camera.stop()
					print("相机已停止")
		except Exception as e:
			print(f"停止相机时发生错误: {str(e)}")
		print("屏幕检测器已停止")


class DetectionThread(QThread):
	update_signal = pyqtSignal(object)
	
	def __init__(self, detector, frame_queue):
		super().__init__()
		self.detector = detector
		self.frame_queue = frame_queue
		self.running = True
	
	def run(self):
		self.detector.run(self.frame_queue)
	
	def stop(self):
		self.running = False
		self.detector.stop()


class MouseButtonLineEdit(QLineEdit):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setReadOnly(True)
		self.setPlaceholderText("长按此处然后按下鼠标按键...")
		self.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.current_button = ""
		self.listening = False
		self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)# 禁用上下文菜单
		self.setStyleSheet("""
            QLineEdit {
                background-color: #3C3C40;
                color: #D4D4D4;
                border: 2px solid #0078D7;
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas;
                font-size: 10pt;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
            }
        """)
	
	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self.start_listening()
		super().mousePressEvent(event)
	
	def start_listening(self):
		self.listening = True
		self.current_button = ""
		self.setText("")
		self.setPlaceholderText("正在监听鼠标按键...")
		self.setStyleSheet("""
            QLineEdit {
                background-color: #2D2D30;
                color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 4px;
                padding: 5px;
                font-family: Consolas;
                font-size: 10pt;
            }
        """)
		self.setFocus()
	
	def keyPressEvent(self, event):
		# 阻止键盘事件，只监听鼠标
		if self.listening:
			event.accept()
		else:
			super().keyPressEvent(event)
	
	def mouseReleaseEvent(self, event):
		if not self.listening:
			super().mouseReleaseEvent(event)
			return
		
		# 获取按下的鼠标按钮
		button = event.button()
		button_name = self.get_button_name(button)
		
		if button_name:
			self.current_button = button_name
			self.setText(button_name)
			self.listening = False
			self.setStyleSheet("""
                QLineEdit {
                    background-color: #3C3C40;
                    color: #D4D4D4;
                    border: 1px solid #3F3F46;
                    border-radius: 4px;
                    padding: 5px;
                    font-family: Consolas;
                    font-size: 10pt;
                }
            """)
		
		event.accept()
	
	def get_button_name(self, button):
		button_map = {
			Qt.MouseButton.LeftButton: "鼠标左键",
			Qt.MouseButton.RightButton: "鼠标右键",
			Qt.MouseButton.MiddleButton: "鼠标中键",
			Qt.MouseButton.BackButton: "鼠标侧键1",
			Qt.MouseButton.ForwardButton: "鼠标侧键2",
			Qt.MouseButton.XButton1: "鼠标侧键1",
			Qt.MouseButton.XButton2: "鼠标侧键2"
		}
		return button_map.get(button, "")
	
	def get_button(self):
		"""获取格式化后的鼠标按钮字符串"""
		return self.current_button


class MainWindow(QMainWindow):
	def __init__(self, detector):
		super().__init__()
		self.detector = detector
		self.setWindowTitle("Object Detection Tool-py")
		self.setGeometry(100, 100, 600, 400)
		
		# 设置窗口图标
		try:
			# 获取当前脚本所在目录
			base_path = os.path.dirname(os.path.abspath(__file__))
			icon_path = os.path.join(base_path, 'logo.ico')
			
			# 检查图标文件是否存在
			if os.path.exists(icon_path):
				self.setWindowIcon(QIcon(icon_path))
			else:
				# 如果直接路径找不到，尝试在打包后的资源目录中查找
				if getattr(sys, 'frozen', False):
					base_path = sys._MEIPASS
					icon_path = os.path.join(base_path, 'logo.ico')
					if os.path.exists(icon_path):
						self.setWindowIcon(QIcon(icon_path))
					else:
						print(f"警告: 图标文件未找到 - {icon_path}")
				else:
					print(f"警告: 图标文件未找到 - {icon_path}")
		except Exception as e:
			print(f"设置图标时出错: {str(e)}")
		
		# 添加缺失的属性初始化
		self.visualization_enabled = True
		self.inference_active = False  # 初始推理状态为停止
		
		# 窗口置顶
		self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
		
		# 创建帧队列
		self.frame_queue = queue.Queue(maxsize=3)
		
		# 初始化UI
		self.init_ui()
		
		# 启动检测线程
		self.detection_thread = DetectionThread(self.detector, self.frame_queue)
		self.detection_thread.start()
		
		# 启动UI更新定时器
		self.update_timer = QTimer()
		self.update_timer.timeout.connect(self.update_ui)
		self.update_timer.start(1)  # 每1ms更新一次
	
	def toggle_visualization(self):
		# 实际更新可视化状态属性
		self.visualization_enabled = not self.visualization_enabled
		
		# 更新按钮文本
		if self.visualization_enabled:
			self.toggle_visualization_btn.setText("禁用可视化")
		else:
			self.toggle_visualization_btn.setText("启用可视化")
	
	def toggle_inference(self):
		"""切换推理状态"""
		self.inference_active = not self.inference_active
		if self.inference_active:
			self.toggle_inference_btn.setText("停止推理")
			self.toggle_inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-family: Segoe UI;
                    font-size: 10pt;

                }
            """)
			self.detector.start_inference()
		else:
			self.toggle_inference_btn.setText("开始推理")
			self.toggle_inference_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-family: Segoe UI;
                    font-size: 10pt;

                }
            """)
			self.detector.stop_inference()
	
	def init_ui(self):
		# 主布局
		central_widget = QWidget()
		self.setCentralWidget(central_widget)
		main_layout = QVBoxLayout(central_widget)
		
		# 分割器（左侧图像/目标信息，右侧控制面板）
		splitter = QSplitter(Qt.Orientation.Horizontal)
		main_layout.addWidget(splitter)
		
		# 左侧区域（图像显示和目标信息）
		left_widget = QWidget()
		left_layout = QVBoxLayout(left_widget)
		
		# 图像显示区域
		self.image_label = QLabel()
		self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.image_label.setMinimumSize(320, 320)
		left_layout.addWidget(self.image_label)
		
		# 目标信息区域
		self.target_info_text = QTextEdit()
		self.target_info_text.setReadOnly(True)
		self.target_info_text.setFixedHeight(150)
		self.target_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D30;
                color: #DCDCDC;
                font-family: Consolas;
                font-size: 10pt;
                border: 1px solid #3F3F46;
                border-radius: 4px;
            }
        """)
		left_layout.addWidget(self.target_info_text)
		
		# 右侧控制面板
		right_widget = QWidget()
		right_layout = QVBoxLayout(right_widget)
		right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		
		# 性能信息
		perf_group = QGroupBox("性能信息")
		perf_layout = QVBoxLayout(perf_group)
		
		self.target_count_label = QLabel("目标数量: 0")
		self.inference_time_label = QLabel("推理时间: 0.000s")
		self.grab_time_label = QLabel("截图时间: 0.000s")
		
		for label in [self.target_count_label, self.inference_time_label, self.grab_time_label]:
			label.setStyleSheet("font-family: Consolas; font-size: 10pt;")
			perf_layout.addWidget(label)
		
		right_layout.addWidget(perf_group)
		
		# 系统信息
		sys_group = QGroupBox("系统信息")
		sys_layout = QVBoxLayout(sys_group)
		
		# 获取模型名称（只显示文件名）
		model_name = os.path.basename(self.detector.model_path)
		
		# 获取显示器编号（如果配置中有则显示，否则显示默认值0）
		monitor_index = self.detector.cfg.get('screen_monitor', '0')
		
		self.model_label = QLabel(f"模型: {model_name}")
		self.device_label = QLabel(f"设备: {self.detector.device.upper()}")
		self.monitor_label = QLabel(f"显示器:{monitor_index}")
		self.screen_res_label = QLabel(f"屏幕分辨率: {self.detector.screen_width}x{self.detector.screen_height}")
		self.region_label = QLabel(f"检测区域: {self.detector.region}")
		
		for label in [self.model_label, self.device_label, self.monitor_label, self.screen_res_label,
		              self.region_label]:
			label.setStyleSheet("font-family: Consolas; font-size: 9pt; color: #A0A0A0;")
			sys_layout.addWidget(label)
		
		right_layout.addWidget(sys_group)
		
		# 鼠标状态
		mouse_group = QGroupBox("自瞄状态")
		mouse_layout = QVBoxLayout(mouse_group)
		
		self.mouse_status = QLabel("未瞄准")
		self.mouse_status.setStyleSheet("""
                            QLabel {
                                font-family: Consolas;
                                font-size: 10pt;
                                color: #FF5252;
                            }
                        """)
		mouse_layout.addWidget(self.mouse_status)
		
		right_layout.addWidget(mouse_group)
		
		# 控制按钮
		btn_group = QGroupBox("控制")
		btn_layout = QVBoxLayout(btn_group)
		
		# 添加推理切换按钮
		self.toggle_inference_btn = QPushButton("开始推理")
		self.toggle_inference_btn.clicked.connect(self.toggle_inference)
		self.toggle_inference_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-family: Segoe UI;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #45A049;

            }
            QPushButton:pressed {
                background-color: #3D8B40;
            }
        """)
		btn_layout.addWidget(self.toggle_inference_btn)
		
		self.toggle_visualization_btn = QPushButton("禁用可视化")
		self.toggle_visualization_btn.clicked.connect(self.toggle_visualization)
		
		self.settings_btn = QPushButton("设置")
		self.settings_btn.clicked.connect(self.open_settings)
		
		for btn in [self.toggle_visualization_btn, self.settings_btn]:
			btn.setStyleSheet("""
                        QPushButton {
                            background-color: #0078D7;
                            color: white;
                            border: none;
                            padding: 8px;
                            border-radius: 4px;
                            font-family: Segoe UI;
                            font-size: 10pt;
                        }
                        QPushButton:hover {
                            background-color: #106EBE;
                        }
                        QPushButton:pressed {
                            background-color: #005A9E;
                        }
                    """)
			btn_layout.addWidget(btn)
		
		right_layout.addWidget(btn_group)
		
		# 添加左右区域到分割器
		splitter.addWidget(left_widget)
		splitter.addWidget(right_widget)
		splitter.setSizes([600, 200])
		
		# 设置样式
		self.setStyleSheet("""
            QMainWindow {
                background-color: #252526;
            }
            QGroupBox {
                font-family: Segoe UI;
                font-size: 10pt;
                color: #CCCCCC;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                margin-top: 1ex;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: transparent;
            }
        """)
	
	def open_settings(self):
		settings_dialog = SettingsDialog(self.detector.cfg, self)
		settings_dialog.exec()
	
	def update_ui(self):
		try:
			# 获取最新数据
			latest_data = None
			while not self.frame_queue.empty():
				latest_data = self.frame_queue.get_nowait()
			
			if latest_data:
				# 解包数据
				frame, targets_count, inference_time, grab_time, target_info = latest_data
				
				# 更新性能信息
				self.target_count_label.setText(f"目标数量: {targets_count}")
				self.inference_time_label.setText(f"推理时间: {inference_time / 1000:.3f}s")
				self.grab_time_label.setText(f"截图时间: {grab_time / 1000:.3f}s")
				
				# 更新目标信息
				self.display_target_info(target_info)
				
				# 更新图像显示
				if self.visualization_enabled and frame is not None:
					# 转换图像为Qt格式
					height, width, channel = frame.shape
					bytes_per_line = 3 * width
					q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
					pixmap = QPixmap.fromImage(q_img)
					
					# 等比例缩放
					scaled_pixmap = pixmap.scaled(
						self.image_label.width(),
						self.image_label.height(),
						Qt.AspectRatioMode.KeepAspectRatio,
						Qt.TransformationMode.SmoothTransformation
					)
					self.image_label.setPixmap(scaled_pixmap)
				else:
					# 显示黑色背景
					pixmap = QPixmap(self.image_label.size())
					pixmap.fill(QColor(0, 0, 0))
					self.image_label.setPixmap(pixmap)
			
			# 更新鼠标状态
			self.update_mouse_status()
		
		except Exception as e:
			print(f"更新UI时出错: {str(e)}")
	
	def display_target_info(self, target_info):
		"""在文本框中显示目标信息"""
		if not target_info:
			self.target_info_text.setPlainText("无检测目标")
			return
		
		info_text = "目标类别与坐标:\n"
		for i, data in enumerate(target_info):
			try:
				parts = data.split(":", 1)
				if len(parts) == 2:
					class_name, coords_str = parts
					coords = list(map(int, coords_str.split(',')))
					if len(coords) == 4:
						display_text = f"{class_name}: [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"
					else:
						display_text = f"坐标格式错误: {data}"
				else:
					display_text = f"数据格式错误: {data}"
			except:
				display_text = f"解析错误: {data}"
			
			info_text += f"{display_text}\n"
		
		self.target_info_text.setPlainText(info_text)
	
	def update_mouse_status(self):
		"""更新鼠标瞄准按键状态显示"""
		with self.detector.button_lock:
			if self.detector.aim_button_pressed:  # 使用通用变量名
				self.mouse_status.setText("瞄准中")
				self.mouse_status.setStyleSheet("color: #4CAF50; font-family: Consolas; font-size: 10pt;")
			else:
				self.mouse_status.setText("未瞄准")
				self.mouse_status.setStyleSheet("color: #FF5252; font-family: Consolas; font-size: 10pt;")
	
	def closeEvent(self, event):
		"""安全关闭程序"""
		self.detection_thread.stop()
		self.detection_thread.wait()
		event.accept()


class SettingsDialog(QDialog):
	def __init__(self, config, parent=None):
		super().__init__(parent)
		self.config = config
		# 保存原始配置的副本用于比较
		self.original_config = config.copy()
		
		self.setWindowTitle("设置")
		self.setGeometry(100, 100, 600, 500)
		
		self.init_ui()
	
	def init_ui(self):
		layout = QVBoxLayout()
		self.setLayout(layout)
		
		# 标签页
		tabs = QTabWidget()
		layout.addWidget(tabs)
		
		# 检测设置标签页
		detection_tab = QWidget()
		detection_layout = QVBoxLayout(detection_tab)
		self.create_detection_settings(detection_layout)
		tabs.addTab(detection_tab, "检测")
		
		# 移动设置标签页
		move_tab = QWidget()
		move_layout = QVBoxLayout(move_tab)
		self.create_move_settings(move_layout)
		tabs.addTab(move_tab, "FOV")
		
		# 目标点设置标签页
		target_tab = QWidget()
		target_layout = QVBoxLayout(target_tab)
		self.create_target_settings(target_layout)
		tabs.addTab(target_tab, "目标点")
		
		# PID设置标签页
		pid_tab = QWidget()
		pid_layout = QVBoxLayout(pid_tab)
		self.create_pid_settings(pid_layout)
		tabs.addTab(pid_tab, "PID")
		
		# 贝塞尔曲线设置标签页
		bezier_tab = QWidget()
		bezier_layout = QVBoxLayout(bezier_tab)
		self.create_bezier_settings(bezier_layout)
		tabs.addTab(bezier_tab, "贝塞尔曲线")
		
		# +++ 新增的快捷键设置标签页 +++
		hotkey_tab = QWidget()
		hotkey_layout = QVBoxLayout(hotkey_tab)
		self.create_hotkey_settings(hotkey_layout)
		tabs.addTab(hotkey_tab, "快捷键")
		
		# 按钮区域
		btn_layout = QHBoxLayout()
		layout.addLayout(btn_layout)
		
		save_btn = QPushButton("保存配置")
		save_btn.clicked.connect(self.save_config)
		
		cancel_btn = QPushButton("取消")
		cancel_btn.clicked.connect(self.reject)
		
		for btn in [save_btn, cancel_btn]:
			btn.setStyleSheet("""
                QPushButton {
                    background-color: #0078D7;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-family: Segoe UI;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: #106EBE;
                }
                QPushButton:pressed {
                    background-color: #005A9E;
                }
            """)
			btn_layout.addWidget(btn)
		
		btn_layout.addStretch()
	
	def create_detection_settings(self, layout):
		# 模型选择
		model_group = QGroupBox("模型设置")
		model_layout = QVBoxLayout(model_group)
		
		# 获取基础路径
		if getattr(sys, 'frozen', False):
			base_path = sys._MEIPASS
		else:
			base_path = os.path.dirname(os.path.abspath(__file__))
		
		# 获取模型文件列表
		models_dir = os.path.join(base_path, 'models')
		model_files = []
		if os.path.exists(models_dir):
			model_files = glob.glob(os.path.join(models_dir, '*.pt'))
		
		# 处理模型显示名称
		model_display_names = [os.path.basename(f) for f in model_files] if model_files else ["未找到模型文件"]
		self.model_name_to_path = {os.path.basename(f): f for f in model_files}
		
		# 当前配置的模型处理
		current_model_path = self.config['model_path']
		current_model_name = os.path.basename(current_model_path)
		
		# 确保当前模型在列表中
		if current_model_name not in model_display_names:
			model_display_names.append(current_model_name)
			self.model_name_to_path[current_model_name] = current_model_path
		
		# 模型选择下拉框
		model_layout.addWidget(QLabel("选择模型:"))
		self.model_combo = QComboBox()
		self.model_combo.addItems(model_display_names)
		self.model_combo.setCurrentText(current_model_name)
		model_layout.addWidget(self.model_combo)
		
		# 设备选择
		model_layout.addWidget(QLabel("运行设备:"))
		self.device_combo = QComboBox()
		self.device_combo.addItems(['auto', 'cuda', 'cpu'])
		self.device_combo.setCurrentText(self.config['model_device'])
		model_layout.addWidget(self.device_combo)
		
		layout.addWidget(model_group)
		
		# 检测参数
		param_group = QGroupBox("检测参数")
		param_layout = QVBoxLayout(param_group)
		
		# 置信度阈值
		param_layout.addWidget(QLabel("置信度阈值:"))
		conf_layout = QHBoxLayout()
		
		self.conf_slider = QSlider(Qt.Orientation.Horizontal)
		self.conf_slider.setRange(10, 100)  # 0.1到1.0，步长0.01
		self.conf_slider.setValue(int(float(self.config['detection_conf_thres']) * 100))
		conf_layout.addWidget(self.conf_slider)
		
		self.conf_value = QLabel(f"{float(self.config['detection_conf_thres']):.2f}")
		self.conf_value.setFixedWidth(50)
		conf_layout.addWidget(self.conf_value)
		
		param_layout.addLayout(conf_layout)
		
		# 连接滑块值变化事件
		self.conf_slider.valueChanged.connect(lambda value: self.conf_value.setText(f"{value / 100:.2f}"))
		
		# IOU阈值 - 改为滑动条
		param_layout.addWidget(QLabel("IOU阈值:"))
		iou_layout = QHBoxLayout()
		
		self.iou_slider = QSlider(Qt.Orientation.Horizontal)
		self.iou_slider.setRange(10, 100)  # 0.1到1.0，步长0.01
		self.iou_slider.setValue(int(float(self.config['detection_iou_thres']) * 100))
		iou_layout.addWidget(self.iou_slider)
		
		self.iou_value = QLabel(f"{float(self.config['detection_iou_thres']):.2f}")
		self.iou_value.setFixedWidth(50)
		iou_layout.addWidget(self.iou_value)
		
		param_layout.addLayout(iou_layout)
		
		# 连接滑块值变化事件
		self.iou_slider.valueChanged.connect(lambda value: self.iou_value.setText(f"{value / 100:.2f}"))
		
		# 检测类别
		param_layout.addWidget(QLabel("检测类别 (逗号分隔):"))
		self.classes_edit = QLineEdit()
		self.classes_edit.setText(self.config['detection_classes'])
		param_layout.addWidget(self.classes_edit)
		
		layout.addWidget(param_group)
		
		# 屏幕设置
		screen_group = QGroupBox("屏幕设置")
		screen_layout = QVBoxLayout(screen_group)
		
		# 显示器编号
		screen_layout.addWidget(QLabel("显示器编号:"))
		self.monitor_spin = QSpinBox()
		self.monitor_spin.setRange(0, 3)  # 假设最多支持4个显示器
		self.monitor_spin.setValue(int(self.config.get('screen_monitor', '0')))
		screen_layout.addWidget(self.monitor_spin)
		
		# 屏幕区域大小
		screen_layout.addWidget(QLabel("截屏尺寸:"))
		self.screen_size_spin = QSpinBox()
		self.screen_size_spin.setRange(100, 2000)
		self.screen_size_spin.setValue(int(self.config['screen_target_size']))
		screen_layout.addWidget(self.screen_size_spin)
		
		layout.addWidget(screen_group)
		
		layout.addStretch()
	
	def create_move_settings(self, layout):
		group = QGroupBox("鼠标移动参数")
		group_layout = QVBoxLayout(group)
		
		# FOV设置
		group_layout.addWidget(QLabel("横向FOV(度):"))
		self.fov_spin = QDoubleSpinBox()
		self.fov_spin.setRange(1, 179)
		self.fov_spin.setValue(float(self.config.get('move_fov_horizontal', '90')))
		group_layout.addWidget(self.fov_spin)
		
		# 鼠标DPI
		group_layout.addWidget(QLabel("鼠标DPI:"))
		self.dpi_spin = QSpinBox()
		self.dpi_spin.setRange(100, 20000)
		self.dpi_spin.setValue(int(self.config.get('move_mouse_dpi', '400')))
		group_layout.addWidget(self.dpi_spin)
		
		layout.addWidget(group)
		layout.addStretch()
	
	def create_target_settings(self, layout):
		group = QGroupBox("目标点偏移")
		group_layout = QVBoxLayout(group)
		
		# X轴偏移 - 添加百分比显示
		group_layout.addWidget(QLabel("X轴偏移:"))
		x_layout = QHBoxLayout()
		
		self.x_offset_slider = QSlider(Qt.Orientation.Horizontal)
		self.x_offset_slider.setRange(0, 100)
		self.x_offset_slider.setValue(int(float(self.config.get('target_offset_x', '50'))))
		x_layout.addWidget(self.x_offset_slider)
		
		self.x_offset_value = QLabel(f"{int(float(self.config.get('target_offset_x', '50')))}%")
		self.x_offset_value.setFixedWidth(50)
		x_layout.addWidget(self.x_offset_value)
		
		group_layout.addLayout(x_layout)
		
		# 连接滑块值变化事件
		self.x_offset_slider.valueChanged.connect(lambda value: self.x_offset_value.setText(f"{value}%"))
		
		# Y轴偏移 - 添加百分比显示
		group_layout.addWidget(QLabel("Y轴偏移:"))
		y_layout = QHBoxLayout()
		
		self.y_offset_slider = QSlider(Qt.Orientation.Horizontal)
		self.y_offset_slider.setRange(0, 100)
		self.y_offset_slider.setValue(int(float(self.config.get('target_offset_y', '50'))))
		y_layout.addWidget(self.y_offset_slider)
		
		self.y_offset_value = QLabel(f"{int(float(self.config.get('target_offset_y', '50')))}%")
		self.y_offset_value.setFixedWidth(50)
		y_layout.addWidget(self.y_offset_value)
		
		group_layout.addLayout(y_layout)
		
		# 连接滑块值变化事件
		self.y_offset_slider.valueChanged.connect(lambda value: self.y_offset_value.setText(f"{value}%"))
		
		# 说明
		info_label = QLabel("(0% = 左上角, 50% = 中心, 100% = 右下角)")
		info_label.setStyleSheet("font-size: 9pt; color: #888888;")
		group_layout.addWidget(info_label)
		
		layout.addWidget(group)
		
		layout.addStretch()
	
	def create_pid_settings(self, layout):
		group = QGroupBox("PID参数")
		group_layout = QVBoxLayout(group)
		
		# Kp参数
		group_layout.addWidget(QLabel("比例增益(Kp):"))
		kp_layout = QHBoxLayout()
		
		self.kp_slider = QSlider(Qt.Orientation.Horizontal)
		self.kp_slider.setRange(1, 1000)  # 0.01到10.0，步长0.01
		self.kp_slider.setValue(int(float(self.config.get('pid_kp', '1.0')) * 100))
		kp_layout.addWidget(self.kp_slider)
		
		self.kp_value = QLabel(f"{float(self.config.get('pid_kp', '1.0')):.2f}")
		self.kp_value.setFixedWidth(50)
		kp_layout.addWidget(self.kp_value)
		
		group_layout.addLayout(kp_layout)
		
		# 连接滑块值变化事件
		self.kp_slider.valueChanged.connect(lambda value: self.kp_value.setText(f"{value / 100:.2f}"))
		
		# Ki参数
		group_layout.addWidget(QLabel("积分增益(Ki):"))
		ki_layout = QHBoxLayout()
		
		self.ki_slider = QSlider(Qt.Orientation.Horizontal)
		self.ki_slider.setRange(0, 100)  # 0.0000到0.1000，步长0.001
		self.ki_slider.setValue(int(float(self.config.get('pid_ki', '0.05')) * 10000))
		ki_layout.addWidget(self.ki_slider)
		
		self.ki_value = QLabel(f"{float(self.config.get('pid_ki', '0.05')):.4f}")
		self.ki_value.setFixedWidth(50)
		ki_layout.addWidget(self.ki_value)
		
		group_layout.addLayout(ki_layout)
		
		# 连接滑块值变化事件
		self.ki_slider.valueChanged.connect(lambda value: self.ki_value.setText(f"{value / 10000:.4f}"))
		
		# Kd参数
		group_layout.addWidget(QLabel("微分增益(Kd):"))
		kd_layout = QHBoxLayout()
		
		self.kd_slider = QSlider(Qt.Orientation.Horizontal)
		self.kd_slider.setRange(0, 5000)  # 0.000到5.000，步长0.001
		self.kd_slider.setValue(int(float(self.config.get('pid_kd', '0.2')) * 1000))
		kd_layout.addWidget(self.kd_slider)
		
		self.kd_value = QLabel(f"{float(self.config.get('pid_kd', '0.2')):.3f}")
		self.kd_value.setFixedWidth(50)
		kd_layout.addWidget(self.kd_value)
		
		group_layout.addLayout(kd_layout)
		
		# 连接滑块值变化事件
		self.kd_slider.valueChanged.connect(lambda value: self.kd_value.setText(f"{value / 1000:.3f}"))
		
		# 说明
		info_text = "建议调整顺序: Kp → Kd → Ki\n\n" \
		            "先调整Kp至响应迅速但不过冲\n" \
		            "再增加Kd抑制震荡\n" \
		            "最后微调Ki消除剩余误差"
		info_label = QLabel(info_text)
		info_label.setStyleSheet("font-size: 9pt; color: #888888;")
		group_layout.addWidget(info_label)
		
		layout.addWidget(group)
		layout.addStretch()
	
	# 创建贝塞尔曲线设置
	def create_bezier_settings(self, layout):
		
		group = QGroupBox("贝塞尔曲线参数")
		group_layout = QVBoxLayout(group)
		# 步数设置
		group_layout.addWidget(QLabel("步数 (1-500):"))
		steps_layout = QHBoxLayout()
		
		self.steps_slider = QSlider(Qt.Orientation.Horizontal)
		self.steps_slider.setRange(1, 500)
		self.steps_slider.setValue(int(self.config.get('bezier_steps', 100)))
		steps_layout.addWidget(self.steps_slider)
		
		self.steps_value = QLabel(str(self.config.get('bezier_steps', 100)))
		self.steps_value.setFixedWidth(50)
		steps_layout.addWidget(self.steps_value)
		
		group_layout.addLayout(steps_layout)
		
		# 连接滑块值变化事件
		self.steps_slider.valueChanged.connect(lambda value: self.steps_value.setText(str(value)))
		
		# 总移动时间设置 (秒)
		group_layout.addWidget(QLabel("总移动时间 (秒):"))
		duration_layout = QHBoxLayout()
		
		self.duration_slider = QSlider(Qt.Orientation.Horizontal)
		self.duration_slider.setRange(0, 100)  # 0.01到1.0，步长0.01
		self.duration_slider.setValue(int(float(self.config.get('bezier_duration', 0.1)) * 100))
		duration_layout.addWidget(self.duration_slider)
		
		self.duration_value = QLabel(f"{float(self.config.get('bezier_duration', 0.1)):.2f}")
		self.duration_value.setFixedWidth(50)
		duration_layout.addWidget(self.duration_value)
		
		group_layout.addLayout(duration_layout)
		
		# 连接滑块值变化事件
		self.duration_slider.valueChanged.connect(lambda value: self.duration_value.setText(f"{value / 100:.2f}"))
		
		# 控制点偏移幅度
		group_layout.addWidget(QLabel("控制点偏移幅度 (0-1):"))
		curve_layout = QHBoxLayout()
		
		self.curve_slider = QSlider(Qt.Orientation.Horizontal)
		self.curve_slider.setRange(0, 100)  # 0.00到1.00，步长0.01
		self.curve_slider.setValue(int(float(self.config.get('bezier_curve', 0.3)) * 100))
		curve_layout.addWidget(self.curve_slider)
		
		self.curve_value = QLabel(f"{float(self.config.get('bezier_curve', 0.3)):.2f}")
		self.curve_value.setFixedWidth(50)
		curve_layout.addWidget(self.curve_value)
		
		group_layout.addLayout(curve_layout)
		
		# 连接滑块值变化事件
		self.curve_slider.valueChanged.connect(lambda value: self.curve_value.setText(f"{value / 100:.2f}"))
		
		# 说明
		info_text = "贝塞尔曲线参数说明:\n\n" \
		            "• 步数: 鼠标移动的细分步数，值越大移动越平滑\n" \
		            "• 总移动时间: 鼠标移动的总时间，值越小移动越快\n" \
		            "• 控制点偏移幅度: 控制贝塞尔曲线的弯曲程度，0为直线，1为最大弯曲"
		info_label = QLabel(info_text)
		info_label.setStyleSheet("font-size: 9pt; color: #888888;")
		group_layout.addWidget(info_label)
		
		layout.addWidget(group)
		layout.addStretch()
	
	def create_hotkey_settings(self, layout):
		"""创建快捷键设置"""
		group = QGroupBox("鼠标按键设置")
		group_layout = QVBoxLayout(group)
		
		# 鼠标按键设置
		group_layout.addWidget(QLabel("瞄准按键:"))
		
		# 使用自定义的鼠标按键输入框
		self.mouse_button_input = MouseButtonLineEdit()
		
		# 设置当前鼠标按键
		current_mouse_button = self.config.get('aim_button', '鼠标右键')
		if current_mouse_button:
			self.mouse_button_input.setText(current_mouse_button)
			self.mouse_button_input.current_button = current_mouse_button
		
		group_layout.addWidget(self.mouse_button_input)
		
		# 清除按钮
		clear_btn = QPushButton("清除鼠标按键")
		clear_btn.clicked.connect(self.clear_mouse_button)
		clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5252;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 4px;
                font-family: Segoe UI;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
        """)
		group_layout.addWidget(clear_btn)
		
		# 鼠标按键说明
		info_text = "鼠标按键设置说明:\n\n" \
		            "• 长按输入框后按下想要的鼠标按键\n" \
		            "• 支持所有标准鼠标按键\n" \
		            "• 包括左右键、中键和侧键\n\n" \
		            "常用推荐:\n" \
		            "• 鼠标右键: 最常用，与游戏右键瞄准一致\n" \
		
		info_label = QLabel(info_text)
		info_label.setStyleSheet("font-size: 9pt; color: #888888;")
		info_label.setWordWrap(True)
		group_layout.addWidget(info_label)
		
		layout.addWidget(group)
		layout.addStretch()
	
	def clear_mouse_button(self):
		"""清除鼠标按键"""
		self.mouse_button_input.setText("")
		self.mouse_button_input.current_button = ""
		self.mouse_button_input.setPlaceholderText("长按此处然后按下鼠标按键...")
	
	def save_config(self):
		try:
			# 获取应用程序根目录
			if getattr(sys, 'frozen', False):
				base_path = os.path.dirname(sys._MEIPASS)
			else:
				base_path = os.path.dirname(os.path.abspath(__file__))
			
			# 保存配置到字典
			model_name = self.model_combo.currentText()
			model_path = self.model_name_to_path.get(model_name, model_name)
			
			# 保存鼠标按键设置
			mouse_button = self.mouse_button_input.get_button()
			if mouse_button:
				self.config['aim_button'] = mouse_button
			else:
				# 如果没有设置鼠标按键，使用默认值
				self.config['aim_button'] = '鼠标右键'
			
			self.config['model_device'] = self.device_combo.currentText()
			self.config['screen_monitor'] = str(self.monitor_spin.value())
			self.config['screen_target_size'] = str(self.screen_size_spin.value())
			
			# 检测参数
			self.config['detection_conf_thres'] = str(self.conf_slider.value() / 100)
			self.config['detection_iou_thres'] = str(self.iou_slider.value() / 100)
			self.config['detection_classes'] = self.classes_edit.text()
			
			# 移动设置
			self.config['move_fov_horizontal'] = str(self.fov_spin.value())
			self.config['move_mouse_dpi'] = str(self.dpi_spin.value())
			
			# 目标点偏移设置
			self.config['target_offset_x'] = str(self.x_offset_slider.value())
			self.config['target_offset_y'] = str(self.y_offset_slider.value())
			
			# PID设置
			self.config['pid_kp'] = str(self.kp_slider.value() / 100)
			self.config['pid_ki'] = str(self.ki_slider.value() / 10000)
			self.config['pid_kd'] = str(self.kd_slider.value() / 1000)
			
			# 贝塞尔曲线设置
			self.config['bezier_steps'] = str(self.steps_slider.value())
			self.config['bezier_duration'] = str(self.duration_slider.value() / 100)
			self.config['bezier_curve'] = str(self.curve_slider.value() / 100)
			
			# 保存鼠标按键设置
			mouse_button = self.mouse_button_input.get_button()
			if mouse_button:
				self.config['aim_button'] = mouse_button
			else:
				# 如果没有设置鼠标按键，使用默认值
				self.config['aim_button'] = '鼠标右键'
			
			# 保存为TXT格式
			with open('detection_config.txt', 'w', encoding='utf-8') as f:
				for key, value in self.config.items():
					f.write(f"{key} = {value}\n")
			
			# 检查需要重启的参数是否被修改
			restart_required = False
			restart_params = []
			
			# 比较模型路径是否变化
			if self.config['model_path'] != self.original_config.get('model_path', ''):
				restart_required = True
				restart_params.append("模型路径")
			
			# 比较设备类型是否变化
			if self.config['model_device'] != self.original_config.get('model_device', ''):
				restart_required = True
				restart_params.append("设备类型")
			
			# 比较屏幕区域大小是否变化
			if self.config['screen_target_size'] != self.original_config.get('screen_target_size', ''):
				restart_required = True
				restart_params.append("屏幕区域大小")
			
			# 比较检测类别是否变化
			if self.config['detection_classes'] != self.original_config.get('detection_classes', ''):
				restart_required = True
				restart_params.append("检测类别")
			
			# 动态更新检测器配置
			if self.parent() and hasattr(self.parent(), 'detector'):
				success = self.parent().detector.update_config('detection_config.txt')
				if success:
					if restart_required:
						# 需要重启的参数已修改
						param_list = "、".join(restart_params)
						QMessageBox.information(
							self,
							"配置已保存",
							f"配置已保存！以下参数需要重启才能生效:\n{param_list}\n\n"
							"其他参数已实时更新。"
						)
					else:
						# 所有参数都已实时更新
						QMessageBox.information(self, "成功", "配置已实时更新生效！")
				else:
					QMessageBox.warning(self, "部分更新", "配置更新失败，请查看日志")
			else:
				QMessageBox.information(self, "成功", "配置已保存！部分参数需重启生效")
			
			self.accept()
		except Exception as e:
			QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")


if __name__ == "__main__":
	detector = ScreenDetector('detection_config.txt')
	print(f"\nDXcam检测器初始化完成 | 设备: {detector.device.upper()}")
	
	app = QApplication(sys.argv)
	
	# 设置全局样式
	app.setStyle("Fusion")
	app.setStyleSheet("""
        QWidget {
            background-color: #252526;
            color: #D4D4D4;
            selection-background-color: #0078D7;
            selection-color: white;
        }

        QPushButton {
            background-color: #0078D7;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
        }

        QPushButton:hover {
            background-color: #106EBE;
        }

        QPushButton:pressed {
            background-color: #005A9E;
        }

        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QSlider {
            background-color: #3C3C40;
            color: #D4D4D4;
            border: 1px solid #3F3F46;
            border-radius: 4px;
            padding: 3px;
        }

        QComboBox:editable {
            background-color: #3C3C40;
        }

        QComboBox QAbstractItemView {
            background-color: #2D2D30;
            color: #D4D4D4;
            selection-background-color: #0078D7;
            selection-color: white;
        }

        QLabel {
            color: #D4D4D4;
        }

        QTabWidget::pane {
            border: 1px solid #3F3F46;
            background: #252526;
        }

        QTabBar::tab {
            background: #1E1E1E;
            color: #A0A0A0;
            padding: 8px 12px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }

        QTabBar::tab:selected {
            background: #252526;
            color: #FFFFFF;
            border-bottom: 2px solid #0078D7;
        }

        QTabBar::tab:hover {
            background: #2D2D30;
        }

        QGroupBox {
            background-color: #252526;
            border: 1px solid #3F3F46;
            border-radius: 4px;
            margin-top: 1ex;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            background-color: transparent;
            color: #CCCCCC;
        }
    """)
	
	window = MainWindow(detector)
	window.show()
	sys.exit(app.exec())
