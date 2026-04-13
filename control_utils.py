import numpy as np
import time
import random
import threading
from logitech.lg import mouse_xy


class CoordinateSmoother:
	# 平滑坐标的函数，使用简单移动平均
	def __init__(self, window_size=5):
		self.window_size = window_size
		self.x_buffer = []
		self.y_buffer = []
	
	def smooth_coordinate(self, x, y):
		self.x_buffer.append(x)
		self.y_buffer.append(y)
		
		if len(self.x_buffer) > self.window_size:
			self.x_buffer.pop(0)
			self.y_buffer.pop(0)
		
		smoothed_x = int(np.mean(self.x_buffer))
		smoothed_y = int(np.mean(self.y_buffer))
		
		return smoothed_x, smoothed_y
	

class PID:
	"""PID控制器类"""
	
	def __init__(self, P=0.2, I=0.01, D=0.1):
		self.kp, self.ki, self.kd = P, I, D
		self.uPrevious, self.uCurent = 0, 0
		self.setValue, self.lastErr, self.errSum = 0, 0, 0
		self.errSumLimit = 10
	
	def pidPosition(self, setValue, curValue):
		"""计算PID控制输出"""
		err = setValue - curValue
		dErr = err - self.lastErr
		self.errSum += err
		outPID = self.kp * err + self.ki * self.errSum + self.kd * dErr
		self.lastErr = err
		return outPID
	
	def reset(self):
		"""重置控制器状态"""
		self.errSum = 0.0
		self.lastErr = 0.0


def calculate_fov_movement(dx, dy, screen_width, screen_height, fov_horizontal, mouse_dpi):
	"""基于FOV算法计算鼠标移动量
	
	参数:
		dx (float): X轴像素偏移量
		dy (float): Y轴像素偏移量
		screen_width (int): 屏幕宽度
		screen_height (int): 屏幕高度
		fov_horizontal (float): 水平视场角(度)
		mouse_dpi (int): 鼠标DPI
		
	返回:
		tuple: (move_x, move_y) 鼠标移动量
	"""
	# 计算屏幕对角线长度
	screen_diagonal = (screen_width ** 2 + screen_height ** 2) ** 0.5
	
	# 计算垂直FOV
	aspect_ratio = screen_width / screen_height
	fov_vertical = fov_horizontal / aspect_ratio
	
	# 计算每像素对应角度
	angle_per_pixel_x = fov_horizontal / screen_width
	angle_per_pixel_y = fov_vertical / screen_height
	
	# 计算角度偏移
	angle_offset_x = dx * angle_per_pixel_x
	angle_offset_y = dy * angle_per_pixel_y
	
	# 转换为鼠标移动量
	move_x = (angle_offset_x / 360) * mouse_dpi
	move_y = (angle_offset_y / 360) * mouse_dpi
	
	return move_x, move_y


class MouseMoverThread(threading.Thread):
	"""在独立线程中执行贝塞尔曲线鼠标移动的类"""

	def __init__(self, target_x, target_y, steps, duration=0.1, control_point_offset=0.3):
		"""
		参数:
		target_x, target_y - 目标相对位移（像素）
		steps - 移动步数
		duration - 总移动时间（秒）
		control_point_offset - 控制点偏移幅度
		"""
		super().__init__()
		self.target_x = target_x
		self.target_y = target_y
		self.steps = steps
		self.duration = duration
		self.control_point_offset = control_point_offset
		self.daemon = True  # 设置为守护线程，主线程退出时自动结束

	def run(self):
		"""线程执行的核心方法"""
		# 1. 生成带随机控制点的贝塞尔路径
		cp_x = self.target_x * 0.5 * (1 + random.uniform(-self.control_point_offset, self.control_point_offset))
		cp_y = self.target_y * 0.5 * (1 + random.uniform(-self.control_point_offset, self.control_point_offset))

		# 2. 计算贝塞尔曲线点（二阶公式）
		points = []
		for i in range(self.steps + 1):
			t = i / self.steps

			# 二阶贝塞尔公式: B(t)=(1-t)²P0 + 2t(1-t)P1 + t²P2
			x = (1 - t) ** 2 * 0 + 2 * (1 - t) * t * cp_x + t ** 2 * self.target_x
			y = (1 - t) ** 2 * 0 + 2 * (1 - t) * t * cp_y + t ** 2 * self.target_y
			points.append((x, y))

		# 3. 转换为相对位移序列（带误差修正）
		moves = []
		cum_error_x, cum_error_y = 0.0, 0.0
		for i in range(1, len(points)):
			dx = points[i][0] - points[i - 1][0] + cum_error_x
			dy = points[i][1] - points[i - 1][1] + cum_error_y

			# 四舍五入取整并累积误差
			move_x = round(dx)
			move_y = round(dy)
			cum_error_x = dx - move_x
			cum_error_y = dy - move_y
			moves.append((int(move_x), int(move_y)))

		# 4. 速度控制核心算法
		start_time = time.monotonic()  # 高精度计时器
		accumulated_delay = 0.0  # 累积延迟补偿值

		for idx, (dx, dy) in enumerate(moves):
			if dx != 0 or dy != 0:
				mouse_xy(dx, dy)  # 调用鼠标移动函数

			# 计算当前步的理论结束时间
			expected_end = start_time + (idx + 1) * (self.duration / self.steps)
			current_time = time.monotonic()

			# 动态延迟调整（含误差补偿）
			remaining_time = max(0, expected_end - current_time - accumulated_delay)
			actual_delay = remaining_time * random.uniform(0.95, 1.05)  # ±5%随机波动

			if actual_delay > 0:
				time.sleep(actual_delay)

			# 更新延迟补偿（防止误差累积）
			elapsed = time.monotonic() - current_time
			accumulated_delay += elapsed - actual_delay

		print("鼠标移动完成!")


def start_mouse_move(target_x, target_y, steps, duration, control_point_offset=0.3):
	"""启动鼠标移动线程的便捷函数"""
	mover = MouseMoverThread(target_x, target_y, steps, duration, control_point_offset)
	mover.start()
	return mover