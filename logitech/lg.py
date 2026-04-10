import os
from ctypes import CDLL
import math
import time
# 获取当前脚本的绝对路径并定位DLL
script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, 'logitech.driver.dll')

try:
    gm = CDLL(dll_path)  # 使用绝对路径加载
    gmok = gm.device_open() == 1
    if not gmok:
        print('未安装ghub或者lgs驱动!!!')
    else:
        print('初始化成功!')
except FileNotFoundError:
    print(f'缺少文件: {dll_path}')
except OSError as e:
    print(f'DLL加载失败: {e}\n可能原因: 依赖缺失/架构不匹配')


#按下鼠标按键
def press_mouse_button(button):
    if gmok:
        gm.mouse_down(button)

#松开鼠标按键
def release_mouse_button(button):
    if gmok:
        gm.mouse_up(button)

#点击鼠标
def click_mouse_button(button):
    press_mouse_button(button)
    release_mouse_button(button)

#按下键盘按键
def press_key(code):
    if gmok:
        gm.key_down(code)

#松开键盘按键
def release_key(code):
    if gmok:
        gm.key_up(code)

#点击键盘按键
def click_key(code):
    press_key(code)
    release_key(code)
"""
#鼠标相对移动
def mouse_xy(x, y):
    if gmok:
        gm.moveR(int(x), int(y))

"""
import time
import random
import threading


def mouse_xy(x, y):
    if gmok:
        gm.moveR(int(x), int(y))

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
