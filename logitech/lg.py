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
def mouse_xy(x, y):
    if gmok:
        gm.moveR(int(x), int(y))
