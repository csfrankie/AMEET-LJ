import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import sys
import os
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from datetime import datetime
import json
import threading
import math
from PIL import Image, ImageTk
import time

class KalmanFilter3D:
    """3D卡爾曼濾波器用於位置和速度追蹤"""
    def __init__(self, dt=0.033):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=9, dim_z=3)
        
        # 狀態轉移矩陣 [x, y, z, vx, vy, vz, ax, ay, az]
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 測量矩陣
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])
        
        # 過程噪聲協方差
        self.kf.Q = np.eye(9) * 0.01
        
        # 測量噪聲協方差
        self.kf.R = np.eye(3) * 1
        
        # 初始狀態協方差
        self.kf.P = np.eye(9) * 100
        
    def predict(self):
        self.kf.predict()
        return self.kf.x[:3]
    
    def update(self, measurement):
        self.kf.update(measurement)
        return self.kf.x[:3]
    
    def get_velocity(self):
        return self.kf.x[3:6]
    
    def get_acceleration(self):
        return self.kf.x[6:9]
    
    def set_dt(self, dt):
        """動態更新時間步長"""
        self.dt = dt
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

class ZoomWindow:
    """改進的放大窗口類，跟隨鼠標位置放大"""
    def __init__(self, parent, main_window, point_index, callback):
        self.parent = parent
        self.main_window = main_window
        self.image = main_window.current_frame.copy()
        self.point_index = point_index
        self.callback = callback
        self.zoom_factor = 3.0  # 初始放大倍數
        
        # 創建窗口
        self.window = tk.Toplevel(parent)
        self.window.title(f"Zoom Window - Point {point_index + 1}")
        self.window.geometry("800x600")
        self.window.configure(bg='black')
        
        # 創建Canvas
        self.canvas = tk.Canvas(self.window, bg='black', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加說明
        self.create_instructions()
        
        # 初始化變量
        self.original_image = self.image.copy()
        self.display_image = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.selected_point = None
        self.click_processed = False  # 防止重複點擊
        
        # 設置初始鼠標位置為圖像中心
        h, w = self.image.shape[:2]
        self.mouse_x = w // 2
        self.mouse_y = h // 2
        
        # 綁定事件
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 右鍵取消
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux上滾
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux下滾
        self.window.bind("<Return>", self.on_confirm)
        self.window.bind("<Escape>", self.on_cancel)
        self.window.bind("<KeyPress>", self.on_key_press)
        
        # 顯示初始圖像
        self.display_zoom_image()
        
        # 設置窗口置頂
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # 設置焦點，確保接收鍵盤事件
        self.window.focus_set()
        self.canvas.focus_set()
        
        # 保存窗口引用
        self.main_window.current_zoom_window = self
    
    def create_instructions(self):
        """創建說明文字"""
        instructions_frame = tk.Frame(self.window, bg='black')
        instructions_frame.place(x=10, y=10)
        
        instructions = [
            f"Point {self.point_index + 1} - Use crosshair to mark accurately",
            "Left Click: Mark point",
            "Right Click: Cancel point",
            "Mouse Wheel: Zoom in/out",
            "Arrow Keys: Move crosshair",
            "Enter: Confirm, Esc: Cancel"
        ]
        
        for i, text in enumerate(instructions):
            color = "yellow" if i == 0 else "white"
            label = tk.Label(instructions_frame, text=text, bg="black", 
                           fg=color, font=('Arial', 10))
            label.pack(anchor=tk.W)
    
    def display_zoom_image(self):
        """顯示放大後的圖像"""
        if self.image is None:
            return
        
        h, w = self.image.shape[:2]
        
        # 計算放大區域大小
        zoom_size = 200  # 放大區域大小（像素）
        half_size = int(zoom_size / (2 * self.zoom_factor))
        
        # 計算顯示區域
        start_x = max(0, self.mouse_x - half_size)
        end_x = min(w, self.mouse_x + half_size)
        start_y = max(0, self.mouse_y - half_size)
        end_y = min(h, self.mouse_y + half_size)
        
        # 如果靠近邊緣，調整位置
        if end_x - start_x < zoom_size / self.zoom_factor:
            if start_x == 0:
                end_x = min(w, start_x + int(zoom_size / self.zoom_factor))
            else:
                start_x = max(0, end_x - int(zoom_size / self.zoom_factor))
        
        if end_y - start_y < zoom_size / self.zoom_factor:
            if start_y == 0:
                end_y = min(h, start_y + int(zoom_size / self.zoom_factor))
            else:
                start_y = max(0, end_y - int(zoom_size / self.zoom_factor))
        
        # 提取放大區域
        zoom_region = self.image[start_y:end_y, start_x:end_x]
        
        if zoom_region.size == 0:
            return
        
        # 放大圖像
        new_width = int(zoom_region.shape[1] * self.zoom_factor)
        new_height = int(zoom_region.shape[0] * self.zoom_factor)
        
        # 使用高質量插值
        zoomed = cv2.resize(zoom_region, (new_width, new_height), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 轉換為RGB
        rgb_zoomed = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
        
        # 轉換為PIL Image
        pil_image = Image.fromarray(rgb_zoomed)
        
        # 創建PhotoImage
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # 顯示圖像
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.display_image, anchor=tk.CENTER)
        
        # 繪製十字準星（始終在中心）
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        
        # 十字準星顏色
        crosshair_color = "red" if self.selected_point is None else "green"
        
        # 垂直線
        self.canvas.create_line(canvas_w//2, 0, canvas_w//2, canvas_h, 
                               fill=crosshair_color, width=2)
        # 水平線
        self.canvas.create_line(0, canvas_h//2, canvas_w, canvas_h//2, 
                               fill=crosshair_color, width=2)
        
        # 十字準星圓圈
        self.canvas.create_oval(canvas_w//2-10, canvas_h//2-10,
                               canvas_w//2+10, canvas_h//2+10,
                               outline=crosshair_color, width=2)
        
        # 顯示坐標信息
        info_text = f"Point {self.point_index + 1}\n"
        info_text += f"Image: ({self.mouse_x}, {self.mouse_y})\n"
        info_text += f"Zoom: {self.zoom_factor:.1f}x\n"
        
        if self.selected_point:
            info_text += "✓ Point marked\n"
            info_text += f"Selected: ({self.selected_point[0]}, {self.selected_point[1]})"
        
        self.canvas.create_text(100, 50, text=info_text, fill="white", 
                               font=('Arial', 12, 'bold'), anchor=tk.W)
        
        # 顯示當前點說明
        point_names = [
            "Takeoff board position",
            "5 meters from takeoff board",
            "10 meters from takeoff board",
            "1m above takeoff board",
            "1m above 5m point",
            "1m above 10m point"
        ]
        
        if self.point_index < len(point_names):
            point_desc = point_names[self.point_index]
            self.canvas.create_text(100, 150, text=point_desc, fill="cyan", 
                                   font=('Arial', 11), anchor=tk.W)
    
    def on_mouse_move(self, event):
        """鼠標移動事件"""
        # 計算在放大圖像中的相對位置
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 600
        
        # 計算相對偏移
        rel_x = (event.x - canvas_w//2) / (canvas_w//2)
        rel_y = (event.y - canvas_h//2) / (canvas_h//2)
        
        # 計算移動速度（基於縮放）
        move_speed = 20 / self.zoom_factor
        
        # 更新鼠標位置
        h, w = self.image.shape[:2]
        self.mouse_x = int(max(0, min(w-1, self.mouse_x + rel_x * move_speed)))
        self.mouse_y = int(max(0, min(h-1, self.mouse_y + rel_y * move_speed)))
        
        # 更新顯示
        self.display_zoom_image()
    
    def on_click(self, event):
        """點擊事件"""
        # 防止重複處理
        if self.click_processed:
            return
            
        self.click_processed = True
        
        try:
            # 計算在原始圖像中的位置
            canvas_w = self.canvas.winfo_width() or 800
            canvas_h = self.canvas.winfo_height() or 600
            
            # 計算相對位置
            rel_x = (event.x - canvas_w//2) / self.zoom_factor
            rel_y = (event.y - canvas_h//2) / self.zoom_factor
            
            # 計算原始圖像位置（中心點對應當前mouse_x, mouse_y）
            original_x = int(self.mouse_x + rel_x)
            original_y = int(self.mouse_y + rel_y)
            
            # 確保在圖像範圍內
            h, w = self.image.shape[:2]
            original_x = max(0, min(w-1, original_x))
            original_y = max(0, min(h-1, original_y))
            
            # 保存選擇的點
            self.selected_point = (original_x, original_y)
            
            # 在圖像上標記
            marked_image = self.original_image.copy()
            cv2.circle(marked_image, (original_x, original_y), 8, (0, 255, 0), -1)
            cv2.putText(marked_image, str(self.point_index + 1), 
                       (original_x + 12, original_y + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            self.image = marked_image
            
            # 更新顯示
            self.display_zoom_image()
            
            # 播放聲音反饋（可選）
            try:
                import winsound
                winsound.Beep(1000, 100)
            except:
                pass
                
            print(f"Point {self.point_index + 1} marked at ({original_x}, {original_y})")
            
        except Exception as e:
            print(f"Error in on_click: {e}")
        
        # 重置點擊處理標誌
        self.window.after(100, lambda: setattr(self, 'click_processed', False))
    
    def on_right_click(self, event):
        """右鍵點擊取消"""
        self.selected_point = None
        self.image = self.original_image.copy()
        self.display_zoom_image()
    
    def on_mouse_wheel(self, event):
        """鼠標滾輪事件"""
        if event.num == 5 or event.delta < 0:  # 向下滾動
            self.zoom_factor = max(1.0, self.zoom_factor * 0.9)
        else:  # 向上滾動
            self.zoom_factor = min(10.0, self.zoom_factor * 1.1)
        
        self.display_zoom_image()
        return "break"  # 阻止事件繼續傳遞
    
    def on_key_press(self, event):
        """鍵盤按鍵事件"""
        move_amount = 5  # 移動步長
        
        if event.keysym == 'Up':
            self.mouse_y = max(0, self.mouse_y - move_amount)
        elif event.keysym == 'Down':
            h = self.image.shape[0]
            self.mouse_y = min(h - 1, self.mouse_y + move_amount)
        elif event.keysym == 'Left':
            self.mouse_x = max(0, self.mouse_x - move_amount)
        elif event.keysym == 'Right':
            w = self.image.shape[1]
            self.mouse_x = min(w - 1, self.mouse_x + move_amount)
        elif event.keysym == 'plus' or event.keysym == 'equal':
            self.zoom_factor = min(10.0, self.zoom_factor * 1.1)
        elif event.keysym == 'minus':
            self.zoom_factor = max(1.0, self.zoom_factor * 0.9)
        
        self.display_zoom_image()
    
    def on_confirm(self, event=None):
        """確認選擇"""
        print(f"Confirm button pressed, selected_point: {self.selected_point}")
        
        if self.selected_point:
            # 清除當前放大窗口引用
            if hasattr(self.main_window, 'current_zoom_window'):
                self.main_window.current_zoom_window = None
            
            # 調用回調
            print(f"Calling callback with point: {self.selected_point}")
            self.callback(self.selected_point)
            self.window.destroy()
        else:
            messagebox.showwarning("Warning", "Please mark a point first by clicking on the image.")
    
    def on_cancel(self, event=None):
        """取消選擇"""
        print("Cancel button pressed")
        # 清除當前放大窗口引用
        if hasattr(self.main_window, 'current_zoom_window'):
            self.main_window.current_zoom_window = None
            
        self.window.destroy()

class ROIWindow:
    """用於選擇ROI的窗口 - 修復版"""
    def __init__(self, parent, image, callback):
        self.parent = parent
        self.image = image
        self.callback = callback
        self.roi = None
        
        # 創建窗口
        self.window = tk.Toplevel(parent)
        self.window.title("Select ROI for Tracking")
        self.window.geometry("1000x800")
        self.window.configure(bg='black')
        
        # 創建Canvas
        self.canvas = tk.Canvas(self.window, bg='black', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加說明
        self.create_instructions()
        
        # 初始化變量
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.scale_factor = 1.0
        
        # 顯示圖像
        self.display_image()
        
        # 綁定事件
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.window.bind("<Return>", self.on_confirm)
        self.window.bind("<Escape>", self.on_cancel)
        
        # 設置窗口置頂
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # 設置焦點
        self.window.focus_set()
        self.canvas.focus_set()
        
        # 保存引用
        self.parent.roi_window = self
    
    def create_instructions(self):
        """創建說明文字"""
        instructions_frame = tk.Frame(self.window, bg='black')
        instructions_frame.pack(fill=tk.X, padx=10, pady=5)
        
        instructions = [
            "ROI Selection - Drag to draw rectangle around athlete",
            "Instructions:",
            "1. Click and drag to draw rectangle around athlete",
            "2. Release mouse to set rectangle",
            "3. Press Enter to confirm or Esc to cancel",
            "Tip: Make sure rectangle covers athlete's feet"
        ]
        
        for i, text in enumerate(instructions):
            color = "yellow" if i == 0 else "cyan" if i == 1 else "white"
            label = tk.Label(instructions_frame, text=text, bg="black", 
                           fg=color, font=('Arial', 10))
            label.pack(anchor=tk.W)
    
    def display_image(self):
        """顯示圖像 - 修復版"""
        if self.image is None:
            return
        
        try:
            # 獲取Canvas尺寸
            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            # 如果Canvas尺寸為0，使用默認尺寸
            if canvas_w <= 1 or canvas_h <= 1:
                canvas_w, canvas_h = 800, 600
            
            # 獲取圖像尺寸
            height, width = self.image.shape[:2]
            
            # 計算縮放比例以適應Canvas
            scale_x = canvas_w / width
            scale_y = canvas_h / height
            self.scale_factor = min(scale_x, scale_y) * 0.9  # 留一點邊距
            
            # 計算新尺寸
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            
            # 調整圖像大小
            if new_width > 0 and new_height > 0:
                resized_frame = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized_frame = self.image
                new_width, new_height = width, height
            
            # 將BGR轉換為RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # 轉換為PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # 轉換為PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # 清除Canvas並顯示新圖像
            self.canvas.delete("all")
            
            # 計算圖像在Canvas中的位置（居中顯示）
            self.x_offset = (canvas_w - new_width) // 2
            self.y_offset = (canvas_h - new_height) // 2
            
            self.canvas.create_image(self.x_offset, self.y_offset, image=self.photo, anchor=tk.NW)
            
            # 顯示提示信息
            self.canvas.create_text(canvas_w//2, 30, 
                                   text="Drag to draw rectangle around athlete", 
                                   fill="yellow", font=('Arial', 12, 'bold'))
            
            print(f"Image displayed: {width}x{height} -> {new_width}x{new_height}")
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def on_button_press(self, event):
        """鼠標按壓事件"""
        self.start_x = event.x
        self.start_y = event.y
        
        # 刪除之前的矩形
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
    
    def on_mouse_drag(self, event):
        """鼠標拖動事件"""
        if self.start_x is None or self.start_y is None:
            return
        
        # 刪除之前的矩形
        if self.rect:
            self.canvas.delete(self.rect)
        
        # 繪製新矩形
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="red", width=2
        )
    
    def on_button_release(self, event):
        """鼠標釋放事件"""
        if self.start_x is None or self.start_y is None:
            return
        
        # 確保坐標順序正確
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # 檢查矩形大小
        if abs(x2 - x1) < 20 or abs(y2 - y1) < 20:
            messagebox.showwarning("Warning", "Rectangle too small. Please draw a larger area.")
            return
        
        # 計算原始圖像中的ROI
        height, width = self.image.shape[:2]
        
        # 轉換為原始圖像坐標
        orig_x1 = int((x1 - self.x_offset) / self.scale_factor)
        orig_y1 = int((y1 - self.y_offset) / self.scale_factor)
        orig_x2 = int((x2 - self.x_offset) / self.scale_factor)
        orig_y2 = int((y2 - self.y_offset) / self.scale_factor)
        
        # 確保在圖像範圍內
        orig_x1 = max(0, min(width-1, orig_x1))
        orig_y1 = max(0, min(height-1, orig_y1))
        orig_x2 = max(0, min(width-1, orig_x2))
        orig_y2 = max(0, min(height-1, orig_y2))
        
        # 確保寬高為正
        if orig_x2 > orig_x1 and orig_y2 > orig_y1:
            self.roi = (orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1)
            
            # 在圖像上標記ROI
            marked_image = self.image.copy()
            cv2.rectangle(marked_image, 
                         (orig_x1, orig_y1), (orig_x2, orig_y2),
                         (0, 255, 0), 2)
            cv2.putText(marked_image, "ROI Selected", (orig_x1, orig_y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新顯示
            self.image = marked_image
            self.display_image()
            
            # 顯示ROI信息
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            info_text = f"ROI: ({orig_x1}, {orig_y1}) - ({orig_x2}, {orig_y2})"
            info_text += f"\nSize: {orig_x2-orig_x1}x{orig_y2-orig_y1}"
            info_text += "\nPress Enter to confirm"
            
            self.canvas.create_text(100, canvas_h-50, text=info_text, 
                                  fill="yellow", font=('Arial', 12, 'bold'), anchor=tk.W)
            
            print(f"ROI selected: {self.roi}")
        else:
            self.roi = None
            messagebox.showwarning("Warning", "Invalid ROI selection. Please try again.")
    
    def on_confirm(self, event=None):
        """確認ROI選擇"""
        if self.roi:
            print(f"ROI confirmed: {self.roi}")
            self.callback(self.roi)
            self.window.destroy()
        else:
            messagebox.showwarning("Warning", "Please select a valid ROI first.")
    
    def on_cancel(self, event=None):
        """取消選擇"""
        self.roi = None
        self.callback(None)
        self.window.destroy()

class JumpMeasurementSystem:
    def __init__(self):
        s = []
        self.calibration_points = []
        self.world_points = []
        self.calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        self.kcf_tracker = None
        self.kalman_filter = None
        self.tracking = False
        self.measurement_points = []
        self.velocities = []
        self.angles = []
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame = None
        self.scale_factor = 1.0
        self.athlete_info = {
            'name': 'Unknown',
            'gender': 'Male',
            'age': 20,
            'team': 'Unknown'
        }
        self.current_step = 0
        self.canvas_transforms = {}
        self.current_zoom_window = None
        self.calibration_active = False
        self.roi_window = None
        self.tracking_thread = None
        self.tracking_bbox = None
        self.show_tracking = True
        
        self.root = tk.Tk()
        self.root.title("Long Jump Measurement System v2.0")
        self.root.geometry("1400x900")
        self.setup_styles()
        self.setup_gui()
        self.update_status("System Ready", "info")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        # 定義顏色方案
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'bg': '#f5f7fa'
        }
        
        # 設置窗口背景
        self.root.configure(bg=self.colors['bg'])
        
        # 創建自定義樣式
        style = ttk.Style()
        style.theme_use('clam')
        
        # 按鈕樣式
        style.configure('Primary.TButton', 
                       background=self.colors['secondary'],
                       foreground='white',
                       borderwidth=1,
                       focusthickness=3,
                       focuscolor='none')
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white')
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground='white')
        
        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground='white')
    
    def setup_gui(self):
        # 創建主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側導航面板
        self.create_navigation_panel(main_container)
        
        # 右側主內容區域
        content_frame = ttk.Frame(main_container)
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # 創建選項卡
        self.create_tabs(content_frame)
    
    def create_navigation_panel(self, parent):
        nav_frame = ttk.LabelFrame(parent, text="Workflow Steps", width=250)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        nav_frame.pack_propagate(False)
        
        # 步驟指示器
        steps = [
            ("1. Load Video", "Load video file for analysis"),
            ("2. Athlete Info", "Enter athlete information"),
            ("3. Calibration", "Calibrate camera perspective"),
            ("4. Tracking", "Track athlete movement"),
            ("5. Analysis", "Analyze jump performance"),
            ("6. Results", "View and export results")
        ]
        
        self.step_widgets = []
        for i, (title, desc) in enumerate(steps):
            step_frame = ttk.Frame(nav_frame)
            step_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # 步驟指示器圓圈
            canvas = tk.Canvas(step_frame, width=30, height=30, 
                              bg=self.colors['light'], highlightthickness=0)
            canvas.pack(side=tk.LEFT)
            
            circle = canvas.create_oval(5, 5, 25, 25, fill='lightgray', 
                                       outline=self.colors['dark'], width=2)
            text = canvas.create_text(15, 15, text=str(i+1), 
                                     font=('Arial', 10, 'bold'))
            
            # 步驟標題和描述
            text_frame = ttk.Frame(step_frame)
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
            
            title_label = ttk.Label(text_frame, text=title, 
                                   font=('Arial', 10, 'bold'))
            title_label.pack(anchor=tk.W)
            
            desc_label = ttk.Label(text_frame, text=desc, 
                                  font=('Arial', 8), foreground='gray')
            desc_label.pack(anchor=tk.W)
            
            self.step_widgets.append({
                'frame': step_frame,
                'canvas': canvas,
                'circle': circle,
                'text': text,
                'title': title_label,
                'desc': desc_label
            })
        
        ttk.Separator(nav_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 快速操作按鈕
        quick_frame = ttk.LabelFrame(nav_frame, text="Quick Actions")
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(quick_frame, text="Open Video", 
                  command=self.open_video, style='Primary.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="Reset All", 
                  command=self.reset_all, style='Warning.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="Save Project", 
                  command=self.save_project).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="Load Project", 
                  command=self.load_project).pack(fill=tk.X, pady=2)
        
        # 系統信息
        info_frame = ttk.LabelFrame(nav_frame, text="System Info")
        info_frame.pack(fill=tk.X, padx=10, pady=(20, 5))
        
        self.system_info = tk.Text(info_frame, height=8, width=30, 
                                  font=('Consolas', 8), bg=self.colors['light'])
        self.system_info.pack(padx=5, pady=5)
        self.update_system_info()
    
    def create_tabs(self, parent):
        # 創建選項卡控件
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 創建各個選項卡
        self.tab1 = self.create_video_tab()
        self.tab2 = self.create_athlete_tab()
        self.tab3 = self.create_calibration_tab()
        self.tab4 = self.create_tracking_tab()
        self.tab5 = self.create_analysis_tab()
        self.tab6 = self.create_results_tab()
        
        # 添加到notebook
        self.notebook.add(self.tab1, text="Video")
        self.notebook.add(self.tab2, text="Athlete Info")
        self.notebook.add(self.tab3, text="Calibration")
        self.notebook.add(self.tab4, text="Tracking")
        self.notebook.add(self.tab5, text="Analysis")
        self.notebook.add(self.tab6, text="Results")
        
        # 狀態欄
        self.status_bar = ttk.Frame(parent, height=30)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(self.status_bar, text="Ready", 
                                     font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.progress = ttk.Progressbar(self.status_bar, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=10)
        
        # 綁定選項卡切換事件
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def create_video_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # 視頻控制面板
        control_frame = ttk.LabelFrame(frame, text="Video Control")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 文件路徑顯示
        path_frame = ttk.Frame(control_frame)
        path_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(path_frame, text="Video File:").pack(side=tk.LEFT)
        self.video_path_label = ttk.Label(path_frame, text="No video loaded", 
                                         foreground="gray")
        self.video_path_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="Browse...", 
                  command=self.open_video, width=15).pack(pady=5)
        
        # 視頻信息
        info_frame = ttk.LabelFrame(control_frame, text="Video Information")
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.video_info_text = tk.Text(info_frame, height=6, width=50,
                                      font=('Consolas', 9), bg=self.colors['light'])
        self.video_info_text.pack(padx=5, pady=5)
        
        # 視頻預覽
        preview_frame = ttk.LabelFrame(frame, text="Video Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.video_canvas = tk.Canvas(preview_frame, bg='black')
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 播放控制
        play_frame = ttk.Frame(preview_frame)
        play_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_btn = ttk.Button(play_frame, text="▶ Play", 
                                  command=self.play_video, state='disabled')
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = ttk.Button(play_frame, text="⏸ Pause", 
                                   command=self.pause_video, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = ttk.Button(play_frame, text="⏹ Stop", 
                                  command=self.stop_video, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        self.frame_slider = ttk.Scale(play_frame, from_=0, to=100, 
                                     orient=tk.HORIZONTAL)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        return frame
    
    def create_athlete_tab(self):
        frame = ttk.Frame(self.notebook)
        
        form_frame = ttk.LabelFrame(frame, text="Athlete Information")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 創建表單
        fields = [
            ("Name:", "name", "entry"),
            ("Age:", "age", "spinbox"),
            ("Gender:", "gender", "combobox"),
            ("Team:", "team", "entry"),
            ("Bib Number:", "bib", "entry"),
            ("Category:", "category", "combobox")
        ]
        
        self.athlete_entries = {}
        
        for i, (label, key, field_type) in enumerate(fields):
            row = ttk.Frame(form_frame)
            row.pack(fill=tk.X, padx=20, pady=10)
            
            ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
            
            if field_type == "entry":
                entry = ttk.Entry(row)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.athlete_entries[key] = entry
                
            elif field_type == "spinbox":
                spinbox = tk.Spinbox(row, from_=10, to=50, width=20)
                spinbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.athlete_entries[key] = spinbox
                
            elif field_type == "combobox":
                if key == "gender":
                    values = ["Male", "Female", "Other"]
                else:
                    values = ["Senior", "Junior", "Youth", "Master"]
                
                combobox = ttk.Combobox(row, values=values, state="readonly", width=18)
                combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
                combobox.set(values[0])
                self.athlete_entries[key] = combobox
        
        # 保存按鈕
        ttk.Button(form_frame, text="Save Athlete Info", 
                  command=self.save_athlete_info, style='Success.TButton').pack(pady=20)
        
        # 顯示當前信息
        info_frame = ttk.LabelFrame(frame, text="Current Information")
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.athlete_info_text = tk.Text(info_frame, height=8, width=50,
                                        font=('Consolas', 9), bg=self.colors['light'])
        self.athlete_info_text.pack(padx=5, pady=5)
        
        return frame
    
    def create_calibration_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # 左側說明區域
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=10)
        
        instructions = ttk.LabelFrame(left_frame, text="Calibration Instructions")
        instructions.pack(fill=tk.BOTH, expand=True)
        
        instruction_text = """STEP-BY-STEP CALIBRATION:

1. Click 'Start Calibration' button
2. Mark 6 points in order:

   Point 1: Takeoff board position
   Point 2: 5 meters from takeoff board
   Point 3: 10 meters from takeoff board
   Point 4: 1m above Point 1
   Point 5: 1m above Point 2
   Point 6: 1m above Point 3

3. Use zoom window for precise marking
4. Check reprojection error (< 4px acceptable)

Tips:
- Use clear markers on the ground
- Ensure vertical alignment
- Mark points accurately for best results"""
        
        text_widget = tk.Text(instructions, height=20, width=40,
                             font=('Consolas', 9), bg=self.colors['light'])
        text_widget.insert(1.0, instruction_text)
        text_widget.config(state='disabled')
        text_widget.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 控制按鈕
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.calibrate_btn = ttk.Button(btn_frame, text="Start Calibration", 
                                       command=self.start_calibration,
                                       style='Primary.TButton')
        self.calibrate_btn.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame, text="Clear Points", 
                  command=self.clear_calibration_points).pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_frame, text="Auto-Detect", 
                  command=self.auto_detect_calibration,
                  state='disabled').pack(fill=tk.X, pady=2)
        
        # 添加確認按鈕
        ttk.Button(btn_frame, text="Manual Confirm", 
                  command=self.manual_confirm_point).pack(fill=tk.X, pady=2)
        
        # 右側顯示區域
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        # 校準顯示
        display_frame = ttk.LabelFrame(right_frame, text="Calibration Display")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.calibration_canvas = tk.Canvas(display_frame, bg='black')
        self.calibration_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 校準結果
        result_frame = ttk.LabelFrame(right_frame, text="Calibration Results")
        result_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.calibration_results = tk.Text(result_frame, height=10, width=60,
                                          font=('Consolas', 9), bg=self.colors['light'])
        self.calibration_results.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 綁定Canvas點擊事件
        self.calibration_canvas.bind("<Button-1>", self.on_calibration_click)
        
        # 添加鼠標位置顯示
        self.mouse_pos_label = tk.Label(self.calibration_canvas, 
                                       text="Mouse: (0, 0)", 
                                       bg="black", 
                                       fg="white",
                                       font=('Arial', 10))
        self.mouse_pos_label.place(x=10, y=10)
        
        # 綁定鼠標移動事件
        self.calibration_canvas.bind("<Motion>", self.on_calibration_mouse_move)
        
        return frame
    
    def create_tracking_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # 上部控制區域
        control_frame = ttk.LabelFrame(frame, text="Tracking Control")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 控制按鈕
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=10)
        
        self.track_btn = ttk.Button(btn_frame, text="Start Tracking", 
                                   command=self.start_tracking,
                                   style='Success.TButton')
        self.track_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Pause Tracking", 
                  command=self.pause_tracking).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Stop Tracking", 
                  command=self.stop_tracking,
                  style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        
        # 追蹤參數
        param_frame = ttk.Frame(control_frame)
        param_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(param_frame, text="Tracking Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.track_method = ttk.Combobox(param_frame, values=["KCF", "CSRT", "MOSSE"], 
                                        state="readonly", width=15)
        self.track_method.set("KCF")
        self.track_method.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(param_frame, text="Smoothing:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0), pady=5)
        self.smoothing_var = tk.StringVar(value="Kalman")
        ttk.Radiobutton(param_frame, text="Kalman", variable=self.smoothing_var, 
                       value="Kalman").grid(row=0, column=3, padx=5)
        ttk.Radiobutton(param_frame, text="None", variable=self.smoothing_var, 
                       value="None").grid(row=0, column=4, padx=5)
        
        # 添加跳轉到特定幀選擇ROI的按鈕
        ttk.Label(param_frame, text="Select Frame:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.select_frame_var = tk.StringVar(value="0")
        frame_entry = ttk.Entry(param_frame, textvariable=self.select_frame_var, width=10)
        frame_entry.grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Button(param_frame, text="Go to Frame", 
                  command=self.goto_frame_for_tracking).grid(row=1, column=2, padx=5)
        
        # 下部顯示區域
        display_frame = ttk.Frame(frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 左側視頻
        video_frame = ttk.LabelFrame(display_frame, text="Tracking View")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.tracking_canvas = tk.Canvas(video_frame, bg='black')
        self.tracking_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右側數據
        data_frame = ttk.LabelFrame(display_frame, text="Real-time Data")
        data_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        data_frame.pack_propagate(False)
        data_frame.config(width=300)
        
        # 實時數據顯示
        self.realtime_data = tk.Text(data_frame, height=20, width=35,
                                    font=('Consolas', 9), bg=self.colors['light'])
        self.realtime_data.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 追蹤統計
        stats_frame = ttk.LabelFrame(data_frame, text="Tracking Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        self.stats_text = tk.Text(stats_frame, height=6, width=35,
                                 font=('Consolas', 8), bg=self.colors['light'])
        self.stats_text.pack(padx=5, pady=5)
        
        return frame
    
    def create_analysis_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # 控制面板
        control_frame = ttk.LabelFrame(frame, text="Analysis Control")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Analyze Jump", 
                  command=self.analyze_jump,
                  style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Manual Point Selection", 
                  command=self.manual_point_selection).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Recalculate", 
                  command=self.recalculate_analysis).pack(side=tk.LEFT, padx=5)
        
        # 結果顯示區域
        result_frame = ttk.Frame(frame)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 左側關鍵點結果
        keypoints_frame = ttk.LabelFrame(result_frame, text="Key Point Measurements")
        keypoints_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 創建表格顯示關鍵點數據
        columns = ('Point', 'Distance', 'Speed', 'Horizontal', 'Vertical')
        self.keypoints_tree = ttk.Treeview(keypoints_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.keypoints_tree.heading(col, text=col)
            self.keypoints_tree.column(col, width=100)
        
        self.keypoints_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右側分析結果
        analysis_frame = ttk.LabelFrame(result_frame, text="Analysis Results")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        analysis_frame.pack_propagate(False)
        analysis_frame.config(width=300)
        
        self.analysis_results = tk.Text(analysis_frame, height=20, width=35,
                                       font=('Consolas', 10), bg=self.colors['light'])
        self.analysis_results.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # 圖表按鈕
        chart_frame = ttk.Frame(analysis_frame)
        chart_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Button(chart_frame, text="Velocity Chart", 
                  command=lambda: self.show_chart('velocity')).pack(side=tk.LEFT, padx=2)
        ttk.Button(chart_frame, text="Trajectory Chart", 
                  command=lambda: self.show_chart('trajectory')).pack(side=tk.LEFT, padx=2)
        ttk.Button(chart_frame, text="Angle Chart", 
                  command=lambda: self.show_chart('angle')).pack(side=tk.LEFT, padx=2)
        
        return frame
    
    def create_results_tab(self):
        frame = ttk.Frame(self.notebook)
        
        # 結果摘要
        summary_frame = ttk.LabelFrame(frame, text="Jump Summary")
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=80,
                                   font=('Arial', 10), bg=self.colors['light'])
        self.summary_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 詳細結果
        detail_frame = ttk.LabelFrame(frame, text="Detailed Results")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # 創建詳細結果表格
        columns = ('Parameter', 'Value', 'Unit', 'Grade')
        self.results_tree = ttk.Treeview(detail_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        scrollbar = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)
        
        # 導出按鈕
        export_frame = ttk.Frame(detail_frame)
        export_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Button(export_frame, text="Export to CSV", 
                  command=self.export_to_csv, style='Success.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="Export to PDF", 
                  command=self.export_to_pdf).pack(side=tk.LEFT, padx=2)
        ttk.Button(export_frame, text="Print Report", 
                  command=self.print_report).pack(side=tk.LEFT, padx=2)
        
        return frame
    
    def update_status(self, message, level="info"):
        """更新狀態欄信息"""
        colors = {
            'info': 'black',
            'success': 'green',
            'warning': 'orange',
            'error': 'red'
        }
        
        self.status_label.config(text=message, foreground=colors.get(level, 'black'))
        self.root.update()
    
    def update_system_info(self):
        """更新系統信息"""
        info = f"""System: Long Jump Measurement
Version: 2.0
Date: {datetime.now().strftime('%Y-%m-%d')}
Status: {'Calibrated' if self.calibrated else 'Not Calibrated'}
Video: {'Loaded' if self.video_path else 'Not Loaded'}
Tracking: {'Active' if self.tracking else 'Inactive'}
Calibration Points: {len(self.calibration_points)}/6"""
        
        # 確保先解鎖文本框再更新
        self.system_info.config(state='normal')
        self.system_info.delete(1.0, tk.END)
        self.system_info.insert(1.0, info)
        self.system_info.config(state='disabled')
        self.system_info.update()  # 強制立即更新
    
    def update_step_indicator(self, step):
        """更新步驟指示器"""
        for i, widget in enumerate(self.step_widgets):
            if i == step - 1:
                # 當前步驟
                widget['canvas'].itemconfig(widget['circle'], fill=self.colors['secondary'])
                widget['title'].config(foreground=self.colors['secondary'])
            elif i < step - 1:
                # 已完成步驟
                widget['canvas'].itemconfig(widget['circle'], fill=self.colors['success'])
                widget['title'].config(foreground=self.colors['success'])
            else:
                # 未完成步驟
                widget['canvas'].itemconfig(widget['circle'], fill='lightgray')
                widget['title'].config(foreground='black')
        # 強制更新UI
        self.root.update()
    
    def on_tab_changed(self, event):
        """選項卡切換事件"""
        tab_index = self.notebook.index(self.notebook.select())
        
        # 更新步驟指示器
        self.update_step_indicator(tab_index + 1)
        
        # 根據選項卡更新內容
        if tab_index == 0:  # Video tab
            if self.current_frame is not None:
                self.display_frame_on_canvas(self.current_frame, self.video_canvas)
        elif tab_index == 2:  # Calibration tab
            if self.current_frame is not None:
                self.display_calibration_frame()
        elif tab_index == 3:  # Tracking tab
            # 當切換到Tracking選項卡時，顯示當前幀
            if self.cap and not self.tracking:
                # 獲取當前幀位置
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos > 0:
                    # 如果已經讀取過幀，顯示當前幀
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame.copy()
                        self.display_frame_on_canvas(frame, self.tracking_canvas)
                else:
                    # 否則顯示第一幀
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame.copy()
                        self.display_frame_on_canvas(frame, self.tracking_canvas)
    
    def open_video(self):
        """打開視頻文件"""
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"), ("All files", "*.*")]
        )
        
        if self.video_path:
            try:
                # 釋放之前的視頻資源
                if self.cap is not None:
                    self.cap.release()
                
                # 打開視頻
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    # 嘗試用其他方式打開
                    self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_ANY)
                    if not self.cap.isOpened():
                        raise ValueError("無法打開視頻文件")
                
                # 獲取視頻信息
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or np.isnan(self.fps):
                    self.fps = 30.0
                    
                self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 讀取第一幀
                ret, frame = self.cap.read()
                if not ret:
                    raise ValueError("無法讀取視頻幀")
                
                self.current_frame = frame.copy()
                
                # 重置到第一幀
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # 更新界面
                self.video_path_label.config(text=os.path.basename(self.video_path))
                self.update_video_info()
                self.update_status(f"Video loaded: {os.path.basename(self.video_path)}", "success")
                self.update_step_indicator(1)
                
                # 啟用播放控制
                self.play_btn.config(state='normal')
                self.pause_btn.config(state='normal')
                self.stop_btn.config(state='normal')
                
                # 顯示第一幀到Video Canvas
                self.display_frame_on_canvas(self.current_frame, self.video_canvas)
                
                # 同時顯示到Tracking Canvas（預覽）
                self.display_frame_on_canvas(self.current_frame, self.tracking_canvas)
                
                print(f"Video successfully loaded: {self.video_path}")
                
            except Exception as e:
                error_msg = f"Error loading video: {str(e)}"
                self.update_status(error_msg, "error")
                messagebox.showerror("Error", error_msg)
                print(f"Error details: {e}")
                # 清理資源
                if self.cap:
                    self.cap.release()
                    self.cap = None
    
    def update_video_info(self):
        """更新視頻信息顯示"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            info = f"""File: {os.path.basename(self.video_path)}
Path: {os.path.dirname(self.video_path)}
Frames: {self.frame_count}
FPS: {self.fps:.2f}
Duration: {self.frame_count/self.fps:.2f} sec
Resolution: {width}x{height}
Size: {os.path.getsize(self.video_path)/1024/1024:.2f} MB"""
            
            self.video_info_text.delete(1.0, tk.END)
            self.video_info_text.insert(1.0, info)
            self.video_info_text.config(state='disabled')
    
    def show_frame(self):
        """顯示當前幀"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame_on_canvas(frame, self.video_canvas)
    
    def display_frame_on_canvas(self, frame, canvas):
        """在指定Canvas上顯示幀"""
        if frame is not None:
            try:
                # 調整大小以適應Canvas
                canvas.update_idletasks()
                canvas_w = canvas.winfo_width()
                canvas_h = canvas.winfo_height()
                
                # 如果Canvas尺寸為0，使用默認尺寸
                if canvas_w <= 1 or canvas_h <= 1:
                    canvas_w, canvas_h = 700, 400
                
                # 獲取圖像尺寸
                height, width = frame.shape[:2]
                
                # 計算縮放比例以適應Canvas
                scale_x = canvas_w / width
                scale_y = canvas_h / height
                scale = min(scale_x, scale_y) * 0.95  # 留一點邊距
                
                # 計算新尺寸
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 調整圖像大小
                if new_width > 0 and new_height > 0:
                    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    resized_frame = frame
                    new_width, new_height = width, height
                
                # 將BGR轉換為RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # 轉換為PIL Image
                pil_image = Image.fromarray(rgb_frame)
                
                # 轉換為PhotoImage
                photo = ImageTk.PhotoImage(pil_image)
                
                # 清除Canvas並顯示新圖像
                canvas.delete("all")
                
                # 計算圖像在Canvas中的位置（居中顯示）
                x_offset = (canvas_w - new_width) // 2
                y_offset = (canvas_h - new_height) // 2
                
                canvas.create_image(x_offset, y_offset, image=photo, anchor=tk.NW)
                
                # 保存圖片引用，防止被垃圾回收
                if canvas == self.video_canvas:
                    self.video_canvas.photo = photo
                elif canvas == self.calibration_canvas:
                    self.calibration_canvas.photo = photo
                elif canvas == self.tracking_canvas:
                    self.tracking_canvas.photo = photo
                
                # 保存轉換參數
                self.canvas_transforms[canvas] = {
                    'scale_x': new_width / width,
                    'scale_y': new_height / height,
                    'x_offset': x_offset,
                    'y_offset': y_offset,
                    'original_width': width,
                    'original_height': height
                }
                
                return True
                
            except Exception as e:
                print(f"Error displaying frame on {canvas}: {e}")
                return False
        return False
    
    def play_video(self):
        """播放視頻"""
        if not self.cap:
            return
        
        self.update_status("Playing video...", "info")
        
        def play_loop():
            while self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_frame_on_canvas(frame, self.video_canvas)
                    self.root.update()
                    # 控制播放速度
                    time.sleep(1.0/self.fps)
                else:
                    break
        
        # 在新線程中播放
        threading.Thread(target=play_loop, daemon=True).start()
    
    def pause_video(self):
        """暫停視頻"""
        self.update_status("Video paused", "warning")
    
    def stop_video(self):
        """停止視頻"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.update_status("Video stopped", "info")
    
    def save_athlete_info(self):
        """保存運動員信息"""
        for key, widget in self.athlete_entries.items():
            if isinstance(widget, (ttk.Entry, tk.Spinbox, ttk.Combobox)):
                self.athlete_info[key] = widget.get()
        
        # 更新顯示
        info_text = f"""Name: {self.athlete_info.get('name', 'Unknown')}
Age: {self.athlete_info.get('age', 'Unknown')}
Gender: {self.athlete_info.get('gender', 'Unknown')}
Team: {self.athlete_info.get('team', 'Unknown')}
Bib: {self.athlete_info.get('bib', 'Unknown')}
Category: {self.athlete_info.get('category', 'Unknown')}"""
        
        self.athlete_info_text.delete(1.0, tk.END)
        self.athlete_info_text.insert(1.0, info_text)
        self.athlete_info_text.config(state='disabled')
        
        self.update_status("Athlete information saved", "success")
        self.update_step_indicator(2)
    
    def manual_confirm_point(self):
        """手動確認點"""
        if self.current_zoom_window and self.current_zoom_window.selected_point:
            print("Manual confirm triggered")
            self.current_zoom_window.on_confirm()
    
    def on_calibration_mouse_move(self, event):
        """校準Canvas鼠標移動事件"""
        # 更新鼠標位置顯示
        transform = self.canvas_transforms.get(self.calibration_canvas)
        if transform:
            original_x = int((event.x - transform['x_offset']) / transform['scale_x'])
            original_y = int((event.y - transform['y_offset']) / transform['scale_y'])
            
            # 確保在圖像範圍內
            if (0 <= original_x < transform['original_width'] and 
                0 <= original_y < transform['original_height']):
                self.mouse_pos_label.config(text=f"Mouse: ({original_x}, {original_y})")
            else:
                self.mouse_pos_label.config(text="Mouse: (Outside)")
        else:
            self.mouse_pos_label.config(text=f"Mouse: ({event.x}, {event.y})")
    
    def start_calibration(self):
        """開始校準"""
        if not self.cap:
            self.update_status("Please load a video first", "error")
            return
        
        self.calibration_points = []
        self.calibrated = False  # 重置校準狀態
        self.calibration_active = True
        
        # 定義世界坐標點
        self.world_points = np.array([
            [0, 0, 0],      # 踏板位置
            [-5, 0, 0],     # 5米點
            [-10, 0, 0],    # 10米點
            [0, 1, 0],      # 踏板上方1米
            [-5, 1, 0],     # 5米點上方1米
            [-10, 1, 0]     # 10米點上方1米
        ], dtype=np.float32)
        
        self.update_status("Calibration started. Click points in order.", "info")
        self.update_step_indicator(3)
        
        # 顯示校準界面
        self.notebook.select(2)  # 切換到校準選項卡
        
        # 重置到第一幀
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.display_calibration_frame()
            
        # 立即更新系統信息
        self.update_system_info()
    
    def on_calibration_click(self, event):
        """校準Canvas點擊事件"""
        # 檢查是否正在校準且還有點需要標記
        if not self.calibration_active or len(self.calibration_points) >= 6:
            return
            
        # 檢查是否已經有放大窗口打開
        if self.current_zoom_window is not None:
            try:
                if self.current_zoom_window.window.winfo_exists():
                    self.update_status("Zoom window already open. Please close it first.", "warning")
                    return
            except:
                pass
        
        # 獲取轉換信息
        transform = self.canvas_transforms.get(self.calibration_canvas)
        if transform is None:
            self.update_status("Cannot get transformation info. Please display the image first.", "error")
            return
        
        # 將Canvas坐標轉換為原始圖像坐標
        original_x = int((event.x - transform['x_offset']) / transform['scale_x'])
        original_y = int((event.y - transform['y_offset']) / transform['scale_y'])
        
        # 確保坐標在圖像範圍內
        if (0 <= original_x < transform['original_width'] and 
            0 <= original_y < transform['original_height']):
            
            # 關閉現有的放大窗口（如果存在）
            if self.current_zoom_window is not None:
                try:
                    if self.current_zoom_window.window.winfo_exists():
                        self.current_zoom_window.window.destroy()
                except:
                    pass
                self.current_zoom_window = None
            
            # 打開改進的放大窗口
            print(f"Opening zoom window for point {len(self.calibration_points) + 1}")
            self.current_zoom_window = ZoomWindow(
                self.root, self, len(self.calibration_points), 
                self.add_calibration_point
            )
            
            # 設置初始鼠標位置為點擊位置
            self.current_zoom_window.mouse_x = original_x
            self.current_zoom_window.mouse_y = original_y
            self.current_zoom_window.display_zoom_image()
            
            self.update_status(f"Zoom window opened for point {len(self.calibration_points) + 1}", "info")
        else:
            self.update_status("Clicked outside the image.", "warning")
    
    def add_calibration_point(self, point):
        """添加校準點"""
        print(f"Adding calibration point: {point}")
        
        # 防止重複添加
        if point in self.calibration_points:
            self.update_status("Point already marked", "warning")
            return
            
        self.calibration_points.append(point)
        
        # 清除當前放大窗口引用
        self.current_zoom_window = None
        
        # 在主窗口更新顯示
        self.display_calibration_frame()
        
        # 更新結果顯示
        self.calibration_results.delete(1.0, tk.END)
        point_num = len(self.calibration_points)
        self.calibration_results.insert(1.0, f"Point {point_num}: {point}\n")
        
        # 更新系統信息
        self.update_system_info()
        
        # 播放成功聲音
        try:
            import winsound
            winsound.Beep(1500, 200)
        except:
            pass
        
        self.update_status(f"Point {point_num} marked successfully", "success")
        
        # 如果收集了6個點，自動校準
        if len(self.calibration_points) == 6:
            self.calibration_active = False
            self.update_status("All points marked. Performing calibration...", "info")
            self.perform_calibration()
    
    def display_calibration_frame(self):
        """在校準Canvas上顯示幀"""
        if self.current_frame is not None:
            # 創建帶有標記的幀
            marked_frame = self.current_frame.copy()
            
            # 繪製已標記的點
            for i, (x, y) in enumerate(self.calibration_points):
                # 使用不同顏色區分點
                if i < 3:  # 地面點
                    color = (0, 255, 0)  # 綠色
                else:  # 上方點
                    color = (0, 255, 255)  # 黃色
                
                cv2.circle(marked_frame, (int(x), int(y)), 10, color, -1)
                cv2.putText(marked_frame, str(i+1), (int(x)+15, int(y)+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 連接地面點和上方點
            if len(self.calibration_points) >= 3:
                # 連接地面點
                for i in range(2):
                    if i+1 < len(self.calibration_points) and i < 3:
                        cv2.line(marked_frame, 
                                (int(self.calibration_points[i][0]), int(self.calibration_points[i][1])),
                                (int(self.calibration_points[i+1][0]), int(self.calibration_points[i+1][1])),
                                (0, 255, 0), 2)
            
            # 繪製垂直線（連接對應的地面點和上方點）
            if len(self.calibration_points) >= 6:
                for i in range(3):
                    cv2.line(marked_frame,
                            (int(self.calibration_points[i][0]), int(self.calibration_points[i][1])),
                            (int(self.calibration_points[i+3][0]), int(self.calibration_points[i+3][1])),
                            (255, 0, 0), 2)
            
            # 顯示指導文字
            if len(self.calibration_points) < 6:
                point_names = [
                    "Takeoff board position",
                    "5 meters from takeoff board",
                    "10 meters from takeoff board",
                    "1m above takeoff board",
                    "1m above 5m point",
                    "1m above 10m point"
                ]
                
                current_point = len(self.calibration_points)
                text = f"Point {current_point + 1}: {point_names[current_point]}"
                cv2.putText(marked_frame, text, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            self.display_frame_on_canvas(marked_frame, self.calibration_canvas)
    
    def perform_calibration(self):
        """執行相機校準 - 修復BUG：確保校準狀態正確更新"""
        try:
            image_points = np.array(self.calibration_points, dtype=np.float32)
            
            # 相機內參
            focal_length = 800
            height, width = self.current_frame.shape[:2]
            center = (width//2, height//2)
            
            self.camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.zeros((4, 1))
            
            print(f"Performing calibration with {len(self.calibration_points)} points")
            print(f"Image points shape: {image_points.shape}")
            print(f"World points shape: {self.world_points.shape}")
            
            # 求解相機姿態
            success, self.rvec, self.tvec = cv2.solvePnP(
                self.world_points, image_points,
                self.camera_matrix, self.dist_coeffs
            )
            
            print(f"Calibration success: {success}")
            
            if success:
                self.calibrated = True  # 確保設置為True
                print(f"self.calibrated set to: {self.calibrated}")
                
                # 計算重投影誤差
                projected_points, _ = cv2.projectPoints(
                    self.world_points, self.rvec, self.tvec,
                    self.camera_matrix, self.dist_coeffs
                )
                
                error = np.mean(np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1))
                
                # 顯示結果 - 放寬標準到4像素
                result_text = f"""Calibration Complete!
                
Reprojection Error: {error:.2f} pixels
{'✓ Good (Error < 2px)' if error < 2 else '✓ Acceptable (Error < 4px)' if error < 4 else '⚠ Fair (Error < 6px)' if error < 6 else '✗ Poor (Error >= 6px)'}

Calibration Status: {'SUCCESSFUL' if error < 4 else 'ACCEPTABLE' if error < 6 else 'POOR - Consider re-calibrating'}

Camera Matrix:
{self.camera_matrix}

Rotation Vector:
{self.rvec.flatten()}

Translation Vector:
{self.tvec.flatten()}

World Points (m):
{self.world_points}

Image Points (px):
{image_points}"""
                
                self.calibration_results.delete(1.0, tk.END)
                self.calibration_results.insert(1.0, result_text)
                
                if error < 4:
                    self.update_status(f"Calibration complete! Error: {error:.2f}px (Good)", "success")
                elif error < 6:
                    self.update_status(f"Calibration complete! Error: {error:.2f}px (Acceptable)", "warning")
                else:
                    self.update_status(f"Calibration complete! Error: {error:.2f}px (Poor - consider re-calibrating)", "error")
                
                # 立即更新系統信息
                self.update_system_info()
                
                # 在校準圖像上繪製重投影點
                result_frame = self.current_frame.copy()
                
                # 繪製原始點
                for i, (x, y) in enumerate(image_points):
                    cv2.circle(result_frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                    cv2.putText(result_frame, str(i+1), (int(x)+15, int(y)+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 繪製重投影點
                for i, (x, y) in enumerate(projected_points.reshape(-1, 2)):
                    cv2.circle(result_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.line(result_frame, (int(image_points[i][0]), int(image_points[i][1])),
                            (int(x), int(y)), (255, 0, 0), 1)
                
                # 顯示誤差信息
                cv2.putText(result_frame, f"Reprojection Error: {error:.2f}px", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 0) if error < 4 else (0, 255, 255) if error < 6 else (0, 0, 255), 2)
                
                self.display_frame_on_canvas(result_frame, self.calibration_canvas)
                
                # 播放成功聲音
                try:
                    import winsound
                    winsound.Beep(1500, 300)
                except:
                    pass
                    
            else:
                self.calibrated = False
                self.update_status("Calibration failed. Please try again.", "error")
                self.update_system_info()
                
        except Exception as e:
            print(f"Error in perform_calibration: {e}")
            self.calibrated = False
            self.update_status(f"Calibration error: {str(e)}", "error")
            self.update_system_info()
    
    def clear_calibration_points(self):
        """清除校準點"""
        self.calibration_points = []
        self.calibrated = False
        self.calibration_active = False
        
        # 關閉現有的放大窗口
        if self.current_zoom_window is not None:
            try:
                if self.current_zoom_window.window.winfo_exists():
                    self.current_zoom_window.window.destroy()
            except:
                pass
            self.current_zoom_window = None
            
        self.display_calibration_frame()
        self.calibration_results.delete(1.0, tk.END)
        self.update_status("Calibration points cleared", "warning")
        self.update_system_info()
    
    def auto_detect_calibration(self):
        """自動檢測校準點"""
        self.update_status("Auto-detection not implemented yet", "warning")
    
    def goto_frame_for_tracking(self):
        """跳轉到指定幀用於追蹤"""
        try:
            frame_num = int(self.select_frame_var.get())
            if self.cap:
                frame_num = max(0, min(frame_num, self.frame_count - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_frame_on_canvas(frame, self.tracking_canvas)
                    self.update_status(f"Jumped to frame {frame_num}", "info")
        except:
            self.update_status("Invalid frame number", "error")
    
    def start_tracking(self):
        """開始追蹤 - 僅開啟ROI選擇視窗，追蹤器於回呼中創建"""
        if not self.calibrated:
            self.update_status("請先完成相機校準", "error")
            messagebox.showwarning("未校準", "請先完成相機校準再開始追蹤。")
            return
        
        if not self.cap:
            self.update_status("請先載入視頻", "error")
            return
        
        # 獲取當前幀用於ROI選擇
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_pos > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.display_frame_on_canvas(frame, self.tracking_canvas)
            self.roi_window = ROIWindow(self.root, frame, self.on_roi_selected)
            self.update_status("請拖動鼠標選擇運動員區域", "info")
        else:
            self.update_status("無法讀取視頻幀", "error")

    def on_roi_selected(self, roi):
        """ROI選擇回調 - 修復版：增加追蹤器實例化邏輯"""
        print(f"ROI selected callback: {roi}")
        if roi:
            x, y, w, h = roi
            bbox = (x, y, w, h)
            
            # 1. 獲取當前選擇的追蹤算法
            tracker_type = self.track_method.get()
            print(f"Initializing tracker: {tracker_type}")
            
            # 2. 根據選擇創建追蹤器實例 (這是之前缺失的關鍵步驟)
            try:
                if tracker_type == 'KCF':
                    self.kcf_tracker = cv2.TrackerKCF_create()
                elif tracker_type == 'CSRT':
                    self.kcf_tracker = cv2.TrackerCSRT_create()
                elif tracker_type == 'MOSSE':
                    # MOSSE 通常在 legacy 中，如果報錯請嘗試 cv2.TrackerMOSSE_create()
                    try:
                        self.kcf_tracker = cv2.legacy.TrackerMOSSE_create()
                    except:
                        self.kcf_tracker = cv2.TrackerMOSSE_create()
                else:
                    # 默認使用 CSRT (最穩定)
                    self.kcf_tracker = cv2.TrackerCSRT_create()
            except AttributeError:
                # 處理 OpenCV 版本兼容性問題
                print("Warning: Standard tracker creation failed, trying legacy...")
                try:
                    if tracker_type == 'KCF':
                        self.kcf_tracker = cv2.legacy.TrackerKCF_create()
                    elif tracker_type == 'CSRT':
                        self.kcf_tracker = cv2.legacy.TrackerCSRT_create()
                except Exception as e:
                    self.update_status(f"Error creating tracker: {e}", "error")
                    messagebox.showerror("Error", f"Could not create tracker: {e}")
                    return

            # 3. 初始化追蹤器
            frame = self.current_frame.copy()
            
            try:
                success = self.kcf_tracker.init(frame, bbox)
                
                if success:
                    self.tracking_bbox = bbox
                    self.tracking = True
                    self.measurement_points = []
                    
                    # 初始化卡爾曼濾波器
                    # 確保這裡使用的是真實的 FPS
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if actual_fps <= 0 or np.isnan(actual_fps):
                        actual_fps = 30
                    dt = 1.0 / actual_fps
                    
                    if self.smoothing_var.get() == "Kalman":
                        print(f"Initializing Kalman filter with dt={dt:.4f} (FPS={actual_fps})")
                        self.kalman_filter = KalmanFilter3D(dt=dt)
                    
                    self.update_status(f"Tracking started using {tracker_type}", "success")
                    self.update_step_indicator(4)
                    
                    # 開始追蹤線程
                    self.start_tracking_thread()
                else:
                    self.update_status("Tracker initialization failed (init returned False)", "error")
                    messagebox.showerror("Error", "Failed to initialize tracker. Please try another region.")
                    
            except Exception as e:
                self.update_status(f"Tracker init crash: {e}", "error")
                print(f"Tracker init error: {e}")
                
        else:
            self.update_status("ROI selection cancelled", "warning")
            # 清除ROI窗口引用
            self.roi_window = None
        
    
    def start_tracking_thread(self):
        """啟動追蹤線程"""
        self.tracking_thread = threading.Thread(target=self.track_athlete)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
    
    def track_athlete(self):
        """追蹤運動員主循環"""
        frame_pos = 0
        
        while self.tracking and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 更新追蹤器
            success, bbox = self.kcf_tracker.update(frame)
            
            if success:
                # 處理追蹤結果
                x, y, w, h = bbox
                foot_x = x + w // 2
                foot_y = y + h
                
                # 轉換為世界坐標
                world_point = self.image_to_world(foot_x, foot_y)
                
                # 卡爾曼濾波
                if self.kalman_filter:
                    filtered_point = self.kalman_filter.update(world_point)
                    velocity = self.kalman_filter.get_velocity()
                else:
                    filtered_point = world_point
                    velocity = np.zeros(3)
                
                # 保存數據
                self.measurement_points.append({
                    'frame': frame_pos,
                    'time': frame_pos / self.fps,
                    'image_point': (foot_x, foot_y),
                    'world_point': world_point,
                    'filtered_point': filtered_point,
                    'velocity': velocity.copy()
                })
                
                # 更新實時數據顯示
                self.update_realtime_data(frame_pos, filtered_point, velocity)
                
                # 繪製追蹤結果
                cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 0, 255), -1)
                
                # 顯示速度和高度信息
                speed = np.linalg.norm(velocity[:2])
                height = filtered_point[2]
                info_text = f"Frame: {frame_pos} | Speed: {speed:.2f} m/s | Height: {height:.2f} m"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 顯示在追蹤Canvas上
                self.display_frame_on_canvas(frame, self.tracking_canvas)
            
            # 更新界面
            self.root.update()
            
            # 控制速度
            time.sleep(1.0/self.fps)
        
        self.tracking = False
        self.update_status("Tracking completed", "success")
    
    def update_realtime_data(self, frame, position, velocity):
        """更新實時數據顯示"""
        speed = np.linalg.norm(velocity[:2])  # 只考慮XY平面速度
        
        # 計算起跳角度（如果速度足夠）
        takeoff_angle = 0
        if abs(velocity[0]) > 0.1:  # 避免除以零
            takeoff_angle_rad = np.arctan2(velocity[2], velocity[0])  # Z/X
            takeoff_angle = np.degrees(takeoff_angle_rad)
        
        data = f"""Frame: {frame}
Time: {frame/self.fps:.2f}s

Position (m):
X: {position[0]:.2f}
Y: {position[1]:.2f}
Z: {position[2]:.2f}

Velocity (m/s):
Speed: {speed:.2f}
Vx: {velocity[0]:.2f}
Vy: {velocity[1]:.2f}
Vz: {velocity[2]:.2f}

Distance: {abs(position[0]):.2f}m
Height: {position[2]:.2f}m
Takeoff Angle: {takeoff_angle:.1f}°"""
        
        self.realtime_data.delete(1.0, tk.END)
        self.realtime_data.insert(1.0, data)
        self.realtime_data.config(state='disabled')
    
    def pause_tracking(self):
        """暫停追蹤"""
        self.tracking = False
        self.update_status("Tracking paused", "warning")
    
    def stop_tracking(self):
        """停止追蹤"""
        self.tracking = False
        self.update_status("Tracking stopped", "info")
    
    def analyze_jump(self):
        """分析跳遠"""
        if len(self.measurement_points) < 10:
            self.update_status("Not enough tracking data", "error")
            return
        
        self.update_status("Analyzing jump...", "info")
        self.update_step_indicator(5)
        
        # 切換到分析選項卡
        self.notebook.select(4)
        
        # 尋找起跳點
        takeoff_index = self.find_takeoff_point()
        
        if takeoff_index is not None:
            # 計算關鍵點速度
            self.calculate_keypoint_velocities(takeoff_index)
            
            # 計算起跳角度
            takeoff_angle = self.calculate_takeoff_angle(takeoff_index)
            
            # 顯示結果
            self.display_analysis_results(takeoff_index, takeoff_angle)
        
        self.update_status("Analysis complete", "success")
    
    def find_takeoff_point(self):
        """尋找起跳點"""
        if len(self.measurement_points) < 20:
            return None
        
        # 尋找垂直速度顯著增加的點
        vertical_velocities = []
        for i in range(5, len(self.measurement_points) - 5):
            # 使用5幀窗口計算垂直速度
            prev_point = self.measurement_points[i-5]
            next_point = self.measurement_points[i+5]
            dt = next_point['time'] - prev_point['time']
            
            if dt > 0:
                vz = (next_point['filtered_point'][2] - prev_point['filtered_point'][2]) / dt
                vertical_velocities.append((i, vz))
        
        if not vertical_velocities:
            return None
        
        # 找到垂直速度最大點
        takeoff_index = max(vertical_velocities, key=lambda x: abs(x[1]))[0]
        
        # 確保起跳點在合理範圍內
        if takeoff_index < 10 or takeoff_index > len(self.measurement_points) - 10:
            takeoff_index = len(self.measurement_points) // 2
        
        return takeoff_index
    
    def calculate_keypoint_velocities(self, takeoff_index):
        """計算關鍵點速度"""
        # 找到10m、5m和起跳點
        takeoff_point = self.measurement_points[takeoff_index]
        takeoff_x = takeoff_point['filtered_point'][0]
        
        ten_meter_index = None
        five_meter_index = None
        
        for i, point in enumerate(self.measurement_points[:takeoff_index]):
            x = point['filtered_point'][0]
            distance = abs(x - takeoff_x)
            
            if abs(distance - 10) < 0.5 and ten_meter_index is None:
                ten_meter_index = i
            elif abs(distance - 5) < 0.5 and five_meter_index is None:
                five_meter_index = i
        
        # 計算速度
        self.keypoint_results = {}
        
        if ten_meter_index and ten_meter_index > 5:
            # 10米點速度
            start_idx = max(0, ten_meter_index - 5)
            end_idx = min(len(self.measurement_points)-1, ten_meter_index + 5)
            
            vx, vy, vz = self.calculate_average_velocity(start_idx, end_idx)
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            self.keypoint_results['10m'] = {
                'speed': speed,
                'velocity': np.array([vx, vy, vz]),
                'distance': 10,
                'position': self.measurement_points[ten_meter_index]['filtered_point']
            }
        
        if five_meter_index and five_meter_index > 5:
            # 5米點速度
            start_idx = max(0, five_meter_index - 5)
            end_idx = min(len(self.measurement_points)-1, five_meter_index + 5)
            
            vx, vy, vz = self.calculate_average_velocity(start_idx, end_idx)
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            
            self.keypoint_results['5m'] = {
                'speed': speed,
                'velocity': np.array([vx, vy, vz]),
                'distance': 5,
                'position': self.measurement_points[five_meter_index]['filtered_point']
            }
        
        # 起跳點速度
        start_idx = max(0, takeoff_index - 5)
        end_idx = min(len(self.measurement_points)-1, takeoff_index + 5)
        
        vx, vy, vz = self.calculate_average_velocity(start_idx, end_idx)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        
        self.keypoint_results['takeoff'] = {
            'speed': speed,
            'velocity': np.array([vx, vy, vz]),
            'distance': 0,
            'position': takeoff_point['filtered_point']
        }
    
    def calculate_average_velocity(self, start_idx, end_idx):
        """計算平均速度"""
        if end_idx <= start_idx:
            return 0, 0, 0
        
        total_vx, total_vy, total_vz = 0, 0, 0
        count = 0
        
        for i in range(start_idx, end_idx):
            total_vx += self.measurement_points[i]['velocity'][0]
            total_vy += self.measurement_points[i]['velocity'][1]
            total_vz += self.measurement_points[i]['velocity'][2]
            count += 1
        
        if count > 0:
            return total_vx/count, total_vy/count, total_vz/count
        else:
            return 0, 0, 0
    
    def calculate_takeoff_angle(self, takeoff_index):
        """計算起跳角度"""
        if takeoff_index < 5 or takeoff_index >= len(self.measurement_points) - 5:
            return 0
        
        # 使用起跳點附近5幀計算角度
        start_idx = max(0, takeoff_index - 5)
        end_idx = min(len(self.measurement_points)-1, takeoff_index + 5)
        
        vx, vy, vz = self.calculate_average_velocity(start_idx, end_idx)
        
        # 計算起跳角度 (水平面角度)
        if abs(vx) > 0.1:
            angle_rad = np.arctan2(vz, vx)  # Z/X 平面
            angle = np.degrees(angle_rad)
        else:
            angle = 90 if vz > 0 else -90
        
        return angle
    
    def display_analysis_results(self, takeoff_index, takeoff_angle):
        """顯示分析結果"""
        # 更新分析結果文本
        self.analysis_results.delete(1.0, tk.END)
        
        result_text = "=== LONG JUMP ANALYSIS RESULTS ===\n\n"
        
        # 顯示關鍵點速度
        for key in ['10m', '5m', 'takeoff']:
            if key in self.keypoint_results:
                v = self.keypoint_results[key]
                result_text += f"{key.upper()} POINT:\n"
                result_text += f"  Speed: {v['speed']:.2f} m/s\n"
                result_text += f"  Horizontal (X): {v['velocity'][0]:.2f} m/s\n"
                result_text += f"  Vertical (Z): {v['velocity'][2]:.2f} m/s\n"
                result_text += f"  Position: ({v['position'][0]:.2f}, {v['position'][2]:.2f}) m\n\n"
        
        # 顯示起跳角度
        result_text += f"TAKEOFF ANGLE: {takeoff_angle:.1f}°\n\n"
        
        # 顯示其他統計
        max_height = max(p['filtered_point'][2] for p in self.measurement_points)
        takeoff_frame = self.measurement_points[takeoff_index]['frame']
        takeoff_time = self.measurement_points[takeoff_index]['time']
        
        result_text += f"Maximum Height: {max_height:.2f} m\n"
        result_text += f"Takeoff Frame: {takeoff_frame}\n"
        result_text += f"Takeoff Time: {takeoff_time:.2f} s\n"
        
        self.analysis_results.insert(1.0, result_text)
        
        # 更新關鍵點表格
        self.keypoints_tree.delete(*self.keypoints_tree.get_children())
        
        for key in ['10m', '5m', 'takeoff']:
            if key in self.keypoint_results:
                v = self.keypoint_results[key]
                self.keypoints_tree.insert('', 'end', values=(
                    key.upper(),
                    f"{v['distance']}m",
                    f"{v['speed']:.2f}",
                    f"{v['velocity'][0]:.2f}",
                    f"{v['velocity'][2]:.2f}"
                ))
    
    def manual_point_selection(self):
        """手動選擇關鍵點"""
        self.update_status("Manual point selection", "info")
    
    def recalculate_analysis(self):
        """重新計算分析"""
        self.update_status("Recalculating...", "info")
    
    def show_chart(self, chart_type):
        """顯示圖表"""
        if not hasattr(self, 'measurement_points') or len(self.measurement_points) == 0:
            self.update_status("No data to plot", "error")
            return
        
        # 創建圖表窗口
        chart_window = tk.Toplevel(self.root)
        chart_window.title(f"{chart_type.capitalize()} Chart")
        chart_window.geometry("800x600")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if chart_type == 'velocity':
            # 速度圖
            times = [p['time'] for p in self.measurement_points]
            velocities = [np.linalg.norm(p['velocity'][:2]) for p in self.measurement_points]
            
            ax.plot(times, velocities, 'b-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title('Athlete Speed Profile')
            ax.grid(True)
            
        elif chart_type == 'trajectory':
            # 軌跡圖
            x_pos = [p['filtered_point'][0] for p in self.measurement_points]
            z_pos = [p['filtered_point'][2] for p in self.measurement_points]  # Z軸是高度
            
            ax.plot(x_pos, z_pos, 'r-', linewidth=2)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Height (m)')
            ax.set_title('Athlete Trajectory (X-Z Plane)')
            ax.grid(True)
            
        elif chart_type == 'angle':
            # 角度圖
            times = [p['time'] for p in self.measurement_points]
            angles = []
            
            for p in self.measurement_points:
                v = p['velocity']
                if np.linalg.norm(v[:2]) > 0.1:  # 避免除以零
                    angle = np.degrees(np.arctan2(v[2], v[0]))  # Z/X
                    angles.append(angle)
                else:
                    angles.append(0)
            
            ax.plot(times, angles, 'g-', linewidth=2)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (degrees)')
            ax.set_title('Takeoff Angle Profile')
            ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具欄
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, chart_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def export_to_csv(self):
        """導出到CSV"""
        if not hasattr(self, 'measurement_points') or len(self.measurement_points) == 0:
            self.update_status("No data to export", "error")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            # 創建DataFrame
            data = []
            for p in self.measurement_points:
                data.append({
                    'Frame': p['frame'],
                    'Time(s)': p['time'],
                    'X(m)': p['filtered_point'][0],
                    'Y(m)': p['filtered_point'][1],
                    'Z(m)': p['filtered_point'][2],
                    'Speed(m/s)': np.linalg.norm(p['velocity'][:2]),
                    'Vx(m/s)': p['velocity'][0],
                    'Vy(m/s)': p['velocity'][1],
                    'Vz(m/s)': p['velocity'][2]
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            self.update_status(f"Data exported to {os.path.basename(file_path)}", "success")
    
    def export_to_pdf(self):
        """導出到PDF"""
        self.update_status("PDF export not implemented yet", "warning")
    
    def print_report(self):
        """打印報告"""
        self.update_status("Print function not implemented yet", "warning")
    
    def save_project(self):
        """保存項目"""
        project_data = {
            'video_path': self.video_path,
            'calibration_points': self.calibration_points,
            'world_points': self.world_points.tolist() if hasattr(self.world_points, 'tolist') else self.world_points,
            'athlete_info': self.athlete_info,
            'calibrated': self.calibrated
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(project_data, f)
            
            self.update_status(f"Project saved to {os.path.basename(file_path)}", "success")
    
    def load_project(self):
        """加載項目"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            # 加載數據
            self.video_path = project_data.get('video_path')
            self.calibration_points = project_data.get('calibration_points', [])
            self.world_points = np.array(project_data.get('world_points', []))
            self.athlete_info = project_data.get('athlete_info', {})
            self.calibrated = project_data.get('calibrated', False)
            
            # 更新界面
            if self.video_path and os.path.exists(self.video_path):
                self.open_video()
            
            # 更新運動員信息
            for key, value in self.athlete_info.items():
                if key in self.athlete_entries:
                    widget = self.athlete_entries[key]
                    if isinstance(widget, (ttk.Entry, tk.Spinbox)):
                        widget.delete(0, tk.END)
                        widget.insert(0, str(value))
                    elif isinstance(widget, ttk.Combobox):
                        widget.set(value)
            
            self.update_status(f"Project loaded from {os.path.basename(file_path)}", "success")
            # 立即更新系統信息
            self.update_system_info()
    
    def reset_all(self):
        """重置所有數據"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all data? This cannot be undone."):
            self.calibration_points = []
            self.world_points = []
            self.calibrated = False
            self.calibration_active = False
            self.camera_matrix = None
            self.dist_coeffs = None
            self.rvec = None
            self.tvec = None
            self.kcf_tracker = None
            self.kalman_filter = None
            self.tracking = False
            self.measurement_points = []
            self.velocities = []
            self.angles = []
            self.video_path = None
            self.current_frame = None
            self.tracking_bbox = None
            
            # 關閉現有的放大窗口
            if self.current_zoom_window is not None:
                try:
                    if self.current_zoom_window.window.winfo_exists():
                        self.current_zoom_window.window.destroy()
                except:
                    pass
                self.current_zoom_window = None
            
            # 關閉ROI窗口
            if self.roi_window is not None:
                try:
                    if self.roi_window.window.winfo_exists():
                        self.roi_window.window.destroy()
                except:
                    pass
                self.roi_window = None
            
            # 釋放視頻資源
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # 重置界面
            self.video_path_label.config(text="No video loaded")
            self.video_info_text.delete(1.0, tk.END)
            self.calibration_results.delete(1.0, tk.END)
            self.realtime_data.delete(1.0, tk.END)
            self.analysis_results.delete(1.0, tk.END)
            self.summary_text.delete(1.0, tk.END)
            
            # 清除Canvas
            self.video_canvas.delete("all")
            self.calibration_canvas.delete("all")
            self.tracking_canvas.delete("all")
            
            # 重置按鈕狀態
            self.play_btn.config(state='disabled')
            self.pause_btn.config(state='disabled')
            self.stop_btn.config(state='disabled')
            
            # 重置步驟
            self.update_step_indicator(0)
            self.update_status("All data has been reset", "info")
            self.update_system_info()
    
    def image_to_world(self, u, v):
        """
        修正版：單目視覺3D轉換
        假設運動員在Y=0的平面上運動（側面視角），解算X(距離)和Z(高度)。
        
        參數：
            u, v: 圖像像素坐標
            
        返回：
            [X, Y, Z] 世界坐標，Y固定為0
        """
        if not self.calibrated:
            return np.array([0, 0, 0])
        
        # 1. 獲取投影矩陣 P = K * [R|t]
        R_mat, _ = cv2.Rodrigues(self.rvec)
        # 組合外部參數矩陣 [R|t] -> 3x4
        rt_mat = np.hstack((R_mat, self.tvec))
        # 總投影矩陣 P -> 3x4
        P = np.dot(self.camera_matrix, rt_mat)
        
        # 2. 應用「平面假設」: Y = 0
        # P矩陣原本是 [col0, col1, col2, col3] 對應 [X, Y, Z, 1]
        # 因為Y=0，P的第二列(col1)對結果無影響，刪除它
        # 形成一個3x3的單應性矩陣H，映射(X, Z, 1) -> (u*s, v*s, s)
        
        col_x = P[:, 0]  # X的係數
        col_z = P[:, 2]  # Z的係數（高度）
        col_c = P[:, 3]  # 常數項
        
        # 構建3x3矩陣 H_xz
        H_xz = np.column_stack((col_x, col_z, col_c))
        
        # 3. 反解X和Z
        # 像素坐標向量 p = [u, v, 1]
        pixel_point = np.array([u, v, 1.0])
        
        # 計算H的逆矩陣
        try:
            H_inv = np.linalg.inv(H_xz)
            world_xz_homo = np.dot(H_inv, pixel_point)
            
            # 歸一化（除以最後一個分量）
            scale = world_xz_homo[2]
            if abs(scale) < 1e-6:  # 防止除以零
                return np.array([0, 0, 0])
                
            world_x = world_xz_homo[0] / scale
            world_z = world_xz_homo[1] / scale
            world_y = 0  # 我們的假設
            
            return np.array([world_x, world_y, world_z])
            
        except np.linalg.LinAlgError:
            print("Error: Matrix inversion failed")
            return np.array([0, 0, 0])
    
    def on_closing(self):
        """窗口關閉事件"""
        # 停止追蹤
        self.tracking = False
        
        # 關閉所有子窗口
        if self.current_zoom_window is not None:
            try:
                if self.current_zoom_window.window.winfo_exists():
                    self.current_zoom_window.window.destroy()
            except:
                pass
        
        if self.roi_window is not None:
            try:
                if self.roi_window.window.winfo_exists():
                    self.roi_window.window.destroy()
            except:
                pass
        
        # 釋放視頻資源
        if self.cap:
            self.cap.release()
        
        # 關閉主窗口
        self.root.destroy()
    
    def run(self):
        """運行主程序"""
        self.root.mainloop()

if __name__ == "__main__":
    app = JumpMeasurementSystem()
    app.run()