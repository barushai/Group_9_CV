import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import time

# Force CPU usage to avoid CUDA NMS compatibility issues
torch.cuda.is_available = lambda: False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DashcamAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Analysis System - Group 29")
        self.geometry("1050x650")

        print("Initializing AI Model (YOLOv8)...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.to('cpu')  # Ensure model runs on CPU
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        self.cap = None
        self.running = False
        self.setup_ui()

    def setup_ui(self):
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Press Start to analyze data")
        self.video_label.pack(fill="both", expand=True)

        self.control_frame = ctk.CTkFrame(self, width=280)
        self.control_frame.pack(side="right", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.control_frame, text="CONTROL PANEL", font=("Arial", 18, "bold")).pack(pady=20)

        self.btn_start = ctk.CTkButton(self.control_frame, text="Start System", command=self.start_video, height=40)
        self.btn_start.pack(pady=10, fill="x", padx=20)

        self.show_lines = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_frame, text="Lane Detection (Hough)", variable=self.show_lines).pack(pady=15, anchor="w", padx=20)

        self.show_yolo = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_frame, text="Object Detection (YOLOv8)", variable=self.show_yolo).pack(pady=15, anchor="w", padx=20)

        self.show_mog2 = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.control_frame, text="Background Subtraction (MOG2)", variable=self.show_mog2).pack(pady=15, anchor="w", padx=20)

        self.fps_label = ctk.CTkLabel(self.control_frame, text="FPS: 0", font=("Arial", 16, "bold"), text_color="#00FF00")
        self.fps_label.pack(side="bottom", pady=30)

    def start_video(self):
        if not self.running:
            self.cap = cv2.VideoCapture("dashcam.mp4")
            if not self.cap.isOpened():
                self.video_label.configure(text="Error: dashcam.mp4 not found!\nPlease run get_data.py first.")
                return
            
            self.running = True
            self.btn_start.configure(text="Stop System", fg_color="red", hover_color="#8B0000")
            self.process_frame()
        else:
            self.running = False
            self.btn_start.configure(text="Start System", fg_color=["#3a7ebf", "#1f538d"])
            if self.cap:
                self.cap.release()

    def process_frame(self):
        if not self.running: return

        start_time = time.time()
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after(10, self.process_frame)
            return

        frame = cv2.resize(frame, (720, 480))
        display_frame = frame.copy()

        if self.show_lines.get():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            roi_edges = edges[240:480, 0:720]
            lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(display_frame, (x1, y1 + 240), (x2, y2 + 240), (0, 255, 255), 2)

        if self.show_yolo.get():
            results = self.yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
            display_frame = results[0].plot()

        if self.show_mog2.get():
            fg_mask = self.mog2.apply(frame)
            red_overlay = np.zeros_like(display_frame)
            red_overlay[:, :] = [0, 0, 255]
            mask = fg_mask > 200
            display_frame[mask] = cv2.addWeighted(display_frame, 0.5, red_overlay, 0.5, 0)[mask]

        elapsed_time = time.time() - start_time
        fps = int(1.0 / elapsed_time) if elapsed_time > 0 else 0
        self.fps_label.configure(text=f"FPS: {fps}")

        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(720, 480))
        
        self.video_label.configure(image=ctk_image, text="")
        self.after(10, self.process_frame)

if __name__ == "__main__":
    app = DashcamAnalyzer()
    app.mainloop()