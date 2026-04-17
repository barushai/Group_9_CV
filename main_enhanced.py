import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import time
import os
from tkinter import filedialog, messagebox
from datetime import datetime
import csv

# Force CPU usage to avoid CUDA NMS compatibility issues
torch.cuda.is_available = lambda: False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class DashcamAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Traffic Analysis System - Group 29")
        self.geometry("1100x800")

        print("Initializing AI Model (YOLOv8)...")
        self.yolo_model = YOLO("yolov8n.pt")
        self.yolo_model.to('cpu')  # Ensure model runs on CPU
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.cap = None
        self.running = False
        self.paused = False
        self.video_file = "dashcam.mp4"
        
        # Statistics tracking
        self.frame_count = 0
        self.total_frames = 0
        self.detection_stats = {2: 0, 3: 0, 5: 0, 7: 0}  # car, motorcycle, bus, truck
        self.class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        self.tracked_ids = set()
        self.detections_log = []
        
        self.setup_ui()

    def setup_ui(self):
        # Main frame with video on left and controls on right
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Press 'Select Video' to choose a file", font=("Arial", 14))
        self.video_label.pack(fill="both", expand=True)

        # Control panel on right
        self.control_frame = ctk.CTkScrollableFrame(self, width=300)
        self.control_frame.pack(side="right", fill="both", padx=10, pady=10)

        # Title
        ctk.CTkLabel(self.control_frame, text="CONTROL PANEL", font=("Arial", 16, "bold")).pack(pady=15)

        # File selection section
        ctk.CTkLabel(self.control_frame, text="Video File", font=("Arial", 12, "bold")).pack(pady=(15, 5), anchor="w", padx=15)
        
        btn_select = ctk.CTkButton(self.control_frame, text="Select Video", command=self.select_video_file, height=35)
        btn_select.pack(pady=8, fill="x", padx=15)
        
        self.video_label_text = ctk.CTkLabel(self.control_frame, text="No video selected", text_color="#FFA500", font=("Arial", 10))
        self.video_label_text.pack(pady=5, padx=15)

        # Play controls section
        ctk.CTkLabel(self.control_frame, text="Playback Controls", font=("Arial", 12, "bold")).pack(pady=(15, 8), anchor="w", padx=15)
        
        control_btn_frame = ctk.CTkFrame(self.control_frame)
        control_btn_frame.pack(pady=8, fill="x", padx=15)
        
        self.btn_start = ctk.CTkButton(control_btn_frame, text="Start", command=self.start_video, height=35, width=60)
        self.btn_start.pack(side="left", padx=3, fill="x", expand=True)
        
        self.btn_pause = ctk.CTkButton(control_btn_frame, text="Pause", command=self.pause_video, height=35, width=60, state="disabled")
        self.btn_pause.pack(side="left", padx=3, fill="x", expand=True)
        
        self.btn_stop = ctk.CTkButton(control_btn_frame, text="Stop", command=self.stop_video, height=35, width=60, state="disabled", fg_color="#DC143C")
        self.btn_stop.pack(side="left", padx=3, fill="x", expand=True)

        # Feature toggles section
        ctk.CTkLabel(self.control_frame, text="Features", font=("Arial", 12, "bold")).pack(pady=(15, 8), anchor="w", padx=15)
        
        self.show_lines = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_frame, text="Lane Detection (Hough)", variable=self.show_lines).pack(pady=6, anchor="w", padx=20)

        self.show_yolo = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_frame, text="YOLOv8 Detection", variable=self.show_yolo).pack(pady=6, anchor="w", padx=20)

        self.show_mog2 = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.control_frame, text="MOG2 Background Sub.", variable=self.show_mog2).pack(pady=6, anchor="w", padx=20)
        
        self.show_clahe = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.control_frame, text="CLAHE (Light Balance)", variable=self.show_clahe).pack(pady=6, anchor="w", padx=20)

        # Save results button
        btn_save = ctk.CTkButton(self.control_frame, text="Save Results", command=self.save_results, height=35, fg_color="#2FA572")
        btn_save.pack(pady=15, fill="x", padx=15)

        # Statistics section
        ctk.CTkLabel(self.control_frame, text="Statistics", font=("Arial", 12, "bold")).pack(pady=(15, 8), anchor="w", padx=15)
        
        stats_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        stats_frame.pack(pady=8, fill="x", padx=15)
        
        self.fps_label = ctk.CTkLabel(stats_frame, text="FPS: 0", font=("Arial", 10))
        self.fps_label.pack(anchor="w", pady=3)
        
        self.frame_label = ctk.CTkLabel(stats_frame, text="Frame: 0/0", font=("Arial", 10))
        self.frame_label.pack(anchor="w", pady=3)
        
        self.detection_label = ctk.CTkLabel(stats_frame, text="Detections: 0 | Tracked: 0", font=("Arial", 10))
        self.detection_label.pack(anchor="w", pady=3)
        
        # Class breakdown
        ctk.CTkLabel(self.control_frame, text="Detection Breakdown", font=("Arial", 10, "bold")).pack(pady=(10, 5), anchor="w", padx=15)
        
        breakdown_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        breakdown_frame.pack(pady=5, fill="x", padx=15)
        
        self.car_label = ctk.CTkLabel(breakdown_frame, text="Cars: 0", font=("Arial", 9), text_color="#90EE90")
        self.car_label.pack(anchor="w", padx=10)
        
        self.moto_label = ctk.CTkLabel(breakdown_frame, text="Motorcycles: 0", font=("Arial", 9), text_color="#FFD700")
        self.moto_label.pack(anchor="w", padx=10)
        
        self.bus_label = ctk.CTkLabel(breakdown_frame, text="Buses: 0", font=("Arial", 9), text_color="#87CEEB")
        self.bus_label.pack(anchor="w", padx=10)
        
        self.truck_label = ctk.CTkLabel(breakdown_frame, text="Trucks: 0", font=("Arial", 9), text_color="#FF69B4")
        self.truck_label.pack(anchor="w", padx=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(self.control_frame, text="Status: Ready", text_color="#90EE90", font=("Arial", 10, "bold"))
        self.status_label.pack(side="bottom", pady=15, padx=15, anchor="w")

    def select_video_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_file = filename
            self.video_label_text.configure(text=f"Selected: {os.path.basename(filename)}")
            self.status_label.configure(text="Status: Ready", text_color="#90EE90")
            self.reset_stats()
    
    def reset_stats(self):
        self.frame_count = 0
        self.total_frames = 0
        self.detection_stats = {2: 0, 3: 0, 5: 0, 7: 0}
        self.tracked_ids = set()
        self.detections_log = []
        self.update_stat_labels()
    
    def update_stat_labels(self):
        self.car_label.configure(text=f"Cars: {self.detection_stats[2]}")
        self.moto_label.configure(text=f"Motorcycles: {self.detection_stats[3]}")
        self.bus_label.configure(text=f"Buses: {self.detection_stats[5]}")
        self.truck_label.configure(text=f"Trucks: {self.detection_stats[7]}")
    
    def start_video(self):
        if not os.path.exists(self.video_file):
            messagebox.showerror("Error", f"Video file not found: {self.video_file}")
            return
        
        if not self.running:
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video: {self.video_file}")
                return
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.reset_stats()
            self.running = True
            self.paused = False
            self.btn_start.configure(state="disabled")
            self.btn_pause.configure(state="normal")
            self.btn_stop.configure(state="normal")
            self.status_label.configure(text="Status: Playing", text_color="#87CEEB")
            self.process_frame()
    
    def pause_video(self):
        if self.running:
            if self.paused:
                self.paused = False
                self.btn_pause.configure(text="Pause")
                self.status_label.configure(text="Status: Playing", text_color="#87CEEB")
                self.process_frame()
            else:
                self.paused = True
                self.btn_pause.configure(text="Resume")
                self.status_label.configure(text="Status: Paused", text_color="#FFD700")
    
    def stop_video(self):
        self.running = False
        self.paused = False
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled", text="Pause")
        self.btn_stop.configure(state="disabled")
        if self.cap:
            self.cap.release()
        self.video_label.configure(text="Video Stopped")
        self.status_label.configure(text="Status: Stopped", text_color="#FF6347")

    def process_frame(self):
        if not self.running: 
            return
        
        if self.paused:
            self.after(100, self.process_frame)
            return

        start_time = time.time()
        ret, frame = self.cap.read()
        
        if not ret:
            self.running = False
            self.status_label.configure(text="Status: Finished", text_color="#90EE90")
            self.btn_start.configure(state="normal")
            self.btn_pause.configure(state="disabled")
            self.btn_stop.configure(state="disabled")
            messagebox.showinfo("Done", f"Video processing completed!\nTotal frames: {self.frame_count}")
            return

        self.frame_count += 1
        frame = cv2.resize(frame, (720, 480))
        display_frame = frame.copy()
        
        # Apply CLAHE if enabled
        if self.show_clahe.get():
            b, g, r = cv2.split(display_frame)
            display_frame = cv2.merge([self.clahe.apply(b), self.clahe.apply(g), self.clahe.apply(r)])

        # Lane detection with Hough Lines
        if self.show_lines.get():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            roi_edges = edges[240:480, 0:720]
            lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(display_frame, (x1, y1 + 240), (x2, y2 + 240), (0, 255, 255), 2)

        detection_count = 0
        # YOLOv8 detection and tracking
        if self.show_yolo.get():
            results = self.yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
            display_frame = results[0].plot()
            
            # Update statistics
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls in self.detection_stats:
                    self.detection_stats[cls] += 1
                detection_count += 1
                
                # Track IDs
                if hasattr(box, 'id') and box.id is not None:
                    self.tracked_ids.add(int(box.id[0]))

        # MOG2 background subtraction
        if self.show_mog2.get():
            fg_mask = self.mog2.apply(frame)
            red_overlay = np.zeros_like(display_frame)
            red_overlay[:, :] = [0, 0, 255]
            mask = fg_mask > 200
            display_frame[mask] = cv2.addWeighted(display_frame, 0.5, red_overlay, 0.5, 0)[mask]

        # Update statistics labels
        elapsed_time = time.time() - start_time
        fps = int(1.0 / elapsed_time) if elapsed_time > 0 else 0
        self.fps_label.configure(text=f"FPS: {fps}")
        self.frame_label.configure(text=f"Frame: {self.frame_count}/{self.total_frames}")
        self.detection_label.configure(text=f"Detections: {detection_count} | Tracked IDs: {len(self.tracked_ids)}")
        self.update_stat_labels()

        # Display frame
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(720, 480))
        
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image  # Keep a reference
        self.after(10, self.process_frame)
    
    def save_results(self):
        if self.frame_count == 0:
            messagebox.showwarning("Warning", "No data to save. Please process a video first.")
            return
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(results_dir, f"detections_{timestamp}.csv")
        
        # Save statistics to CSV
        try:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["---", "---"])
                writer.writerow(["Total Frames Processed", self.frame_count])
                writer.writerow(["Total Frames in Video", self.total_frames])
                writer.writerow(["Total Detections", sum(self.detection_stats.values())])
                writer.writerow(["Unique Track IDs", len(self.tracked_ids)])
                writer.writerow(["---", "---"])
                
                for class_id in [2, 3, 5, 7]:
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                    count = self.detection_stats[class_id]
                    writer.writerow([f"{class_name} Detections", count])
                
                writer.writerow(["---", "---"])
                writer.writerow(["Export Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            
            messagebox.showinfo("Success", f"Results saved to:\n{csv_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    app = DashcamAnalyzer()
    app.mainloop()
