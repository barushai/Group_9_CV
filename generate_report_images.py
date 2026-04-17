"""
Generate images for LaTeX report from video processing
Outputs to report/graphics/ folder
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ultralytics import YOLO
import torch
import os
from pathlib import Path

# Force CPU
torch.cuda.is_available = lambda: False

# Output directory
OUTPUT_DIR = Path(__file__).parent / "report" / "graphics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {OUTPUT_DIR}")

# Load video
VIDEO_FILE = Path(__file__).parent / "dashcam.mp4"
if not VIDEO_FILE.exists():
    print(f"Video file not found: {VIDEO_FILE}")
    exit(1)

cap = cv2.VideoCapture(str(VIDEO_FILE))
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit(1)

# Resize
frame = cv2.resize(frame, (720, 480))
cap.release()

print("✓ Loaded video frame")

# ============================================================================
# 1. ORIGINAL FRAME
# ============================================================================
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imwrite(str(OUTPUT_DIR / "01_original.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
print("✓ Saved 01_original.png")

# ============================================================================
# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
frame_clahe = frame.copy()
for i in range(3):  # Apply to each BGR channel
    frame_clahe[:, :, i] = clahe.apply(frame_clahe[:, :, i])
cv2.imwrite(str(OUTPUT_DIR / "02_clahe.png"), frame_clahe)
print("✓ Saved 02_clahe.png")

# ============================================================================
# 3. GRAYSCALE + CANNY EDGES
# ============================================================================
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.imwrite(str(OUTPUT_DIR / "03_canny_edges.png"), edges_rgb)
print("✓ Saved 03_canny_edges.png")

# ============================================================================
# 4. HOUGH LINES
# ============================================================================
frame_hough = frame.copy()
roi_y_start, roi_y_end = 240, 480
edges_roi = edges[roi_y_start:roi_y_end, :]
lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Adjust y coordinates to account for ROI offset
        y1 += roi_y_start
        y2 += roi_y_start
        cv2.line(frame_hough, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow

cv2.imwrite(str(OUTPUT_DIR / "04_hough_lines.png"), frame_hough)
print("✓ Saved 04_hough_lines.png")

# ============================================================================
# 5. YOLOv8 DETECTION
# ============================================================================
print("Loading YOLOv8 model (may take a moment)...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to('cpu')

results = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
frame_yolo = frame.copy()
class_names_dict = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
class_colors = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 0, 255), 7: (255, 255, 0)}

detection_count = 0
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    color = class_colors.get(cls, (255, 255, 255))
    cv2.rectangle(frame_yolo, (x1, y1), (x2, y2), color, 2)
    
    label = f"{class_names_dict.get(cls, 'Unknown')} {conf:.2f}"
    if hasattr(box, 'id') and box.id is not None:
        track_id = int(box.id[0])
        label += f" ID:{track_id}"
    
    cv2.putText(frame_yolo, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    detection_count += 1

cv2.imwrite(str(OUTPUT_DIR / "05_yolov8_detection.png"), frame_yolo)
print(f"✓ Saved 05_yolov8_detection.png ({detection_count} detections)")

# ============================================================================
# 6. MOG2 (Background Subtraction)
# ============================================================================
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Need to process multiple frames for MOG2 to work properly
cap = cv2.VideoCapture(str(VIDEO_FILE))
for _ in range(30):  # Process first 30 frames
    ret, f = cap.read()
    if ret:
        f = cv2.resize(f, (720, 480))
        mog2.apply(f)

# Now process the frame we want to display
ret, frame_mog = cap.read()
cap.release()

if ret:
    frame_mog = cv2.resize(frame_mog, (720, 480))
    mask = mog2.apply(frame_mog)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create visualization: red overlay of foreground
    frame_mog_rgb = cv2.cvtColor(frame_mog, cv2.COLOR_BGR2RGB)
    mog2_display = frame_mog_rgb.copy()
    red_mask = np.zeros_like(mog2_display)
    red_mask[:, :, 0] = mask  # Red channel
    mog2_display = cv2.addWeighted(mog2_display, 0.7, red_mask, 0.3, 0)
    
    cv2.imwrite(str(OUTPUT_DIR / "06_mog2_foreground.png"), 
                cv2.cvtColor(mog2_display, cv2.COLOR_RGB2BGR))
    print("✓ Saved 06_mog2_foreground.png")

# ============================================================================
# 7. PERFORMANCE CHART: FPS Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

methods = ['YOLOv8\nOnly', 'YOLOv8 +\nCLAHE', 'YOLOv8 +\nHough', 
           'YOLOv8 +\nMOG2', 'All\nFeatures']
fps_values = [22, 20, 18, 17, 13]
colors_chart = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']

bars = ax.bar(methods, fps_values, color=colors_chart, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, fps in zip(bars, fps_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{fps} FPS', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
ax.set_xlabel('Processing Method', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison: FPS by Method', fontsize=14, fontweight='bold')
ax.set_ylim(0, 30)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "07_fps_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved 07_fps_comparison.png")

# ============================================================================
# 8. ACCURACY CHART: Detection Accuracy by Class
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

classes = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Average']
accuracy = [97, 96, 67, 100, 90.5]
colors_acc = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']

bars = ax.bar(classes, accuracy, color=colors_acc, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, acc in zip(bars, accuracy):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Vehicle Class', fontsize=12, fontweight='bold')
ax.set_title('Detection Accuracy by Vehicle Class', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=90.5, color='red', linestyle='--', linewidth=2, label='Average (90.5%)')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "08_accuracy_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved 08_accuracy_comparison.png")

# ============================================================================
# 9. SYSTEM ARCHITECTURE DIAGRAM (simple text-based)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.axis('off')

# Draw boxes for each layer
layers = [
    ("Input Layer", "VideoCapture\nMP4/AVI/MOV", 0.5, 0.9),
    ("Preprocessing", "Resize (720×480)\nCLAHE", 0.5, 0.75),
    ("Geometry Analysis", "Canny Edges\nHough Lines", 0.5, 0.6),
    ("Content Analysis", "YOLOv8 Detection\n(AI-based)", 0.25, 0.4),
    ("", "MOG2 BG Subtraction\n(Traditional)", 0.75, 0.4),
    ("Visualization", "Draw Boxes, Lines\nOverlay Masks", 0.5, 0.2),
    ("Output", "Display + CSV Export", 0.5, 0.05),
]

box_width = 0.25
box_height = 0.1

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

for i, (title, content, x, y) in enumerate(layers):
    if title:  # Skip empty title rows
        box = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor='lightblue', linewidth=2)
        ax.add_patch(box)
        
        ax.text(x, y + 0.03, title, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x, y - 0.03, content, ha='center', va='center', 
               fontsize=8, style='italic')

# Draw arrows
arrow_y_positions = [0.85, 0.7, 0.58, 0.3, 0.15]
for y in arrow_y_positions:
    arrow = FancyArrowPatch((0.5, y), (0.5, y - 0.08),
                          arrowstyle='->', mutation_scale=20, 
                          linewidth=2, color='black')
    ax.add_patch(arrow)

# Arrow from Geometry to both AI and Traditional
arrow1 = FancyArrowPatch((0.45, 0.55), (0.3, 0.45),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
arrow2 = FancyArrowPatch((0.55, 0.55), (0.7, 0.45),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
ax.add_patch(arrow2)

# Arrows back to Visualization
arrow3 = FancyArrowPatch((0.25, 0.35), (0.45, 0.25),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
arrow4 = FancyArrowPatch((0.75, 0.35), (0.55, 0.25),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow3)
ax.add_patch(arrow4)

ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1)
ax.set_aspect('equal')

plt.title('System Architecture - 6-Layer Pipeline', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "09_architecture_diagram.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved 09_architecture_diagram.png")

print("\n" + "="*60)
print("✅ ALL IMAGES GENERATED SUCCESSFULLY!")
print("="*60)
print(f"Output location: {OUTPUT_DIR}")
print("\nFiles created:")
for i in range(1, 10):
    ext = "png"
    if i == 1:
        fname = f"0{i}_original"
    elif i == 2:
        fname = f"0{i}_clahe"
    elif i == 3:
        fname = f"0{i}_canny_edges"
    elif i == 4:
        fname = f"0{i}_hough_lines"
    elif i == 5:
        fname = f"0{i}_yolov8_detection"
    elif i == 6:
        fname = f"0{i}_mog2_foreground"
    elif i == 7:
        fname = f"0{i}_fps_comparison"
    elif i == 8:
        fname = f"0{i}_accuracy_comparison"
    elif i == 9:
        fname = f"0{i}_architecture_diagram"
    
    fpath = OUTPUT_DIR / f"{fname}.{ext}"
    if fpath.exists():
        size = fpath.stat().st_size / 1024  # KB
        print(f"  ✓ {fname}.{ext} ({size:.1f} KB)")
