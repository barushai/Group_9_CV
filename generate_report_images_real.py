"""
Generate images for LaTeX report with REAL data from video processing
Outputs to report/graphics/ folder
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import time
from pathlib import Path
from collections import defaultdict

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

print("\n" + "="*60)
print("PROCESSING VIDEO FOR REAL DATA...")
print("="*60)

# Open video
cap = cv2.VideoCapture(str(VIDEO_FILE))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

print(f"Total frames in video: {total_frames}")
print(f"Video FPS: {fps_video}")

# Get first frame for image processing examples
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video")
    exit(1)

# Resize
first_frame = cv2.resize(first_frame, (720, 480))

# ============================================================================
# 1. ORIGINAL FRAME
# ============================================================================
frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
cv2.imwrite(str(OUTPUT_DIR / "01_original.png"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
print("✓ Saved 01_original.png")

# ============================================================================
# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ============================================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
frame_clahe = first_frame.copy()
for i in range(3):  # Apply to each BGR channel
    frame_clahe[:, :, i] = clahe.apply(frame_clahe[:, :, i])
cv2.imwrite(str(OUTPUT_DIR / "02_clahe.png"), frame_clahe)
print("✓ Saved 02_clahe.png")

# ============================================================================
# 3. GRAYSCALE + CANNY EDGES
# ============================================================================
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv2.imwrite(str(OUTPUT_DIR / "03_canny_edges.png"), edges_rgb)
print("✓ Saved 03_canny_edges.png")

# ============================================================================
# 4. HOUGH LINES
# ============================================================================
frame_hough = first_frame.copy()
roi_y_start, roi_y_end = 240, 480
edges_roi = edges[roi_y_start:roi_y_end, :]
lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1 += roi_y_start
        y2 += roi_y_start
        cv2.line(frame_hough, (x1, y1), (x2, y2), (0, 255, 255), 2)

cv2.imwrite(str(OUTPUT_DIR / "04_hough_lines.png"), frame_hough)
print("✓ Saved 04_hough_lines.png")

# ============================================================================
# 5. YOLOv8 DETECTION
# ============================================================================
print("\nLoading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to('cpu')

results = yolo_model.track(first_frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
frame_yolo = first_frame.copy()
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
print(f"✓ Saved 05_yolov8_detection.png ({detection_count} detections in first frame)")

# ============================================================================
# 6. MOG2 (Background Subtraction)
# ============================================================================
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Process first 30 frames for MOG2 training
for _ in range(30):
    ret, f = cap.read()
    if ret:
        f = cv2.resize(f, (720, 480))
        mog2.apply(f)

# Get a frame with motion
ret, frame_mog = cap.read()
if ret:
    frame_mog = cv2.resize(frame_mog, (720, 480))
    mask = mog2.apply(frame_mog)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    frame_mog_rgb = cv2.cvtColor(frame_mog, cv2.COLOR_BGR2RGB)
    mog2_display = frame_mog_rgb.copy()
    red_mask = np.zeros_like(mog2_display)
    red_mask[:, :, 0] = mask
    mog2_display = cv2.addWeighted(mog2_display, 0.7, red_mask, 0.3, 0)
    
    cv2.imwrite(str(OUTPUT_DIR / "06_mog2_foreground.png"), 
                cv2.cvtColor(mog2_display, cv2.COLOR_RGB2BGR))
    print("✓ Saved 06_mog2_foreground.png")

# ============================================================================
# 7. FPS COMPARISON - REAL DATA from video processing
# ============================================================================
print("\nProcessing video to measure real FPS...")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
mog2_test = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

fps_measurements = {
    'YOLOv8_only': [],
    'CLAHE': [],
    'Hough': [],
    'MOG2': [],
    'All': []
}

frame_count = 0
max_frames_to_process = min(100, total_frames)  # Process max 100 frames for speed

while frame_count < max_frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (720, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. YOLOv8 only
    start = time.time()
    _ = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
    fps_measurements['YOLOv8_only'].append(1.0 / (time.time() - start))
    
    # 2. YOLOv8 + CLAHE
    start = time.time()
    frame_clahe = frame.copy()
    for i in range(3):
        frame_clahe[:, :, i] = clahe.apply(frame_clahe[:, :, i])
    _ = yolo_model.track(frame_clahe, persist=True, verbose=False, classes=[2, 3, 5, 7])
    fps_measurements['CLAHE'].append(1.0 / (time.time() - start))
    
    # 3. YOLOv8 + Hough
    start = time.time()
    edges = cv2.Canny(gray, 50, 150)
    edges_roi = edges[240:480, :]
    _ = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
    _ = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
    fps_measurements['Hough'].append(1.0 / (time.time() - start))
    
    # 4. YOLOv8 + MOG2
    start = time.time()
    _ = mog2_test.apply(frame)
    _ = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
    fps_measurements['MOG2'].append(1.0 / (time.time() - start))
    
    # 5. All features
    start = time.time()
    frame_clahe = frame.copy()
    for i in range(3):
        frame_clahe[:, :, i] = clahe.apply(frame_clahe[:, :, i])
    edges = cv2.Canny(gray, 50, 150)
    edges_roi = edges[240:480, :]
    _ = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 50, minLineLength=40, maxLineGap=20)
    _ = mog2_test.apply(frame)
    _ = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
    fps_measurements['All'].append(1.0 / (time.time() - start))
    
    frame_count += 1
    if frame_count % 20 == 0:
        print(f"  Processed {frame_count}/{max_frames_to_process} frames...")

# Calculate average FPS
avg_fps = {k: np.mean(v) for k, v in fps_measurements.items()}

print("\nAverage FPS measured:")
for method, fps in avg_fps.items():
    print(f"  {method}: {fps:.1f} FPS")

# Create FPS chart from REAL data
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

methods = ['YOLOv8\nOnly', 'YOLOv8 +\nCLAHE', 'YOLOv8 +\nHough', 
           'YOLOv8 +\nMOG2', 'All\nFeatures']
fps_values = [
    avg_fps['YOLOv8_only'],
    avg_fps['CLAHE'],
    avg_fps['Hough'],
    avg_fps['MOG2'],
    avg_fps['All']
]
colors_chart = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']

bars = ax.bar(methods, fps_values, color=colors_chart, edgecolor='black', linewidth=1.5)

for bar, fps in zip(bars, fps_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{fps:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
ax.set_xlabel('Processing Method', fontsize=12, fontweight='bold')
ax.set_title('Real Performance Measurement: FPS by Method', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(fps_values) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "07_fps_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved 07_fps_comparison.png (REAL DATA from video)")

# ============================================================================
# 8. ACCURACY - Count detections across all frames
# ============================================================================
print("\nProcessing all frames for detection statistics...")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
detection_stats = {2: 0, 3: 0, 5: 0, 7: 0}  # car, motorcycle, bus, truck
class_counts = {2: [], 3: [], 5: [], 7: []}
total_frames_processed = 0

mog2_accuracy = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while total_frames_processed < max_frames_to_process:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (720, 480))
    
    # MOG2 training
    _ = mog2_accuracy.apply(frame)
    
    # YOLOv8 detection
    results = yolo_model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7])
    
    frame_class_count = {2: 0, 3: 0, 5: 0, 7: 0}
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detection_stats[cls] += 1
        frame_class_count[cls] += 1
    
    for cls in [2, 3, 5, 7]:
        class_counts[cls].append(frame_class_count[cls])
    
    total_frames_processed += 1
    if total_frames_processed % 25 == 0:
        print(f"  Processed {total_frames_processed}/{max_frames_to_process} frames...")

# Calculate precision: frames with detection / total frames
precision_by_class = {}
class_names_full = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

for cls in [2, 3, 5, 7]:
    frames_with_detection = sum(1 for count in class_counts[cls] if count > 0)
    precision = (frames_with_detection / total_frames_processed * 100) if total_frames_processed > 0 else 0
    precision_by_class[cls] = precision
    print(f"  {class_names_full[cls]}: {frames_with_detection}/{total_frames_processed} frames = {precision:.1f}%")

# Calculate average precision
avg_precision = np.mean(list(precision_by_class.values()))

# Create accuracy chart from REAL data
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)

classes = [class_names_full[c] for c in [2, 3, 5, 7]] + ['Average']
accuracy = [precision_by_class[c] for c in [2, 3, 5, 7]] + [avg_precision]
colors_acc = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']

bars = ax.bar(classes, accuracy, color=colors_acc, edgecolor='black', linewidth=1.5)

for bar, acc in zip(bars, accuracy):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Vehicle Class', fontsize=12, fontweight='bold')
ax.set_title('Real Detection Rate: Frames with Detections by Class', fontsize=14, fontweight='bold')
ax.set_ylim(0, 110)
ax.axhline(y=avg_precision, color='red', linestyle='--', linewidth=2, label=f'Average ({avg_precision:.1f}%)')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "08_accuracy_comparison.png"), dpi=100, bbox_inches='tight')
plt.close()
print("✓ Saved 08_accuracy_comparison.png (REAL DATA from video)")

# ============================================================================
# 9. SYSTEM ARCHITECTURE DIAGRAM
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.axis('off')

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
    if title:
        box = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.01", 
                            edgecolor='black', facecolor='lightblue', linewidth=2)
        ax.add_patch(box)
        
        ax.text(x, y + 0.03, title, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        ax.text(x, y - 0.03, content, ha='center', va='center', 
               fontsize=8, style='italic')

arrow_y_positions = [0.85, 0.7, 0.58, 0.3, 0.15]
for y in arrow_y_positions:
    arrow = FancyArrowPatch((0.5, y), (0.5, y - 0.08),
                          arrowstyle='->', mutation_scale=20, 
                          linewidth=2, color='black')
    ax.add_patch(arrow)

arrow1 = FancyArrowPatch((0.45, 0.55), (0.3, 0.45),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
arrow2 = FancyArrowPatch((0.55, 0.55), (0.7, 0.45),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)
ax.add_patch(arrow2)

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

cap.release()

print("\n" + "="*60)
print("✅ ALL IMAGES GENERATED WITH REAL DATA!")
print("="*60)
print(f"Output location: {OUTPUT_DIR}")
print(f"\nFrames processed: {total_frames_processed}")
print(f"Total detections across all frames: {sum(detection_stats.values())}")
print(f"Detection breakdown: Car={detection_stats[2]}, Motorcycle={detection_stats[3]}, Bus={detection_stats[5]}, Truck={detection_stats[7]}")
print(f"\nAverage FPS by method:")
for method, fps_list in fps_measurements.items():
    print(f"  {method}: {np.mean(fps_list):.2f} FPS")
