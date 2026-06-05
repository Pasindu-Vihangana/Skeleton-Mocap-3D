import time
import cv2
import numpy as np
import os
from scipy.signal import savgol_filter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"
video_path = "also-not-catherine.mp4"
debug_video_path = "scratch/debug_playback_test.mp4"

print("Starting pipeline test...")
start_total = time.time()

# Phase 0: Init
t0 = time.time()
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)
print(f"Init took: {time.time() - t0:.4f} seconds")

# Phase 1: Inference
t1 = time.time()
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

raw_3d_data = []
raw_2d_data = []
saved_frames = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_idx / fps) * 1000)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    saved_frames.append(frame)
    
    frame_3d = np.zeros((33, 4))
    frame_2d = np.zeros((33, 4))
    if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
        world_landmarks = results.pose_world_landmarks[0]
        for i, lm in enumerate(world_landmarks):
            frame_3d[i] = [lm.x, -lm.y, lm.z, lm.visibility]
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        image_landmarks = results.pose_landmarks[0]
        for i, lm in enumerate(image_landmarks):
            frame_2d[i] = [lm.x, lm.y, lm.z, lm.visibility]
    raw_3d_data.append(frame_3d)
    raw_2d_data.append(frame_2d)
    frame_idx += 1
cap.release()
detector.close()
p1_time = time.time() - t1
print(f"Phase 1 (Inference on {frame_idx} frames) took: {p1_time:.4f} seconds ({frame_idx / p1_time:.2f} FPS)")

# Phase 2: Smoothing
t2 = time.time()
data_3d = np.array(raw_3d_data)
data_2d = np.array(raw_2d_data)
window = min(7, frame_idx if frame_idx % 2 != 0 else frame_idx - 1)
if window > 3:
    for i in range(33):
        for j in range(3):
            data_3d[:, i, j] = savgol_filter(data_3d[:, i, j], window, 3)
            data_2d[:, i, j] = savgol_filter(data_2d[:, i, j], window, 3)
print(f"Phase 2 (Smoothing) took: {time.time() - t2:.4f} seconds")

# Phase 3: JSON Save
t3 = time.time()
final_output = {"fps": fps, "total_frames": frame_idx, "frames": {}}
for f_idx in range(frame_idx):
    keypoints_3d = {}
    keypoints_2d = {}
    for lm_idx in range(33):
        name = str(lm_idx)
        x3, y3, z3, vis3 = data_3d[f_idx, lm_idx]
        keypoints_3d[name] = {"x": float(x3), "y": float(y3), "z": float(z3), "visibility": float(vis3)}
        x2, y2, z2, vis2 = data_2d[f_idx, lm_idx]
        keypoints_2d[name] = {"x": float(x2), "y": float(y2), "z": float(z2), "visibility": float(vis2)}
    final_output["frames"][str(f_idx)] = {"keypoints_3d": keypoints_3d, "keypoints_2d": keypoints_2d}
print(f"Phase 3 (JSON build) took: {time.time() - t3:.4f} seconds")

# Phase 4: Video Write
t4 = time.time()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
for idx, frame in enumerate(saved_frames):
    annotated_frame = frame.copy()
    for conn in POSE_CONNECTIONS:
        p1, p2 = conn
        x1, y1 = int(data_2d[idx, p1, 0] * width), int(data_2d[idx, p1, 1] * height)
        x2, y2 = int(data_2d[idx, p2, 0] * width), int(data_2d[idx, p2, 1] * height)
        if data_2d[idx, p1, 3] > 0.4 and data_2d[idx, p2, 3] > 0.4:
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i in range(33):
        x, y = int(data_2d[idx, i, 0] * width), int(data_2d[idx, i, 1] * height)
        if data_2d[idx, i, 3] > 0.4:
            cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
    out_video.write(annotated_frame)
out_video.release()
print(f"Phase 4 (Video Write) took: {time.time() - t4:.4f} seconds")

print(f"Total time: {time.time() - start_total:.4f} seconds")
