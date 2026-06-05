import os
import time
import urllib.request
import ssl
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_HEAVY_PATH = "pose_landmarker_heavy.task"
MODEL_FULL_PATH = "pose_landmarker_full.task"

if not os.path.exists(MODEL_HEAVY_PATH):
    print("Downloading Heavy model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, MODEL_HEAVY_PATH)
    print("Heavy model downloaded.")

def run_benchmark(model_path, resize_height=None, num_frames=50):
    base_options = python.BaseOptions(
        model_asset_path=model_path,
        delegate=python.BaseOptions.Delegate.CPU
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture("not-catherine.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Warm up
    if len(frames) > 0:
        frame = frames[0]
        if resize_height:
            h, w = frame.shape[:2]
            aspect = w / h
            new_w = int(resize_height * aspect)
            frame = cv2.resize(frame, (new_w, resize_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detector.detect_for_video(mp_img, 0)
    
    # Benchmark
    start = time.time()
    ts = 0
    landmarks_sample = None
    for frame in frames:
        if resize_height:
            h, w = frame.shape[:2]
            aspect = w / h
            new_w = int(resize_height * aspect)
            frame = cv2.resize(frame, (new_w, resize_height))
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = detector.detect_for_video(mp_img, ts)
        ts += int(1000 / fps)
        
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            landmarks_sample = results.pose_world_landmarks[0]
            
    end = time.time()
    detector.close()
    
    elapsed = end - start
    fps_run = len(frames) / elapsed
    return fps_run, elapsed, landmarks_sample

print("Benchmarking configurations on 50 frames of not-catherine.mp4 (718x1280)...")

# 1. Full model, no resize
fps1, time1, lm1 = run_benchmark(MODEL_FULL_PATH, resize_height=None)
print(f"1. Full Model (No Resize): {fps1:.2f} FPS ({time1:.3f}s)")

# 2. Full model, resize height=512
fps2, time2, lm2 = run_benchmark(MODEL_FULL_PATH, resize_height=512)
print(f"2. Full Model (Resize 512h): {fps2:.2f} FPS ({time2:.3f}s)")

# 3. Full model, resize height=256
fps3, time3, lm3 = run_benchmark(MODEL_FULL_PATH, resize_height=256)
print(f"3. Full Model (Resize 256h): {fps3:.2f} FPS ({time3:.3f}s)")

# 4. Heavy model, no resize
fps4, time4, lm4 = run_benchmark(MODEL_HEAVY_PATH, resize_height=None)
print(f"4. Heavy Model (No Resize): {fps4:.2f} FPS ({time4:.3f}s)")

# 5. Heavy model, resize height=512
fps5, time5, lm5 = run_benchmark(MODEL_HEAVY_PATH, resize_height=512)
print(f"5. Heavy Model (Resize 512h): {fps5:.2f} FPS ({time5:.3f}s)")

# 6. Heavy model, resize height=256
fps6, time6, lm6 = run_benchmark(MODEL_HEAVY_PATH, resize_height=256)
print(f"6. Heavy Model (Resize 256h): {fps6:.2f} FPS ({time6:.3f}s)")

# Compare landmarks (e.g. distance of landmark 11 [left shoulder] between Full no resize and Heavy no resize vs Heavy resized)
def get_xyz(lm):
    if lm is None: return (0,0,0)
    return (lm[11].x, lm[11].y, lm[11].z)

print("\nLeft Shoulder coordinates (x, y, z):")
print(f"Full Model (No Resize): {get_xyz(lm1)}")
print(f"Heavy Model (No Resize): {get_xyz(lm4)}")
print(f"Heavy Model (Resize 512h): {get_xyz(lm5)}")
print(f"Heavy Model (Resize 256h): {get_xyz(lm6)}")
