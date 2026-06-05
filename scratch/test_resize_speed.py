import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_FULL_PATH = "pose_landmarker_full.task"

def run_benchmark(resize_height=None, num_frames=100):
    base_options = python.BaseOptions(
        model_asset_path=MODEL_FULL_PATH,
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
    
    # Benchmark - start timestamp at 33ms to avoid conflict with warmup (0ms)
    start = time.time()
    ts = 33
    for frame in frames:
        if resize_height:
            h, w = frame.shape[:2]
            aspect = w / h
            new_w = int(resize_height * aspect)
            frame = cv2.resize(frame, (new_w, resize_height))
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        detector.detect_for_video(mp_img, ts)
        ts += int(1000 / fps)
            
    end = time.time()
    detector.close()
    
    elapsed = end - start
    fps_run = len(frames) / elapsed
    return fps_run, elapsed

print("Benchmarking resizing on 100 frames of not-catherine.mp4 (718x1280)...")
fps1, time1 = run_benchmark(resize_height=None)
print(f"1. No Resize (718x1280): {fps1:.2f} FPS ({time1:.3f}s)")

fps2, time2 = run_benchmark(resize_height=512)
print(f"2. Resize to 512h (287x512): {fps2:.2f} FPS ({time2:.3f}s)")

fps3, time3 = run_benchmark(resize_height=256)
print(f"3. Resize to 256h (143x256): {fps3:.2f} FPS ({time3:.3f}s)")
