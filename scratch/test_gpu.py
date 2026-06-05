import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("Loading model with CPU delegate...")
base_options_cpu = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task",
    delegate=python.BaseOptions.Delegate.CPU
)
options_cpu = vision.PoseLandmarkerOptions(
    base_options=base_options_cpu,
    running_mode=vision.RunningMode.VIDEO
)
detector_cpu = vision.PoseLandmarker.create_from_options(options_cpu)

cap = cv2.VideoCapture("also-not-catherine.mp4")
ret, frame = cap.read()
if ret:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Warm up CPU
    ts = 0
    for _ in range(5):
        detector_cpu.detect_for_video(mp_image, ts)
        ts += 33
        
    start = time.time()
    for i in range(30):
        detector_cpu.detect_for_video(mp_image, ts)
        ts += 33
    end = time.time()
    print(f"CPU Time for 30 frames: {end - start:.4f} seconds ({(30/(end - start)):.2f} FPS)")

detector_cpu.close()

print("\nLoading model with GPU delegate...")
try:
    base_options_gpu = python.BaseOptions(
        model_asset_path="pose_landmarker_full.task",
        delegate=python.BaseOptions.Delegate.GPU
    )
    options_gpu = vision.PoseLandmarkerOptions(
        base_options=base_options_gpu,
        running_mode=vision.RunningMode.VIDEO
    )
    detector_gpu = vision.PoseLandmarker.create_from_options(options_gpu)
    
    # Warm up GPU
    ts = 0
    for _ in range(5):
        detector_gpu.detect_for_video(mp_image, ts)
        ts += 33
        
    start = time.time()
    for i in range(30):
        detector_gpu.detect_for_video(mp_image, ts)
        ts += 33
    end = time.time()
    print(f"GPU Time for 30 frames: {end - start:.4f} seconds ({(30/(end - start)):.2f} FPS)")
    detector_gpu.close()
except Exception as e:
    print(f"GPU delegate failed: {e}")
