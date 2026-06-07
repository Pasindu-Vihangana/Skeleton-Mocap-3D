import json
import cv2
import numpy as np
import os
import urllib.request
import ssl
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ssl._create_default_https_context = ssl._create_unverified_context

# Tuning Parameters
MODEL_COMPLEXITY = "full"        # "lite", "full", or "heavy" (heavy is recommended for accurate depth)
MIN_POSE_DETECTION_CONFIDENCE = 0.6
MIN_POSE_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# One Euro Filter parameters
# mincutoff: Decrease to reduce jitter during slow movements/still poses
# beta: Increase to reduce lag/drag during fast movements
ONE_EURO_MIN_CUTOFF = 0.5         # default 0.5 Hz
ONE_EURO_BETA = 0.05              # default 0.05
ONE_EURO_DCUTOFF = 1.0            # default 1.0 Hz (dcutoff filters speed estimates)

WRITE_DEBUG_VIDEO = True          # Generate an annotated debug video (disable to save ~50% processing time)

# Map complexity to model task file
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
}

MODEL_PATH = "public/pose_landmarker_full.task"

if MODEL_COMPLEXITY not in MODEL_URLS:
    raise ValueError(f"Invalid MODEL_COMPLEXITY: {MODEL_COMPLEXITY}. Choose from 'lite', 'full', 'heavy'.")

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"Downloading Pose Landmarker model ({MODEL_COMPLEXITY})...")
    url = MODEL_URLS[MODEL_COMPLEXITY]
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully.")

POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

class OneEuroFilter:
    """One Euro Filter algorithm for real-time noise reduction in human tracking signals."""
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, x, timestamp):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = timestamp
            return x

        t_e = timestamp - self.t_prev
        self.t_prev = timestamp

        if t_e <= 0:
            return self.x_prev

        # 1. Filter derivative (velocity) to estimate speed
        a_d = self._smoothing_factor(t_e, self.dcutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)

        # 2. Adaptive cutoff frequency increases as velocity increases
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        
        # 3. Filter coordinates
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

def process_dance_video_optimized(video_path, output_json_path, debug_video_path):
    print(f"Initializing MediaPipe Pose Landmarker ({MODEL_COMPLEXITY} model)...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Instantiate One Euro filters for both 3D world landmarks and 2D normalized coordinates
    # We create a filter list for (x, y, z) coordinates for all 33 landmark joints
    filters_3d = [[OneEuroFilter(freq=fps, mincutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, dcutoff=ONE_EURO_DCUTOFF) for _ in range(3)] for _ in range(33)]
    filters_2d = [[OneEuroFilter(freq=fps, mincutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA, dcutoff=ONE_EURO_DCUTOFF) for _ in range(3)] for _ in range(33)]

    # Initialize video output if requested
    out_video = None
    if WRITE_DEBUG_VIDEO:
        print("Initializing debug video writer...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Ensure parent dir exists
        os.makedirs(os.path.dirname(debug_video_path), exist_ok=True)
        out_video = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))

    final_output = {"fps": fps, "total_frames": 0, "frames": {}}
    
    print("Processing video and smoothing keypoints in a single pass...")
    frame_idx = 0
    
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # arms
        (11, 23), (12, 24), (23, 24), # torso
        (23, 25), (25, 27), (24, 26), (26, 28) # legs
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in ms
        timestamp_ms = int((frame_idx / fps) * 1000)
        timestamp_sec = frame_idx / fps
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        keypoints_3d = {}
        keypoints_2d = {}
        
        # 1. Process and smooth 3D world landmarks
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_landmarks = results.pose_world_landmarks[0]
            for i, name in enumerate(POSE_LANDMARK_NAMES):
                lm = world_landmarks[i]
                
                # lm.y coordinate increases downwards in image space. In world landmarks it's also flipped,
                # so we store -lm.y to align with Three.js Y-up.
                x_val = filters_3d[i][0](lm.x, timestamp_sec)
                y_val = filters_3d[i][1](-lm.y, timestamp_sec)
                z_val = filters_3d[i][2](lm.z, timestamp_sec)
                
                keypoints_3d[name] = {"x": float(x_val), "y": float(y_val), "z": float(z_val), "visibility": float(lm.visibility)}
        else:
            # Fallback if no detection
            for name in POSE_LANDMARK_NAMES:
                keypoints_3d[name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
                
        # 2. Process and smooth 2D image coordinates
        draw_coords = [] # store smoothed (x, y) coordinates for drawing
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            image_landmarks = results.pose_landmarks[0]
            for i, name in enumerate(POSE_LANDMARK_NAMES):
                lm = image_landmarks[i]
                
                x_val = filters_2d[i][0](lm.x, timestamp_sec)
                y_val = filters_2d[i][1](lm.y, timestamp_sec)
                z_val = filters_2d[i][2](lm.z, timestamp_sec)
                
                keypoints_2d[name] = {"x": float(x_val), "y": float(y_val), "z": float(z_val), "visibility": float(lm.visibility)}
                draw_coords.append((int(x_val * width), int(y_val * height), lm.visibility))
        else:
            # Fallback if no detection
            for name in POSE_LANDMARK_NAMES:
                keypoints_2d[name] = {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
            draw_coords = [(0, 0, 0.0)] * 33

        # 3. Add to output
        final_output["frames"][str(frame_idx)] = {
            "keypoints_3d": keypoints_3d,
            "keypoints_2d": keypoints_2d
        }
        
        # 4. Generate annotated debug frame and write to debug video
        if out_video is not None:
            annotated_frame = frame.copy()
            
            # Draw skeleton connection lines
            for conn in POSE_CONNECTIONS:
                p1, p2 = conn
                x1, y1, vis1 = draw_coords[p1]
                x2, y2, vis2 = draw_coords[p2]
                if vis1 > 0.4 and vis2 > 0.4:
                    cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            # Draw joint circles
            for i, (x, y, vis) in enumerate(draw_coords):
                if vis > 0.4:
                    cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
                    
            out_video.write(annotated_frame)
            
        frame_idx += 1
        
    cap.release()
    detector.close()
    
    if out_video is not None:
        out_video.release()
        print("Debug video generated at:", debug_video_path)
        
    final_output["total_frames"] = frame_idx
    
    # Make sure parent directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, separators=(',', ':'))

    print("Processing complete. Keypoints exported to:", output_json_path)

if __name__ == "__main__":
    process_dance_video_optimized(
        "public/videos/not-catherine.mp4", 
        "public/dance_3d_keypoints.json", 
        "public/debug_playback.mp4"
    )
