import json
import cv2
import numpy as np
from scipy.signal import savgol_filter
import os
import urllib.request
import ssl
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ssl._create_default_https_context = ssl._create_unverified_context

# Tuning Parameters
MODEL_COMPLEXITY = "heavy"        # "lite", "full", or "heavy" (heavy is recommended for accurate depth)
MIN_POSE_DETECTION_CONFIDENCE = 0.6
MIN_POSE_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
SMOOTHING_WINDOW_SIZE = 11        # Window size for Savitzky-Golay filter (must be an odd number > 3)
WRITE_DEBUG_VIDEO = True          # Generate an annotated debug video (disable to save ~50% processing time)

# Map complexity to model task file
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
}

MODEL_PATH = f"pose_landmarker_{MODEL_COMPLEXITY}.task"

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
    
    raw_3d_data = []
    raw_2d_data = []
    
    print("Phase 1: Extracting keypoints (memory-efficient)...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp in ms
        timestamp_ms = int((frame_idx / fps) * 1000)
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        
        frame_3d = np.zeros((33, 4))
        frame_2d = np.zeros((33, 4))
        
        if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            world_landmarks = results.pose_world_landmarks[0]
            for i, lm in enumerate(world_landmarks):
                # lm.y coordinate increases downwards in image space. In world landmarks it's also flipped,
                # so we store -lm.y to align with Three.js Y-up.
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
    
    data_3d = np.array(raw_3d_data)
    data_2d = np.array(raw_2d_data)
    total_frames = data_3d.shape[0]
    
    print("Phase 2: Applying smoothing filters...")
    # Calculate appropriate window size (must be odd, less than total frames, and >= 3)
    window = SMOOTHING_WINDOW_SIZE
    if window >= total_frames:
        window = total_frames - 1 if total_frames % 2 == 0 else total_frames
    if window < 3:
        window = 3
    if window % 2 == 0:
        window -= 1
        
    if window > 3:
        print(f"Applying Savitzky-Golay filter with window size {window}...")
        for i in range(33):
            for j in range(3): 
                data_3d[:, i, j] = savgol_filter(data_3d[:, i, j], window, 3)
                data_2d[:, i, j] = savgol_filter(data_2d[:, i, j], window, 3)

    print("Phase 3: Saving data to JSON...")
    final_output = {"fps": fps, "total_frames": total_frames, "frames": {}}
    
    for f_idx in range(total_frames):
        keypoints_3d = {}
        keypoints_2d = {}
        
        for lm_idx, name in enumerate(POSE_LANDMARK_NAMES):
            x3, y3, z3, vis3 = data_3d[f_idx, lm_idx]
            keypoints_3d[name] = {"x": float(x3), "y": float(y3), "z": float(z3), "visibility": float(vis3)}
            
            x2, y2, z2, vis2 = data_2d[f_idx, lm_idx]
            keypoints_2d[name] = {"x": float(x2), "y": float(y2), "z": float(z2), "visibility": float(vis2)}
            
        final_output["frames"][str(f_idx)] = {
            "keypoints_3d": keypoints_3d,
            "keypoints_2d": keypoints_2d
        }
        
    # Make sure parent directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, separators=(',', ':'))

    if WRITE_DEBUG_VIDEO:
        print("Phase 4: Generating debug playback video (2-pass memory-efficient)...")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
        
        POSE_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # arms
            (11, 23), (12, 24), (23, 24), # torso
            (23, 25), (25, 27), (24, 26), (26, 28) # legs
        ]
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or idx >= total_frames:
                break
                
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
            idx += 1
            
        cap.release()
        out_video.release()
        print("Debug video generated at:", debug_video_path)
    else:
        print("Phase 4: Skipping debug playback video generation as configured.")
        
    print("Processing complete. Keypoints exported to:", output_json_path)

if __name__ == "__main__":
    process_dance_video_optimized(
        "also-not-catherine.mp4", 
        "public/dance_3d_keypoints.json", 
        "public/debug_playback.mp4"
    )

