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

# Download the model if it doesn't exist
MODEL_PATH = "pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading Pose Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
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
    print("Initializing MediaPipe Pose Landmarker...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    raw_3d_data = []
    raw_2d_data = []
    saved_frames = [] 
    
    print("Phase 1: Extracting keypoints and caching frames...")
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
        
        saved_frames.append(frame)
        
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
    window = min(7, total_frames if total_frames % 2 != 0 else total_frames - 1)
    
    if window > 3:
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

    print("Phase 4: Generating debug playback video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
    
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # arms
        (11, 23), (12, 24), (23, 24), # torso
        (23, 25), (25, 27), (24, 26), (26, 28) # legs
    ]
    
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
    print("Processing complete. Keypoints exported to:", output_json_path)

if __name__ == "__main__":
    process_dance_video_optimized(
        "not-catherine.mp4", 
        "public/dance_3d_keypoints.json", 
        "public/debug_playback.mp4"
    )
