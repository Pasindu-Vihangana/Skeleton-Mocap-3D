# 3D Dancer Motion Sync

A high-performance, premium web dashboard that extracts 3D skeleton joints from video footage using MediaPipe Tasks and maps them in real-time onto a 3D GLTF character model (`dancer.glb`) in a Three.js scene.

---

## Architecture Overview

```
                        ┌─────────────────────────┐
                        │    Input Dance Video    │
                        │  (e.g. not-catherine)   │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │ 3d_keypoint_extractor.py│
                        │    (MediaPipe Pose)     │
                        └────────────┬────────────┘
                                     │
                                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                           Public Assets                             │
  │  - public/dance_3d_keypoints.json (Extracted coordinate telemetry) │
  │  - public/dancer.glb (Target skinned avatar)                       │
  │  - public/not-catherine.mp4 (Background sync video)                 │
  └──────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │      Three.js App       │
                        │  (FK Rotation Mapping)  │
                        └─────────────────────────┘
```

---

## System Requirements

- **Python 3.10+** (to run the MediaPipe keypoint extractor)
- **Node.js 18+** (to run the local Vite frontend development server)

---

## Step 1: Python Setup & Keypoint Extraction

This phase downloads the MediaPipe model, extracts 33 body landmarks, applies smoothing filters to eliminate jitter, and exports them to a JSON format.

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Python Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Keypoint Extractor**:
   ```bash
   python 3d_keypoint_extractor.py
   ```
   *Note: The script will automatically download the required `pose_landmarker_full.task` file on its first run, process `not-catherine.mp4` frame-by-frame, and write the output to `public/dance_3d_keypoints.json`.*

---

## Step 2: Frontend Setup & Running the Web App

This phase loads the character model and synchronizes its bone rotations with the exported keypoint database.

1. **Install Node Dependencies**:
   Ensure you are in the project root directory:
   ```bash
   npm install
   ```

2. **Start the Development Server**:
   ```bash
   npm run dev
   ```

3. **View the Application**:
   Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

---

## Production Build

To bundle the frontend assets for deployment:
```bash
npm run build
```
This generates a production-ready folder in `dist/`.

---

## Controls & Features

- **Timeline Scrubber**: Scrub at the bottom of the screen to seek to individual frames; the reference video will scrub in perfect sync with the 3D model.
- **Speed Multiplier**: Adjust speed from `0.25x` to `2.0x` for slow-motion playback analysis.
- **Reference Skeleton**: Toggle the purple 3D stick-figure in the viewport to view the raw MediaPipe coordinates side-by-side with the mapped character.
- **Bone Telemetry Panel**: Inspect real-time Euler angles (`X`, `Y`, `Z` rotations) for all 14 tracked joints as the character dances.
