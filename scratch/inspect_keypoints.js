import fs from 'fs';
import path from 'path';

const jsonPath = path.resolve('public/dance_3d_keypoints.json');
if (!fs.existsSync(jsonPath)) {
  console.error("JSON not found at:", jsonPath);
  process.exit(1);
}

const data = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
console.log("FPS:", data.fps);
console.log("Total Frames:", data.total_frames);

const frame0 = data.frames["0"];
if (frame0) {
  console.log("\nFrame 0 - 3D keypoints sample:");
  const sampleKeys = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "left_wrist"];
  sampleKeys.forEach(key => {
    console.log(`${key}:`, frame0.keypoints_3d[key]);
  });
} else {
  console.log("Frame 0 not found");
}
