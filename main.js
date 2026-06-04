import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// DOM Elements
const systemStatusEl = document.getElementById('system-status');
const btnPlayPause = document.getElementById('btn-play-pause');
const playIcon = document.getElementById('play-icon');
const btnReset = document.getElementById('btn-reset');
const speedButtons = document.querySelectorAll('.speed-btn');
const chkGrid = document.getElementById('chk-grid');
const chkBones = document.getElementById('chk-bones');
const chkMpHelper = document.getElementById('chk-mp-helper');
const chkAutoCamera = document.getElementById('chk-auto-camera');
const refVideo = document.getElementById('ref-video');
const videoOverlayStatus = document.getElementById('video-overlay-status');
const threeContainer = document.getElementById('three-container');
const hudFrame = document.getElementById('hud-frame');
const hudTotal = document.getElementById('hud-total');
const hudFps = document.getElementById('hud-fps');
const boneTelemetryList = document.getElementById('bone-telemetry-list');
const timelineSlider = document.getElementById('timeline-slider');
const timelineProgress = document.getElementById('timeline-progress');
const timeCurrent = document.getElementById('time-current');
const timeTotal = document.getElementById('time-total');

// Playback State
let keypointData = null;
let totalFrames = 0;
let currentFrame = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let fps = 30;
let lastFrameTime = 0;
let frameDurationMs = 1000 / 30;

// Three.js State
let renderer, camera, scene, controls, clock;
let floorGrid;
let dancerModel = null;
let skeletonHelper = null;
let skeletonBones = {};
let initialHipsPos = new THREE.Vector3();
let initialBoneRotations = {};
let initialBoneQuaternions = {};
let allBonesSorted = [];
let firstFrameMidHip = null;

// Rest-pose orientation for the hips (computed from bone world positions at load time)
let q_hips_rest_world = new THREE.Quaternion();
let q_hips_rest_local = new THREE.Quaternion();

// Reference Skeleton (MediaPipe Raw Points)
let mpSkeletonGroup = null;
let mpJointSpheres = {};
let mpConnectionLines = null;
let mpLineGeometry = null;

// Scale Factors
const positionScale = 1.0; // scale MediaPipe hips translations
const skeletonScale = 1.0; // scale reference skeleton
const refSkeletonOffset = new THREE.Vector3(1.2, 0, 0); // side-by-side display offset

// Mixamo Bone Names
const BONE_NAMES = [
  'mixamorig2Hips',
  'mixamorig2Spine',
  'mixamorig2RightShoulder',
  'mixamorig2RightArm',
  'mixamorig2RightForeArm',
  'mixamorig2RightHand',
  'mixamorig2LeftShoulder',
  'mixamorig2LeftArm',
  'mixamorig2LeftForeArm',
  'mixamorig2LeftHand',
  'mixamorig2RightUpLeg',
  'mixamorig2RightLeg',
  'mixamorig2LeftUpLeg',
  'mixamorig2LeftLeg'
];

// MediaPipe to Mixamo Joint Mapping Configurations
// childBoneName points to the child joint in the skeleton hierarchy.
// defaultDir will be dynamically computed at runtime based on child positions.
const BONE_MAPPINGS = {
  'mixamorig2Spine': {
    parentKp: 'mid_hip',
    childKp: 'mid_shoulder',
    childBoneName: 'mixamorig2Spine1'
  },
  'mixamorig2LeftShoulder': {
    parentKp: 'mid_shoulder',
    childKp: 'left_shoulder',
    childBoneName: 'mixamorig2LeftArm'
  },
  'mixamorig2LeftArm': {
    parentKp: 'left_shoulder',
    childKp: 'left_elbow',
    childBoneName: 'mixamorig2LeftForeArm'
  },
  'mixamorig2LeftForeArm': {
    parentKp: 'left_elbow',
    childKp: 'left_wrist',
    childBoneName: 'mixamorig2LeftHand'
  },
  'mixamorig2LeftHand': {
    parentKp: 'left_wrist',
    childKp: 'left_index',
    childBoneName: 'mixamorig2LeftHandIndex1'
  },
  'mixamorig2RightShoulder': {
    parentKp: 'mid_shoulder',
    childKp: 'right_shoulder',
    childBoneName: 'mixamorig2RightArm'
  },
  'mixamorig2RightArm': {
    parentKp: 'right_shoulder',
    childKp: 'right_elbow',
    childBoneName: 'mixamorig2RightForeArm'
  },
  'mixamorig2RightForeArm': {
    parentKp: 'right_elbow',
    childKp: 'right_wrist',
    childBoneName: 'mixamorig2RightHand'
  },
  'mixamorig2RightHand': {
    parentKp: 'right_wrist',
    childKp: 'right_index',
    childBoneName: 'mixamorig2RightHandIndex1'
  },
  'mixamorig2LeftUpLeg': {
    parentKp: 'left_hip',
    childKp: 'left_knee',
    childBoneName: 'mixamorig2LeftLeg'
  },
  'mixamorig2LeftLeg': {
    parentKp: 'left_knee',
    childKp: 'left_ankle',
    childBoneName: 'mixamorig2LeftFoot'
  },
  'mixamorig2RightUpLeg': {
    parentKp: 'right_hip',
    childKp: 'right_knee',
    childBoneName: 'mixamorig2RightLeg'
  },
  'mixamorig2RightLeg': {
    parentKp: 'right_knee',
    childKp: 'right_ankle',
    childBoneName: 'mixamorig2RightFoot'
  }
};

// Heirarchical update order: top down
const BONE_UPDATE_ORDER = [
  'mixamorig2Spine',
  'mixamorig2LeftShoulder',
  'mixamorig2LeftArm',
  'mixamorig2LeftForeArm',
  'mixamorig2LeftHand',
  'mixamorig2RightShoulder',
  'mixamorig2RightArm',
  'mixamorig2RightForeArm',
  'mixamorig2RightHand',
  'mixamorig2LeftUpLeg',
  'mixamorig2LeftLeg',
  'mixamorig2RightUpLeg',
  'mixamorig2RightLeg'
];

// MediaPipe Connection Lines mapping
const SKELETON_CONNECTIONS = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle']
];

// Map MediaPipe Landmarks list
const MP_LANDMARKS = [
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
  'left_index', 'right_index'
];

/* Initialize Scene */
function initThree() {
  const width = threeContainer.clientWidth;
  const height = threeContainer.clientHeight;

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  threeContainer.appendChild(renderer.domElement);

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050508);
  scene.fog = new THREE.FogExp2(0x050508, 0.15);

  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(0, 1.6, 3.5);

  // Controls
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't orbit below ground
  controls.minDistance = 1;
  controls.maxDistance = 15;
  controls.target.set(0, 0.9, 0);

  // Clock
  clock = new THREE.Clock();

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xaaccff, 0.8);
  dirLight.position.set(5, 10, 7);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.bias = -0.001;
  scene.add(dirLight);

  const spotLight = new THREE.SpotLight(0xaa88ff, 4, 15, Math.PI / 6, 0.5, 1);
  spotLight.position.set(-3, 8, 3);
  spotLight.target.position.set(0, 0.5, 0);
  spotLight.castShadow = true;
  scene.add(spotLight);
  scene.add(spotLight.target);

  // Neon Floor Grid
  createFloorGrid();

  // Create Reference Skeleton Group
  createReferenceSkeleton();

  // Window Resize
  window.addEventListener('resize', onWindowResize);
}

function createFloorGrid() {
  const gridHelper = new THREE.GridHelper(20, 40, 0x00ffff, 0x111122);
  gridHelper.position.y = 0;
  
  // Custom neon material for grid lines
  gridHelper.material.opacity = 0.25;
  gridHelper.material.transparent = true;
  
  floorGrid = gridHelper;
  scene.add(floorGrid);

  // Glowing base ring
  const ringGeo = new THREE.RingGeometry(1.8, 1.82, 64);
  const ringMat = new THREE.MeshBasicMaterial({ 
    color: 0x8800ff, 
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.2
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.005;
  scene.add(ring);
}

/* Create MediaPipe raw visual skeleton helper */
function createReferenceSkeleton() {
  mpSkeletonGroup = new THREE.Group();
  mpSkeletonGroup.position.copy(refSkeletonOffset);
  scene.add(mpSkeletonGroup);

  // Joint Spheres
  const sphereGeo = new THREE.SphereGeometry(0.03, 16, 16);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0xbb00ff });

  MP_LANDMARKS.forEach((name) => {
    const mesh = new THREE.Mesh(sphereGeo, sphereMat);
    mesh.visible = false;
    mpSkeletonGroup.add(mesh);
    mpJointSpheres[name] = mesh;
  });

  // Lines
  const lineMat = new THREE.LineBasicMaterial({ 
    color: 0x00ffcc,
    transparent: true,
    opacity: 0.7,
    linewidth: 2 
  });
  
  const positions = new Float32Array(SKELETON_CONNECTIONS.length * 2 * 3);
  mpLineGeometry = new THREE.BufferGeometry();
  mpLineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  
  mpConnectionLines = new THREE.LineSegments(mpLineGeometry, lineMat);
  mpSkeletonGroup.add(mpConnectionLines);
}

/* Load 3D Model */
function loadDancerModel() {
  setStatus('Loading GLTF Model...');
  const loader = new GLTFLoader();
  
  loader.load('/dancer.glb', 
    (gltf) => {
      dancerModel = gltf.scene;
      dancerModel.scale.set(0.01, 0.01, 0.01); // Scale model from centimeters to meters
      
      // Apply materials and shadows to meshes
      dancerModel.traverse((node) => {
        if (node.isMesh) {
          node.castShadow = true;
          node.receiveShadow = true;
          if (node.material) {
            node.material.roughness = 0.4;
            node.material.metalness = 0.1;
          }
        }
      });

      // Find active skinned mesh skeleton to index bones.
      // This protects against duplicate bone/dummy nodes in the model.
      let skinnedMesh = null;
      dancerModel.traverse((node) => {
        if (node.isSkinnedMesh && !skinnedMesh) {
          skinnedMesh = node;
        }
      });

      if (skinnedMesh && skinnedMesh.skeleton) {
        console.log("Skinning skeleton found. Indexing bones from:", skinnedMesh.name);
        skinnedMesh.skeleton.bones.forEach((bone) => {
          skeletonBones[bone.name] = bone;
          initialBoneRotations[bone.name] = bone.rotation.clone();
          initialBoneQuaternions[bone.name] = bone.quaternion.clone();
        });
      }

      // Also traverse the full model hierarchy to find bones that exist in the
      // scene graph but aren't bound to the skinned mesh (e.g. lower legs, feet, hands).
      // These bones are still needed for animation even if they don't directly deform vertices.
      dancerModel.traverse((node) => {
        if (node.isBone && !skeletonBones[node.name]) {
          skeletonBones[node.name] = node;
          initialBoneRotations[node.name] = node.rotation.clone();
          initialBoneQuaternions[node.name] = node.quaternion.clone();
          console.log('Found additional bone via traversal:', node.name);
        }
      });

      console.log('Total bones indexed:', Object.keys(skeletonBones).length, Object.keys(skeletonBones));

      // Depth-sort skeleton bones
      function getBoneDepth(bone) {
        let depth = 0;
        let p = bone.parent;
        while (p) {
          depth++;
          p = p.parent;
        }
        return depth;
      }
      allBonesSorted = Object.values(skeletonBones).sort((a, b) => getBoneDepth(a) - getBoneDepth(b));

      // Find hips and save initial state
      const hips = skeletonBones['mixamorig2Hips'];
      if (hips) {
        initialHipsPos.copy(hips.position);
      }

      // Add to scene
      scene.add(dancerModel);
      dancerModel.updateMatrixWorld(true);

      // Compute dynamic rest-pose directions (restDirWorld) and world rotations (restQuatWorld) for mapped joints
      Object.keys(BONE_MAPPINGS).forEach(boneName => {
        const mapping = BONE_MAPPINGS[boneName];
        const bone = skeletonBones[boneName];
        const childBone = skeletonBones[mapping.childBoneName];
        
        if (bone && childBone) {
          const boneWP = new THREE.Vector3();
          const childWP = new THREE.Vector3();
          bone.getWorldPosition(boneWP);
          childBone.getWorldPosition(childWP);
          
          mapping.restDirWorld = new THREE.Vector3().subVectors(childWP, boneWP).normalize();
          mapping.restQuatWorld = new THREE.Quaternion();
          bone.getWorldQuaternion(mapping.restQuatWorld);
          
          console.log(`Rest pose computed for ${boneName}: restDirWorld=`, mapping.restDirWorld);
        } else {
          // Fallbacks if child is not found
          let defaultDir = new THREE.Vector3(0, 1, 0);
          if (boneName.includes('Arm') || boneName.includes('Hand')) {
            defaultDir = new THREE.Vector3(0, 1, 0);
          } else if (boneName.includes('Leg')) {
            defaultDir = new THREE.Vector3(0, -1, 0);
          }
          mapping.restDirWorld = defaultDir.clone().applyQuaternion(bone.quaternion).normalize();
          mapping.restQuatWorld = bone.quaternion.clone();
        }
      });

      // Compute rest-pose world orientation of hips from bone world positions.
      // This avoids relying on matrixWorld.decompose() which can fail with reflections/negative scales.
      {
        const hipsWP = new THREE.Vector3();
        const spineWP = new THREE.Vector3();
        const lulWP = new THREE.Vector3();
        const rulWP = new THREE.Vector3();
        skeletonBones['mixamorig2Hips'].getWorldPosition(hipsWP);
        skeletonBones['mixamorig2Spine'].getWorldPosition(spineWP);
        skeletonBones['mixamorig2LeftUpLeg'].getWorldPosition(lulWP);
        skeletonBones['mixamorig2RightUpLeg'].getWorldPosition(rulWP);

        // Build an orthonormal basis from the model's rest-pose bone positions
        let rv = new THREE.Vector3().subVectors(spineWP, hipsWP).normalize();   // vertical (up)
        let rl = new THREE.Vector3().subVectors(rulWP, lulWP).normalize();       // lateral (character's left→right)
        let rf = new THREE.Vector3().crossVectors(rv, rl).normalize();           // forward = cross(vertical, lateral)
        rl.crossVectors(rv, rf).normalize(); // re-orthogonalize lateral

        const restBasis = new THREE.Matrix4().makeBasis(rl, rv, rf);
        q_hips_rest_world.setFromRotationMatrix(restBasis);
        q_hips_rest_local.copy(skeletonBones['mixamorig2Hips'].quaternion);

        console.log('Hips rest-pose: lateral=', rl, 'vertical=', rv, 'forward=', rf);
        console.log('Hips rest-pose world quat:', q_hips_rest_world);
        console.log('Hips rest-pose local quat:', q_hips_rest_local);
      }

      // Create skeleton helper for visualization
      skeletonHelper = new THREE.SkeletonHelper(dancerModel);
      skeletonHelper.material.linewidth = 2;
      skeletonHelper.visible = chkBones.checked;
      scene.add(skeletonHelper);

      setStatus('Ready');
      systemStatusEl.classList.add('ready');
      
      // Load Keypoint data
      loadKeypoints();
    },
    (xhr) => {
      const pct = Math.round((xhr.loaded / xhr.total) * 100);
      setStatus(`Loading GLTF Model: ${pct}%`);
    },
    (err) => {
      console.error(err);
      setStatus('Error loading dancer.glb');
    }
  );
}

/* Load JSON Keypoints */
function loadKeypoints() {
  setStatus('Loading keypoints database...');
  
  fetch('/dance_3d_keypoints.json')
    .then(response => response.json())
    .then(data => {
      keypointData = data;
      fps = data.fps || 30;
      totalFrames = data.total_frames || Object.keys(data.frames).length;
      frameDurationMs = 1000 / fps;
      
      timelineSlider.max = totalFrames - 1;
      hudTotal.textContent = totalFrames;
      hudFps.textContent = fps.toFixed(1);
      
      // Display duration
      const totalSecs = totalFrames / fps;
      timeTotal.textContent = formatTime(totalSecs);
      
      setStatus('Ready to Dance');
      systemStatusEl.classList.add('ready');
      videoOverlayStatus.textContent = `${totalFrames} frames loaded`;
      
      // Compute first frame's mid-hip keypoint for relative tracking
      const firstFrame = data.frames['0'] || data.frames[Object.keys(data.frames)[0]];
      if (firstFrame && firstFrame.keypoints_3d) {
        const lh = getMPKeypoint(firstFrame.keypoints_3d['left_hip']);
        const rh = getMPKeypoint(firstFrame.keypoints_3d['right_hip']);
        firstFrameMidHip = new THREE.Vector3().addVectors(lh, rh).multiplyScalar(0.5);
        console.log('Successfully initialized firstFrameMidHip:', firstFrameMidHip);
      }

      // Build Telemetry Card List in DOM
      buildTelemetryUI();
      
      // Initialize first frame skeleton pose
      updatePose(0);
    })
    .catch(err => {
      console.error(err);
      setStatus('Error loading keypoint JSON. Make sure python script ran successfully.');
      videoOverlayStatus.textContent = 'Failed to load keypoints';
    });
}


function buildTelemetryUI() {
  boneTelemetryList.innerHTML = '';
  BONE_NAMES.forEach(boneName => {
    const card = document.createElement('div');
    card.className = 'bone-telemetry-card';
    card.id = `telemetry-${boneName}`;
    card.innerHTML = `
      <div class="bone-card-header">
        <span class="bone-card-title">${boneName.replace('mixamorig2', '')}</span>
        <span class="bone-card-status" id="status-${boneName}">IDLE</span>
      </div>
      <div class="bone-angles">
        <div class="angle-val">X: <span id="val-${boneName}-x">0.0°</span></div>
        <div class="angle-val">Y: <span id="val-${boneName}-y">0.0°</span></div>
        <div class="angle-val">Z: <span id="val-${boneName}-z">0.0°</span></div>
      </div>
    `;
    card.addEventListener('click', () => {
      // Highlight bone in 3D helper or console
      document.querySelectorAll('.bone-telemetry-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      console.log(`Inspecting joint bone: ${boneName}`, skeletonBones[boneName]);
    });
    boneTelemetryList.appendChild(card);
  });
}

/* Set status helper */
function setStatus(text) {
  systemStatusEl.innerHTML = text;
}

/* Format seconds to MM:SS */
function formatTime(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

/* Coordinate Converter: MediaPipe to Three.js model space */
function getMPKeypoint(kp) {
  if (!kp) return new THREE.Vector3(0, 0, 0);
  // MediaPipe world landmarks: +X = subject's right, +Y = up, +Z = toward camera
  // Model convention:           +X = character's LEFT,  +Y = up, +Z = forward (toward camera)
  // Only negate X (subject's right → character's left). Z convention already matches.
  return new THREE.Vector3(-kp.x, kp.y, kp.z);
}

/* Compute Pose calculations and apply to Bones */
function updatePose(frameIdx) {
  if (!keypointData || !dancerModel) return;

  const frame = keypointData.frames[frameIdx.toString()];
  if (!frame) return;

  const kps = frame.keypoints_3d;
  if (!kps || Object.keys(kps).length === 0) return;

  // 1. Get raw MediaPipe coordinates
  const left_hip = getMPKeypoint(kps['left_hip']);
  const right_hip = getMPKeypoint(kps['right_hip']);
  const left_shoulder = getMPKeypoint(kps['left_shoulder']);
  const right_shoulder = getMPKeypoint(kps['right_shoulder']);
  const left_elbow = getMPKeypoint(kps['left_elbow']);
  const right_elbow = getMPKeypoint(kps['right_elbow']);
  const left_wrist = getMPKeypoint(kps['left_wrist']);
  const right_wrist = getMPKeypoint(kps['right_wrist']);
  const left_knee = getMPKeypoint(kps['left_knee']);
  const right_knee = getMPKeypoint(kps['right_knee']);
  const left_ankle = getMPKeypoint(kps['left_ankle']);
  const right_ankle = getMPKeypoint(kps['right_ankle']);
  const left_index = getMPKeypoint(kps['left_index'] || kps['left_wrist']);
  const right_index = getMPKeypoint(kps['right_index'] || kps['right_wrist']);

  // Midpoints
  const mid_hip = new THREE.Vector3().addVectors(left_hip, right_hip).multiplyScalar(0.5);
  const mid_shoulder = new THREE.Vector3().addVectors(left_shoulder, right_shoulder).multiplyScalar(0.5);

  // Put midpoints in a temporary dictionary for BONE_MAPPINGS resolution
  const jointPositions = {
    left_hip, right_hip, left_shoulder, right_shoulder,
    left_elbow, right_elbow, left_wrist, right_wrist,
    left_knee, right_knee, left_ankle, right_ankle,
    left_index, right_index, mid_hip, mid_shoulder
  };

  // 2. Update Hips (Root) Rotation & Position
  const hipsBone = skeletonBones['mixamorig2Hips'];
  if (hipsBone) {
    // Compute target body orientation in world space from MediaPipe keypoints
    // Lateral: right_hip - left_hip (direction from character's left to right)
    const v_lateral = new THREE.Vector3().subVectors(right_hip, left_hip).normalize();
    // Vertical: hips→shoulders (positive Y, upward)
    const v_vertical = new THREE.Vector3().subVectors(mid_shoulder, mid_hip).normalize();
    // Forward: cross(vertical, lateral) — matches the model's +Z forward direction
    const v_forward = new THREE.Vector3().crossVectors(v_vertical, v_lateral).normalize();
    // Re-orthogonalize lateral
    v_lateral.crossVectors(v_vertical, v_forward).normalize();
    
    // Build target world quaternion from the orthonormal basis
    const basisMat = new THREE.Matrix4().makeBasis(v_lateral, v_vertical, v_forward);
    const q_target_world = new THREE.Quaternion().setFromRotationMatrix(basisMat);
    
    // Compute hips local rotation using rest-pose correction.
    const q_rest_world_inv = q_hips_rest_world.clone().invert();
    hipsBone.quaternion.copy(q_hips_rest_local)
      .multiply(q_rest_world_inv)
      .multiply(q_target_world);
    
    // Position: translate hips based on relative MediaPipe hip center movement
    const baseMidHip = firstFrameMidHip || mid_hip;
    const posOffset = new THREE.Vector3().subVectors(mid_hip, baseMidHip).multiplyScalar(positionScale);
    posOffset.x *= 1.0;
    posOffset.y *= 1.0; // Preserve vertical range for crouching (pliés)
    posOffset.z *= 1.0;
    
    // Derive parent inverse rotation from rest-pose data (avoids decompose):
    const q_parent_inv = q_hips_rest_local.clone().multiply(q_rest_world_inv);
    const posOffsetLocal = posOffset.clone().applyQuaternion(q_parent_inv).multiplyScalar(100);
    hipsBone.position.copy(initialHipsPos).add(posOffsetLocal);
    
    // Propagate matrix changes down the hierarchy
    hipsBone.updateMatrix();
    if (hipsBone.parent) {
      hipsBone.matrixWorld.multiplyMatrices(hipsBone.parent.matrixWorld, hipsBone.matrix);
    } else {
      hipsBone.matrixWorld.copy(hipsBone.matrix);
    }
    
    updateTelemetryUI('mixamorig2Hips', hipsBone.quaternion, 'TRACKING');
  }

  // Decompose current body rotation relative to rest pose
  const q_body_rotation = new THREE.Quaternion();
  if (hipsBone) {
    hipsBone.matrixWorld.decompose(new THREE.Vector3(), q_body_rotation, new THREE.Vector3());
    q_body_rotation.multiply(q_hips_rest_world.clone().invert());
  }

  // 3. Update Bones hierarchically using depth-sorted traversal
  allBonesSorted.forEach(bone => {
    const boneName = bone.name;
    if (boneName === 'mixamorig2Hips') {
      return; // already updated in step 2
    }
    
    const mapping = BONE_MAPPINGS[boneName];
    if (bone && mapping) {
      const p_start = jointPositions[mapping.parentKp];
      const p_end = jointPositions[mapping.childKp];
      
      if (p_start && p_end && p_start.distanceTo(p_end) > 0.0001) {
        // Target direction in world space
        const v_target_world = new THREE.Vector3().subVectors(p_end, p_start).normalize();
        
        // Rotate rest-pose world direction and quaternion by the current body rotation
        const v_rest_world_current = mapping.restDirWorld.clone().applyQuaternion(q_body_rotation).normalize();
        const q_rest_world_current = q_body_rotation.clone().multiply(mapping.restQuatWorld);
        
        // Shortest path rotation in world space from the rotated rest world direction to the target world direction
        const q_diff_world = new THREE.Quaternion().setFromUnitVectors(v_rest_world_current, v_target_world);
        
        // Combine with rotated rest world rotation to get target world rotation
        const q_target_world = q_diff_world.clone().multiply(q_rest_world_current);
        
        // Parent's world rotation
        const q_parent_world = new THREE.Quaternion();
        if (bone.parent) {
          bone.parent.matrixWorld.decompose(new THREE.Vector3(), q_parent_world, new THREE.Vector3());
        }
        
        // Transform target world rotation to parent's local space
        const q_parent_world_inv = q_parent_world.clone().invert();
        const q_local = q_parent_world_inv.clone().multiply(q_target_world);
        
        // Apply local rotation
        bone.quaternion.copy(q_local);
        updateTelemetryUI(boneName, bone.quaternion, 'TRACKING');
      } else {
        // Fallback if keypoints are collapsed/invalid
        if (initialBoneQuaternions[boneName]) {
          bone.quaternion.copy(initialBoneQuaternions[boneName]);
        }
        updateTelemetryUI(boneName, bone.quaternion, 'MISSING KPS');
      }
    } else if (bone) {
      // Keep static bones at their rest local pose
      if (initialBoneQuaternions[boneName]) {
        bone.quaternion.copy(initialBoneQuaternions[boneName]);
      }
      updateTelemetryUI(boneName, bone.quaternion, 'STATIC');
    }
    
    // Propagate matrix updates top-down
    bone.updateMatrix();
    if (bone.parent) {
      bone.matrixWorld.multiplyMatrices(bone.parent.matrixWorld, bone.matrix);
    } else {
      bone.matrixWorld.copy(bone.matrix);
    }
  });

  // Update Skeleton Helper if shown
  if (skeletonHelper && skeletonHelper.visible) {
    skeletonHelper.update();
  }

  // 4. Update MediaPipe Reference Skeleton
  if (mpSkeletonGroup.visible) {
    // Move individual joints
    Object.keys(mpJointSpheres).forEach(name => {
      const kp = kps[name];
      const mesh = mpJointSpheres[name];
      if (kp && mesh) {
        const localPos = getMPKeypoint(kp).multiplyScalar(skeletonScale);
        // Align vertically with hips (converting cm bone position to meters * 0.01)
        localPos.y += initialHipsPos.y * 0.01;
        mesh.position.copy(localPos);
        mesh.visible = true;
      } else if (mesh) {
        mesh.visible = false;
      }
    });

    // Update lines segment geometry
    const lineAttr = mpLineGeometry.attributes.position;
    const array = lineAttr.array;
    let idx = 0;

    SKELETON_CONNECTIONS.forEach(([kpA, kpB]) => {
      const valA = kps[kpA];
      const valB = kps[kpB];
      if (valA && valB) {
        const posA = getMPKeypoint(valA).multiplyScalar(skeletonScale);
        const posB = getMPKeypoint(valB).multiplyScalar(skeletonScale);
        
        // Apply height offset to align with 3D model (converting cm bone position to meters * 0.01)
        posA.y += initialHipsPos.y * 0.01;
        posB.y += initialHipsPos.y * 0.01;

        array[idx++] = posA.x;
        array[idx++] = posA.y;
        array[idx++] = posA.z;

        array[idx++] = posB.x;
        array[idx++] = posB.y;
        array[idx++] = posB.z;
      } else {
        // Zero out connection if missing
        for (let i = 0; i < 6; i++) {
          array[idx++] = 0;
        }
      }
    });
    lineAttr.needsUpdate = true;
  }
}

/* Update Telemetry Values in Sidebar */
function updateTelemetryUI(boneName, quaternion, status) {
  const statusEl = document.getElementById(`status-${boneName}`);
  if (statusEl) {
    statusEl.textContent = status;
    if (status === 'TRACKING') {
      statusEl.className = 'bone-card-status';
      statusEl.style.color = 'var(--accent-cyan)';
    } else if (status === 'STATIC') {
      statusEl.className = 'bone-card-status';
      statusEl.style.color = 'var(--text-muted)';
    } else {
      statusEl.className = 'bone-card-status';
      statusEl.style.color = '#ff9900';
    }
  }

  // Convert quaternion to Euler angles (yaw, pitch, roll) in degrees
  const euler = new THREE.Euler().setFromQuaternion(quaternion, 'YXZ');
  const xEl = document.getElementById(`val-${boneName}-x`);
  const yEl = document.getElementById(`val-${boneName}-y`);
  const zEl = document.getElementById(`val-${boneName}-z`);

  if (xEl) xEl.textContent = `${(euler.x * 180 / Math.PI).toFixed(1)}°`;
  if (yEl) yEl.textContent = `${(euler.y * 180 / Math.PI).toFixed(1)}°`;
  if (zEl) zEl.textContent = `${(euler.z * 180 / Math.PI).toFixed(1)}°`;
}

/* Set frame and slider sync */
function setFrame(frameIdx) {
  currentFrame = Math.max(0, Math.min(totalFrames - 1, frameIdx));
  timelineSlider.value = currentFrame;
  hudFrame.textContent = currentFrame;
  
  // Progress bar styling
  const pct = (currentFrame / (totalFrames - 1)) * 100;
  timelineProgress.style.width = `${pct}%`;
  
  // Current time display
  const currentSecs = currentFrame / fps;
  timeCurrent.textContent = formatTime(currentSecs);

  // Sync reference video playback
  if (refVideo.src) {
    const videoTime = currentFrame / fps;
    // Set video time only if far enough out of sync, to avoid scrubbing jitter
    if (Math.abs(refVideo.currentTime - videoTime) > 0.08) {
      refVideo.currentTime = videoTime;
    }
  }

  updatePose(currentFrame);
}

/* Playback Loop */
function animate(timestamp) {
  requestAnimationFrame(animate);

  if (isPlaying) {
    if (!lastFrameTime) lastFrameTime = timestamp;
    const elapsed = timestamp - lastFrameTime;

    if (elapsed >= frameDurationMs / playbackSpeed) {
      currentFrame = (currentFrame + 1) % totalFrames;
      setFrame(currentFrame);
      lastFrameTime = timestamp;
    }
  }

  // Camera Auto-rotation if enabled
  if (chkAutoCamera.checked) {
    const time = clock.getElapsedTime() * 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;
  } else {
    controls.autoRotate = false;
  }

  controls.update();
  renderer.render(scene, camera);
}

/* Resize listener */
function onWindowResize() {
  const width = threeContainer.clientWidth;
  const height = threeContainer.clientHeight;

  camera.aspect = width / height;
  camera.updateProjectionMatrix();

  renderer.setSize(width, height);
}

/* Toggle controls and checkboxes listeners */
function setupUIListeners() {
  btnPlayPause.addEventListener('click', () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      playIcon.textContent = '⏸';
      btnPlayPause.innerHTML = '<span id="play-icon">⏸</span> Pause';
      btnPlayPause.classList.remove('btn-primary');
      btnPlayPause.classList.add('btn-secondary');
      systemStatusEl.className = 'status-badge playing';
      systemStatusEl.textContent = 'Syncing...';
      refVideo.play().catch(e => console.log('Video autoplay blocked or pending', e));
      lastFrameTime = 0;
    } else {
      playIcon.textContent = '▶';
      btnPlayPause.innerHTML = '<span id="play-icon">▶</span> Play';
      btnPlayPause.classList.remove('btn-secondary');
      btnPlayPause.classList.add('btn-primary');
      systemStatusEl.className = 'status-badge ready';
      systemStatusEl.textContent = 'Paused';
      refVideo.pause();
    }
  });

  btnReset.addEventListener('click', () => {
    isPlaying = false;
    playIcon.textContent = '▶';
    btnPlayPause.innerHTML = '<span id="play-icon">▶</span> Play';
    btnPlayPause.classList.remove('btn-secondary');
    btnPlayPause.classList.add('btn-primary');
    systemStatusEl.className = 'status-badge ready';
    systemStatusEl.textContent = 'Ready';
    refVideo.pause();
    refVideo.currentTime = 0;
    setFrame(0);
  });

  // Slider scrubbing
  timelineSlider.addEventListener('input', () => {
    isPlaying = false;
    playIcon.textContent = '▶';
    btnPlayPause.innerHTML = '<span id="play-icon">▶</span> Play';
    btnPlayPause.classList.remove('btn-secondary');
    btnPlayPause.classList.add('btn-primary');
    systemStatusEl.className = 'status-badge ready';
    systemStatusEl.textContent = 'Scrubbing';
    refVideo.pause();
    
    setFrame(parseInt(timelineSlider.value));
  });

  // Speed selectors
  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playbackSpeed = parseFloat(btn.dataset.speed);
      refVideo.playbackRate = playbackSpeed;
    });
  });

  // Visualization toggles
  chkGrid.addEventListener('change', () => {
    floorGrid.visible = chkGrid.checked;
  });

  chkBones.addEventListener('change', () => {
    if (skeletonHelper) {
      skeletonHelper.visible = chkBones.checked;
    }
  });

  chkMpHelper.addEventListener('change', () => {
    mpSkeletonGroup.visible = chkMpHelper.checked;
  });
}

// Initializing code
initThree();
loadDancerModel();
setupUIListeners();
requestAnimationFrame(animate);
