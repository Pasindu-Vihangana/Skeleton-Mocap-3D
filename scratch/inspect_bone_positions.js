import fs from 'fs';
import path from 'path';

const glbPath = path.resolve('public/dancer.glb');
const buffer = fs.readFileSync(glbPath);
const chunkLength = buffer.readUInt32LE(12);
const jsonBuffer = buffer.subarray(20, 20 + chunkLength);
const gltf = JSON.parse(jsonBuffer.toString('utf-8'));

const boneNames = [
  'mixamorig2Hips',
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

console.log("Bone local translations in glTF JSON:");
gltf.nodes.forEach((node, idx) => {
  if (boneNames.includes(node.name)) {
    console.log(`Node [${idx}] "${node.name}":`, {
      translation: node.translation,
      rotation: node.rotation,
      scale: node.scale,
      children: node.children
    });
  }
});
