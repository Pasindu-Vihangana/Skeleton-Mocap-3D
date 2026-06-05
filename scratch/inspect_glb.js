import fs from 'fs';
import path from 'path';

const glbPath = path.resolve('public/dancer.glb');
const buffer = fs.readFileSync(glbPath);
const chunkLength = buffer.readUInt32LE(12);
const jsonBuffer = buffer.subarray(20, 20 + chunkLength);
const gltf = JSON.parse(jsonBuffer.toString('utf-8'));

console.log("Checking if nodes define 'matrix':");
let foundMatrixCount = 0;
gltf.nodes.forEach((node, idx) => {
  if (node.matrix) {
    foundMatrixCount++;
    if (foundMatrixCount <= 10) {
      console.log(`Node [${idx}] "${node.name}" has matrix:`, node.matrix);
    }
  }
});
console.log(`Total nodes with matrix: ${foundMatrixCount} out of ${gltf.nodes.length}`);
