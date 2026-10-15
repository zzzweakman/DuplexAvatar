import assert from 'node:assert/strict';
import {CameraInput} from '../frontend/camera-input.mjs';

// Independent 5 Hz sampling and 960 ms audio produce 4/5-frame batches, not
// five duplicated snapshots taken together at the end of each audio block.
const camera = new CameraInput(5, 6), batches = [];
for (let now = 40; now <= 48000; now += 40) {
  camera.sample(now, () => ({image: `frame-${now}`, capturedAt: now}));
  if (now % 960 === 0) batches.push(camera.drain());
}
const frames = batches.flat();
assert.equal(frames.length, 240);
assert.deepEqual([...new Set(batches.map(batch => batch.length))].sort(), [4, 5]);
assert.equal(new Set(frames.map(frame => frame.image)).size, 240);
assert.ok(frames.slice(1).every((frame, i) => frame.capturedAt - frames[i].capturedAt === 200));

// Camera sampling also stays at 5 FPS with the 1,000 ms voice-output clock.
camera.reset(0);
for (let now = 200; now <= 10000; now += 200) {
  camera.sample(now, () => ({capturedAt: now}));
  if (now % 1000 === 0) assert.equal(camera.drain().length, 5);
}

// A delayed callback may capture one current frame; it cannot invent history.
camera.reset(0);
let captures = 0;
camera.sample(5000, () => { captures++; return {image: 'after-stall', capturedAt: 5000}; });
camera.sample(5001, () => { captures++; return {image: 'too-early'}; });
assert.equal(captures, 1);
assert.equal(camera.drain().length, 1);
for (let now = 5200; now <= 10000; now += 200) camera.sample(now, () => ({capturedAt: now}));
assert.equal(camera.frames.length, 6);
assert.equal(camera.frames[0].capturedAt, 9000);

// Pause/session reset releases queued images and does not replay old captures.
camera.reset(10000);
assert.deepEqual(camera.drain(), []);
camera.sample(10200, () => null);
assert.deepEqual(camera.drain(), []);
camera.sample(10400, () => ({image: 'resumed', capturedAt: 10400}));
assert.deepEqual(camera.drain().map(frame => frame.image), ['resumed']);
console.log('Camera sampling: 5 FPS, ordered 4/5-frame batches, bounded stalls and reset passed');
