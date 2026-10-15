import assert from 'node:assert/strict';
import {bridgePlan, blendWeight} from '../frontend/motion-transition.mjs';

const frames = Array.from({length: 24}, (_, i) => ({sequence: 12, frame_index: i, time: 10 + i / 25, epoch: 0, image: {}}));
const plan = bridgePlan(frames, 10.08, 1);
assert.equal(plan.frames.length, 4);
assert.equal(plan.anchor.frame_index, 5);
assert.equal(plan.anchor.sequence, 12);
assert.ok(plan.frames.at(-1).time <= 10.08 + .251);
assert.ok(plan.frames.every(f => f.epoch === 1 && f.bridge));
assert.ok(frames.every(f => f.epoch === 0), 'Original queued frames are not relabelled accidentally');
assert.equal(bridgePlan(frames, 10.08, 1, true).frames.length, 0);
assert.equal(bridgePlan([], 10, 1).anchor, null);
assert.equal(blendWeight(0, .12), 0);
assert.equal(blendWeight(.12, .12), 1);
assert.ok(blendWeight(.06, .12) > .5);
console.log('Visual interruption: bounded bridge, exact anchor, reduced motion and blend endpoints passed.');
