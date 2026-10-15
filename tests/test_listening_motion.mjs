import assert from 'node:assert/strict';
import {ListeningMotion} from '../frontend/listening-motion.mjs';

let copied = 0, closed = 0;
const copy = image => { copied++; return {id: image.id, close() { closed++; }}; };
const motion = new ListeningMotion(4, 3);
for (let i = 0; i < 6; i++) motion.remember({image: {id: i}, epoch: 2, sequence: 9,
  frame_index: i, fps: 25, idle: true}, i / 25, copy);
assert.equal(motion.frames.length, 4);
assert.equal(closed, 2, 'The frame cache must be bounded');
assert.equal(motion.next(.21, 2), null, 'Do not replace the normal 25 FPS timeline');
assert.equal(motion.next(.30, 2), null, 'Ordinary frame jitter must not start backwards motion');
const indices = Array.from({length: 7}, (_, i) => motion.next(.35 + i * .05, 2).frame_index);
assert.deepEqual(indices, [4, 3, 2, 3, 4, 5, 4], 'Resume at the visible endpoint without a looping jump');
assert.equal(motion.next(1, 3), null, 'Old epochs cannot supply listening motion');
assert.equal(motion.next(4, 2), null, 'Do not hide a prolonged outage');
motion.remember({image: {id: 100}, epoch: 2, idle: false}, 1, copy);
motion.remember({...motion.frames[0], fallback: true}, 1, copy);
assert.equal(copied, 6, 'Never retain speech or recursively retain fallback frames');
motion.remember({image: {id: 6}, epoch: 3, idle: true}, 1, copy);
assert.equal(motion.frames.length, 1, 'A new epoch clears all previous frames');
motion.clear();
assert.equal(closed, copied, 'Every owned frame must be released');
assert.equal(motion.next(2, 3), null);
console.log('Listening motion: silent frames only, bounded memory/time, continuous endpoint, epoch isolation passed.');
