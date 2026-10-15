import assert from 'node:assert/strict';
import {InputFlow} from '../frontend/input-flow.mjs';

const sent = [];
let writable = true;
const queue = new InputFlow(item => sent.push(item), () => writable);
for (let i = 0; i < 18; i++) queue.enqueue({audio: `voice-${i}`});
assert.equal(sent.length, 2, 'Two upload credits bound network work independently of GPU ACKs');
assert.equal(queue.queue.length, 16, 'Voice beyond six seconds is retained');
for (let i = 0; i < 18; i++) queue.received(i + 1);
assert.deepEqual(sent.map(i => i.audio), Array.from({length: 18}, (_, i) => `voice-${i}`));
assert.equal(queue.inflight.size, 0);
writable = false;
for (let i = 0; i < 30; i++) queue.enqueue({audio: `network-${i}`});
assert.equal(queue.queue.length, 30);
assert.throws(() => queue.enqueue({audio: 'overflow'}));
writable = true; queue.pump();
assert.equal(queue.inflight.size, 2);
console.log('Input flow: voice preservation, upload credits, network backpressure and bounded memory passed.');

function simulate(seconds, maxWindow) {
  let now = 0, due = [], largest = 0, delivered = 0;
  const flow = new InputFlow(item => due.push({at: now + 2600, id: item.client_input_id}), () => true,
    2, 10000, {unitMs: 960, maxWindow, clock: () => now});
  for (; now < seconds * 1000; now += 20) {
    const ready = due.filter(item => item.at <= now); due = due.filter(item => item.at > now);
    for (const item of ready) { flow.received(item.id); delivered++; }
    if (now % 960 === 0) flow.enqueue({audio: `speech-${now}`});
    largest = Math.max(largest, flow.queue.length);
  }
  return {largest, queued: flow.queue.length, delivered, window: flow.window};
}
const fixed = simulate(120, 2), adaptive = simulate(600, 8);
assert.ok(fixed.queued >= 25, JSON.stringify(fixed));
assert.ok(adaptive.largest <= 1 && adaptive.queued === 0, JSON.stringify(adaptive));
assert.equal(adaptive.window, 4);
console.log('2600 ms RTT: fixed two-credit window accumulates', fixed.queued, 'blocks in 120 s; adaptive window has', adaptive.queued, 'queued after 600 s.');
