// Upload credits adapt to measured RTT; capture and inference have separate clocks.
export class InputFlow {
  constructor(send, canSend, window = 2, limit = 30, {unitMs = 1000, maxWindow = 8, clock = () => performance.now()} = {}) {
    this.send = send; this.canSend = canSend; this.window = window; this.limit = limit;
    this.minimumWindow = window; this.maxWindow = maxWindow; this.unitMs = unitMs; this.clock = clock; this.rttMs = 0;
    this.queue = []; this.inflight = new Map(); this.sequence = 0;
  }
  enqueue(item) {
    if (this.queue.length >= this.limit) throw new Error('网络发送缓冲已满，请检查连接后重新开始。');
    item.client_input_id = ++this.sequence;
    this.queue.push(item); this.pump();
  }
  received(id) {
    const sent = this.inflight.get(id);
    if (sent !== undefined) {
      const sample = Math.max(0, this.clock() - sent);
      this.rttMs = this.rttMs ? .8 * this.rttMs + .2 * sample : sample;
      this.window = Math.max(this.minimumWindow, Math.min(this.maxWindow, Math.ceil((this.rttMs + 100) / this.unitMs) + 1));
    }
    this.inflight.delete(id); this.pump();
  }
  pump() {
    while (this.queue.length && this.inflight.size < this.window && this.canSend()) {
      const item = this.queue.shift(); this.inflight.set(item.client_input_id, this.clock()); this.send(item);
    }
  }
}
