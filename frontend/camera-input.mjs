// Sample the live camera on its own clock; audio blocks drain ordered snapshots.
export class CameraInput {
  constructor(fps = 5, limit = 6) {
    if (!(fps > 0) || !Number.isInteger(limit) || limit < 1) throw new Error('无效的摄像头采样配置');
    this.period = 1000 / fps; this.limit = limit; this.reset();
  }
  reset(now = 0) {
    this.frames = []; this.nextAt = now + this.period;
  }
  sample(now, capture) {
    if (now < this.nextAt) return;
    // Skip missed slots after a browser stall. Never repeat the current image
    // several times to pretend that missed historical frames were captured.
    this.nextAt += (Math.floor((now - this.nextAt) / this.period) + 1) * this.period;
    const frame = capture();
    if (!frame) return;
    this.frames.push(frame);
    if (this.frames.length > this.limit) this.frames.shift();
  }
  drain() {
    const frames = this.frames; this.frames = []; return frames;
  }
}
