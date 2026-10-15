// Only already displayed, silent A2V frames may cover a short delivery gap.
// Walk backwards and forwards from the visible endpoint, without a loop seam.
export class ListeningMotion {
  constructor(limit = 24, maxGap = 3, grace = .12) {
    this.limit = limit; this.maxGap = maxGap; this.grace = grace; this.frames = []; this.active = false;
  }
  clear() {
    for (const frame of this.frames) frame.image.close();
    this.frames = []; this.active = false;
  }
  remember(frame, now, copy) {
    if (!frame.idle || frame.fallback) return;
    if (this.frames.length && this.frames[0].epoch !== frame.epoch) this.clear();
    this.frames.push({...frame, image: copy(frame.image)});
    while (this.frames.length > this.limit) this.frames.shift().image.close();
    this.lastLive = now; this.nextAt = now + 1 / (frame.fps || 25);
    this.cursor = this.frames.length - 2; this.direction = -1; this.active = false;
  }
  next(now, epoch) {
    if (this.frames.length < 3 || this.frames[0].epoch !== epoch || now - this.lastLive < this.grace ||
        now - this.lastLive > this.maxGap || now < this.nextAt) return null;
    const frame = this.frames[this.cursor];
    this.active = true; this.nextAt = now + 1 / (frame.fps || 25);
    if (this.cursor === 0) this.direction = 1;
    if (this.cursor === this.frames.length - 1) this.direction = -1;
    this.cursor += this.direction;
    return {...frame, fallback: true}; // Borrowed image; clear() owns its lifetime.
  }
}
