class PCMCollector extends AudioWorkletProcessor {
  constructor() {
    super(); this.last = 0; this.position = 0; this.buffer = new Float32Array(640); this.used = 0;
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input?.length) return true;
    const values = new Float32Array(input.length + 1); values[0] = this.last; values.set(input, 1);
    const step = sampleRate / 16000;
    while (this.position < values.length - 1) {
      const index = Math.floor(this.position), weight = this.position - index;
      this.buffer[this.used++] = values[index] * (1 - weight) + values[index + 1] * weight;
      this.position += step;
      if (this.used === this.buffer.length) {
        this.port.postMessage(this.buffer, [this.buffer.buffer]);
        this.buffer = new Float32Array(640); this.used = 0;
      }
    }
    this.position -= values.length - 1; this.last = input[input.length - 1];
    return true;
  }
}
registerProcessor('pcm-collector', PCMCollector);
