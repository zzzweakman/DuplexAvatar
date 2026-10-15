"""PCM chunk accounting. Original 24 kHz audio is never synthesized again."""
import base64
import numpy as np


def encode_pcm(samples):
    return base64.b64encode(np.asarray(samples, dtype="<f4").tobytes()).decode("ascii")


def decode_pcm(value, max_samples=24000 * 30):
    data = base64.b64decode(value, validate=True)
    if len(data) % 4 or len(data) > max_samples * 4:
        raise ValueError("Invalid PCM length")
    samples = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(samples).all():
        raise ValueError("PCM contains non-finite samples")
    return samples


class AudioChunks:
    def __init__(self, size):
        self.size = size
        self.pending = np.empty(0, dtype=np.float32)

    def append(self, samples):
        self.pending = np.concatenate((self.pending, samples))
        blocks = []
        while self.pending.size >= self.size:
            blocks.append(self.pending[:self.size].copy())
            self.pending = self.pending[self.size:]
        return blocks

    def flush(self):
        if not self.pending.size:
            return None
        tail = self.pending.copy()
        self.clear()
        return tail

    def clear(self):
        self.pending = np.empty(0, dtype=np.float32)
