"""Bounded utterance segmentation for display-only ASR, off the brain's path."""
from collections import deque
import numpy as np


class Utterances:
    rate = 16000
    frame = 1600  # 100 ms; keep 200 ms before speech, close after 700 ms silence.

    def __init__(self):
        self.prefix = deque(maxlen=2)
        self.parts = []
        self.start_ms = 0
        self.end_ms = 0
        self.last_voice_ms = 0
        self.voiced_ms = 0
        self.counter = 0
        self.noise = deque(maxlen=64)
        self.previous_end_ms = None

    def flush(self):
        job = None
        if self.parts and self.voiced_ms >= 300:
            self.counter += 1
            audio = np.concatenate(self.parts)
            job = {"utterance_id": f"user-{self.counter}", "start_ms": self.start_ms,
                   "end_ms": min(self.end_ms, self.last_voice_ms + 200),
                   "pcm": (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()}
        self.parts = []
        self.voiced_ms = 0
        return job

    def append(self, audio, start_ms):
        jobs = []
        if self.previous_end_ms is not None and start_ms > self.previous_end_ms + 150:
            job = self.flush()
            if job:
                jobs.append(job)
            self.prefix.clear()
        for offset in range(0, len(audio), self.frame):
            part = audio[offset:offset + self.frame].copy()
            start = start_ms + offset * 1000 / self.rate
            end = start + len(part) * 1000 / self.rate
            self.previous_end_ms = end
            rms = float(np.sqrt(np.mean(part ** 2)))
            self.noise.append(min(rms, .02))
            floor = float(np.quantile(self.noise, .2))
            threshold = max(.0035, min(floor * 1.8, .0105))
            voiced = rms >= threshold
            if not self.parts and voiced:
                self.start_ms = self.prefix[0][0] if self.prefix else start
                self.parts = [p for _, p in self.prefix]
                self.prefix.clear()
            if self.parts or voiced:
                self.parts.append(part)
                self.end_ms = end
                if voiced:
                    self.last_voice_ms = end
                    self.voiced_ms += end - start
                if end - self.last_voice_ms >= 700 or end - self.start_ms >= 12000:
                    job = self.flush()
                    if job:
                        jobs.append(job)
            else:
                self.prefix.append((start, part))
        return jobs
