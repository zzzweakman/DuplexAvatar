"""One negotiated block clock; audio sample counts remain exact integers."""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class BlockClock:
    input_samples: int
    driving_samples: int
    duration_ms: int
    frames: int = 0
    fps: int = 0

    @classmethod
    def from_worker(cls, worker=None):
        if worker is None:
            return cls(16000, 24000, 1000)
        frames, fps = int(worker["frames_per_chunk"]), int(worker["fps"])
        if frames <= 0 or fps <= 0:
            raise ValueError("Invalid A2V block clock")
        seconds = Fraction(frames, fps)
        values = [seconds * rate for rate in (16000, 24000, 1000)]
        if any(value.denominator != 1 for value in values):
            raise ValueError("A2V block is not exactly representable in audio samples/milliseconds")
        clock = cls(*(int(value) for value in values), frames, fps)
        if clock.driving_samples != worker["chunk_samples"] or clock.duration_ms % 10:
            raise ValueError("A2V timing disagrees with its audio samples or MiniCPM's mel hop")
        return clock

    def metadata(self):
        return {"decision_ms": self.duration_ms, "input_samples": self.input_samples,
                "driving_samples": self.driving_samples, "frames_per_block": self.frames, "fps": self.fps,
                "tts_output": "lossless_sample_buffer", "tail": "minimum_silence_then_complete_block"}
