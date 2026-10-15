"""Bounded playback checkpoints: resume motion at a displayed frame, not a future chunk."""
from collections import OrderedDict
import numpy as np


class MotionCache:
    def __init__(self, limit=10):
        self.limit = limit
        self.clips = OrderedDict()

    def clear(self):
        self.clips.clear()

    def remember(self, sequence, frames, history, driving, prefix, fps, rate):
        self.clips[sequence] = (frames, history, driving, prefix, fps, rate)
        while len(self.clips) > self.limit:
            self.clips.popitem(last=False)

    def resume(self, anchor):
        if not isinstance(anchor, dict):
            return None
        sequence, index = anchor.get("sequence"), anchor.get("frame_index")
        if type(sequence) is not int or type(index) is not int or sequence not in self.clips:
            return None
        frames, history, driving, prefix, fps, rate = self.clips[sequence]
        if not 0 <= index < len(frames) - prefix:
            return None
        # The cached prefix is the previous clip's motion context. Every window
        # below ends exactly on the requested generated/displayed frame.
        motion = frames[index + 1:index + 1 + prefix]
        consumed = min(len(driving), round((index + 1) * rate / fps))
        audio = np.concatenate((history, driving[:consumed]))[-len(history):]
        return motion, audio, {"sequence": sequence, "frame_index": index}
