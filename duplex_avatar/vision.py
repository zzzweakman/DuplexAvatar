"""Bounds and capture timestamps for ordered camera batches."""
import math
from dataclasses import replace


def configure_window(decoder, enabled, units=24, previous_tokens=500):
    """Bound camera history with the model's own text-preserving window."""
    decoder.set_window_config(replace(decoder._window_config,
                                     sliding_window_mode="context" if enabled else "off",
                                     context_max_units=units, context_previous_max_tokens=previous_tokens))
    decoder.set_window_enabled(enabled)


def frame_limit(duration_ms, fps):
    # One extra frame tolerates capture/audio callback jitter at a block edge.
    return math.ceil(duration_ms * fps / 1000) + 1


def parse_frames(message, duration_ms, fps, fallback_ms):
    frames = message.get("video_frames") or []
    limit = frame_limit(duration_ms, fps)
    if not isinstance(frames, list) or len(frames) > limit:
        raise ValueError(f"At most {limit} JPEG frames are accepted per input")
    if any(not isinstance(frame, str) or not frame or len(frame) > 2_000_000 for frame in frames):
        raise ValueError("Invalid camera JPEG frame")
    times = message.get("video_frame_times_ms")
    if times is not None:
        if not isinstance(times, list) or len(times) != len(frames):
            raise ValueError("Camera timestamps must match the frame count")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 48 * 3600 * 1000 for value in times):
            raise ValueError("Invalid camera capture timestamp")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Camera frames must be in capture order")
    latest = float(message.get("video_capture_ms", times[-1] if times else fallback_ms))
    if not math.isfinite(latest) or not 0 <= latest <= 48 * 3600 * 1000:
        raise ValueError("Invalid camera capture timestamp")
    if times and not math.isclose(latest, times[-1], abs_tol=.001):
        raise ValueError("Latest camera timestamp does not match its frame")
    return frames, times, latest
