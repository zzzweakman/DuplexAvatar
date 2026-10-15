"""Independent H.264 blocks: every block starts with a keyframe and keeps all frames."""
import base64
import re
import subprocess
from fractions import Fraction

try:
    import av
except ImportError:
    av = None


def encoder_backend():
    return "pyav" if av is not None else "ffmpeg"


def encode_h264(frames, fps, *, backend="auto"):
    count, height, width, _ = frames.shape
    if backend == "auto":
        backend = encoder_backend()
    if backend == "pyav":
        data = _encode_in_process(frames, fps)
    elif backend == "ffmpeg":
        data = _encode_subprocess(frames, fps)
    else:
        raise ValueError(f"Unknown video encoder: {backend}")
    nals = [(m.start(), m.end()) for m in re.finditer(b"\x00\x00(?:\x00)?\x01", data)]
    starts = [start for start, payload in nals if data[payload] & 31 == 9]
    if len(starts) != count:
        raise RuntimeError(f"H.264 frame count mismatch: {len(starts)} != {count}")
    starts[0] = 0
    boundaries = starts[1:] + [len(data)]
    sps = next(payload for _, payload in nals if data[payload] & 31 == 7)
    return {"codec": "avc1." + data[sps + 1:sps + 4].hex().upper(), "format": "annexb",
            "frame_count": count, "chunks": [base64.b64encode(data[a:b]).decode("ascii")
                                               for a, b in zip(starts, boundaries)]}


def _encode_in_process(frames, fps):
    """Keep the existing block format without starting an FFmpeg process per second."""
    count, height, width, _ = frames.shape
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = width, height, "yuv420p"
    codec.time_base, codec.framerate = Fraction(1, fps), Fraction(fps, 1)
    codec.thread_count = 2
    codec.options = {"preset": "veryfast", "tune": "zerolatency", "crf": "22",
                     "profile": "baseline", "level": "3.0",
                     "x264-params": f"keyint={count}:min-keyint={count}:scenecut=0:aud=1:repeat-headers=1"}
    packets = []
    for index, pixels in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts, frame.time_base = index, codec.time_base
        packets.extend(bytes(packet) for packet in codec.encode(frame))
    packets.extend(bytes(packet) for packet in codec.encode(None))
    return b"".join(packets)


def _encode_subprocess(frames, fps):
    """Compatibility path for environments without PyAV."""
    count, height, width, _ = frames.shape
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-threads", "2", "-filter_threads", "1",
        "-preset", "veryfast", "-tune", "zerolatency", "-crf", "22",
        "-profile:v", "baseline", "-level:v", "3.0", "-pix_fmt", "yuv420p",
        "-x264-params", f"keyint={count}:min-keyint={count}:scenecut=0:aud=1:repeat-headers=1",
        "-f", "h264", "pipe:1"], input=frames.tobytes(), capture_output=True, check=True, timeout=10)
    return result.stdout
