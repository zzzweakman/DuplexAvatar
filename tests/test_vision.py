import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from duplex_avatar.vision import configure_window, frame_limit, parse_frames


class VisionInputTests(unittest.TestCase):
    def test_camera_window_is_disabled_again_for_the_next_audio_session(self):
        @dataclass
        class Window:
            sliding_window_mode: str = "off"
            context_max_units: int = 24
            context_previous_max_tokens: int = 500
        decoder = SimpleNamespace(_window_config=Window())
        decoder.set_window_config = lambda config: setattr(decoder, "_window_config", config)
        decoder.set_window_enabled = lambda enabled: setattr(decoder, "enabled", enabled)
        configure_window(decoder, True)
        self.assertTrue(decoder.enabled)
        self.assertEqual(decoder._window_config.sliding_window_mode, "context")
        configure_window(decoder, False)
        self.assertFalse(decoder.enabled)
        self.assertEqual(decoder._window_config.sliding_window_mode, "off")

    def test_multiple_frames_preserve_order_and_latest_capture_time(self):
        for duration in (960, 1000):
            frames = [f"jpeg-{i}" for i in range(5)]
            times = [i * 200 + 20 for i in range(5)]
            self.assertEqual(parse_frames({"video_frames": frames, "video_frame_times_ms": times}, duration, 5, 0),
                             (frames, times, 820))
            self.assertEqual(frame_limit(duration, 5), 6)

    def test_oversized_batches_and_invalid_capture_order_are_rejected(self):
        for message in (
            {"video_frames": ["jpeg"] * 7},
            {"video_frames": "jpeg"},
            {"video_frames": ["a", "b"], "video_frame_times_ms": [20]},
            {"video_frames": ["a", "b"], "video_frame_times_ms": [200, 20]},
            {"video_frames": ["a", "b"], "video_frame_times_ms": [20, 20]},
            {"video_frames": ["a"], "video_frame_times_ms": [float("nan")]},
            {"video_frames": ["a"], "video_frame_times_ms": [200], "video_capture_ms": 20},
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                parse_frames(message, 960, 5, 0)

    def test_single_image_clients_and_audio_only_remain_compatible(self):
        self.assertEqual(parse_frames({"video_frames": ["jpeg"], "video_capture_ms": 1900}, 1000, 5, 1000),
                         (["jpeg"], None, 1900))
        self.assertEqual(parse_frames({}, 1000, 5, 1000), ([], None, 1000))
