import base64
import unittest

import numpy as np

from duplex_avatar.video_transport import av, encode_h264


@unittest.skipIf(av is None, "Use the avatar environment for the PyAV codec test")
class VideoTransportTests(unittest.TestCase):
    def test_blocks_decode_independently_and_preserve_every_frame(self):
        for offset in (0, 80):
            frames = np.zeros((24, 64, 64, 3), dtype=np.uint8)
            for index in range(24):
                frames[index] = offset + index * 3
            block = encode_h264(frames, 25, backend="pyav")
            self.assertEqual(block["frame_count"], 24)
            self.assertEqual(len(block["chunks"]), 24)
            decoder = av.CodecContext.create("h264", "r")
            decoded = []
            for chunk in block["chunks"]:
                decoded.extend(decoder.decode(av.Packet(base64.b64decode(chunk))))
            decoded.extend(decoder.decode(None))
            self.assertEqual(len(decoded), 24)
            self.assertTrue(decoded[0].key_frame)
            means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in decoded]
            np.testing.assert_allclose(means, [offset + i * 3 for i in range(24)], atol=4)
