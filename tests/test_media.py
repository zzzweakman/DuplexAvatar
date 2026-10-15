import base64
import unittest
import numpy as np
from duplex_avatar.media import AudioChunks, decode_pcm, encode_pcm


class MediaTests(unittest.TestCase):
    def test_arbitrary_model_chunks_preserve_every_sample(self):
        original = np.random.default_rng(42).standard_normal(78139).astype(np.float32)
        chunks = AudioChunks(23040)
        emitted = []
        cursor = 0
        for length in (960, 1479, 24000, 7200, 18901, 25600):
            emitted += chunks.append(original[cursor:cursor + length])
            cursor += length
        tail = chunks.flush()
        if tail is not None:
            emitted.append(tail)
        self.assertTrue(np.array_equal(original, np.concatenate(emitted)))
        self.assertIsNone(chunks.flush())

    def test_interruption_discards_only_unsubmitted_tail(self):
        chunks = AudioChunks(23040)
        chunks.append(np.ones(9000, dtype=np.float32))
        chunks.clear()
        result = chunks.append(np.full(23040, 2, dtype=np.float32))
        self.assertTrue(np.all(result[0] == 2))

    def test_pcm_transport_round_trip(self):
        original = np.linspace(-1, 1, 16000, dtype=np.float32)
        self.assertTrue(np.array_equal(original, decode_pcm(encode_pcm(original))))

    def test_invalid_pcm_is_rejected(self):
        for payload in ("!!", base64.b64encode(b"abc").decode(), encode_pcm([float("nan")]), encode_pcm([float("inf")])):
            with self.assertRaises(ValueError):
                decode_pcm(payload)
        with self.assertRaises(ValueError):
            decode_pcm(encode_pcm(np.zeros(11)), max_samples=10)


if __name__ == "__main__":
    unittest.main()
