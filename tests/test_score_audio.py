import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.score_audio import write_wav


class WriteWavTest(unittest.TestCase):
    def test_pcm16_round_trip_keeps_rate_channels_and_samples(self):
        # av-benchmark reads these files, so they must hold the decoded track at its native layout.
        x = np.stack([np.linspace(-1, 1, 3200), np.linspace(1, -1, 3200)]).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'a.wav'
            write_wav(path, x, 32000)
            with wave.open(str(path)) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()), (2, 2, 32000, 3200))
                pcm = np.frombuffer(w.readframes(3200), '<i2').reshape(-1, 2).T
        np.testing.assert_array_equal(pcm, (np.clip(x, -1, 1) * 32767).astype('<i2'))


if __name__ == '__main__':
    unittest.main()
