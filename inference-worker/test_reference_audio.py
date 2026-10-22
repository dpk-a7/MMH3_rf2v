import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_h3 as p  # noqa: E402


def _make_video(path, with_audio):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=f=440:sample_rate=48000", "-shortest"]
    subprocess.run(cmd + ["-t", "1", path], check=True, timeout=60)


class ReferenceVideoAudioTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        os.environ["EPHEMERAL_DIR"] = self._dir.name
        # The host RAM gate is covered elsewhere; keep this test independent of the machine.
        gate = mock.patch.object(p, "check_host_gate")
        gate.start()
        self.addCleanup(gate.stop)

    def tearDown(self):
        self._dir.cleanup()

    def _pack(self, with_audio):
        path = os.path.join(self._dir.name, "ref.mp4")
        _make_video(path, with_audio)
        refs = {"videos": [path], "video_audios": [None]}
        return p._reference_pack(refs, 64, 64, 124, "match")

    def test_video_soundtrack_is_the_default_paired_audio(self):
        _images, videos, video_audios, _audios = self._pack(with_audio=True)
        self.assertIn("ref_video_0", videos)
        self.assertEqual(video_audios["ref_video_audio_0"]["sample_rate"], 48000)

    def test_i2v_canvas_keeps_the_first_frame_aspect(self):
        from PIL import Image

        path = os.path.join(self._dir.name, "first.png")
        Image.new("RGB", (896, 1152)).save(path)
        self.assertEqual(p._first_frame_canvas(path, 864 * 480), (576, 736))

    def test_silent_video_is_accepted_without_audio(self):
        _images, videos, video_audios, _audios = self._pack(with_audio=False)
        self.assertIn("ref_video_0", videos)
        self.assertEqual(video_audios, {})


if __name__ == "__main__":
    unittest.main()
