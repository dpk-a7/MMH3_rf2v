import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_h3 as p  # noqa: E402


class FakeRegistry:
    plan = {slot: torch.device("cpu") for slot in ("clip", "video_vae", "audio_vae", "diffusion", "lora", "controlnet")}

    def cached(self, slot, key):
        return False

    def get_or_swap(self, slot, subdir, name, loader, key=None, keep=()):
        return slot

    def trim_all_memory(self):
        pass


class SequenceTests(unittest.TestCase):
    def run_sequence(self, segments, frames=22):
        guides, muxed = [], {}

        def render(payload, *args):
            index = len(guides)
            guides.append(args[-1])
            video = torch.full((frames, 4, 4, 3), index / 10.0)
            video[-1] = 0.9  # a distinct last frame
            return video, torch.full((frames * p.AUDIO_PER_FRAME + 7, 2), float(index))

        def mux(video, audio, *args):
            muxed.update(video=video, audio=audio)

        payload = {"prompt": "one", "segments": segments, "keyframes": {"first": "a.png", "last": "b.png"},
                   "output_path": "out.mp4"}
        models = {"clip": "c", "video_vae": "v", "audio_vae": "a"}
        with mock.patch.object(p, "_render_segment", side_effect=render) as rendered, mock.patch.object(p, "_mux", mux):
            p._run_job_inner(payload, FakeRegistry(), lambda *a: None, lambda: None, models, "normal", 4, 4, frames,
                             24, 1, 8, 1.0, "simple", "match", {}, 1.0, None, None, None, 0)
        return rendered, guides, muxed

    def test_single_prompt_is_one_unguided_segment(self):
        rendered, guides, muxed = self.run_sequence([])
        self.assertEqual(guides, [None])
        self.assertEqual(muxed["video"].shape[0], 22)
        self.assertEqual(muxed["audio"].shape[0], 22 * p.AUDIO_PER_FRAME)

    def test_continuations_are_guided_by_the_previous_tail_and_drop_the_overlap(self):
        rendered, guides, muxed = self.run_sequence(["two", "three"])
        self.assertIsNone(guides[0])
        tail_frames, tail_audio = guides[1]
        self.assertEqual(tail_frames.shape[0], p.GUIDE_FRAMES)
        self.assertTrue(torch.all(tail_frames[-1] == 0.9))  # ends on the previous segment's last frame
        self.assertEqual(tail_audio.shape[0], p.GUIDE_FRAMES * p.AUDIO_PER_FRAME)
        total = 22 + 2 * (22 - p.GUIDE_FRAMES)
        self.assertEqual(muxed["video"].shape[0], total)
        self.assertEqual(muxed["audio"].shape[0], total * p.AUDIO_PER_FRAME)
        prompts = [call.args[0]["prompt"] for call in rendered.call_args_list]
        self.assertEqual(prompts, ["one", "two", "three"])
        keyframes = [call.args[0]["keyframes"] for call in rendered.call_args_list]
        self.assertEqual(keyframes[0], {"first": "a.png", "last": None})
        self.assertEqual(keyframes[2], {"first": None, "last": "b.png"})


if __name__ == "__main__":
    unittest.main()
