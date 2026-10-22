import inspect
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_h3 as p  # noqa: E402


class Stage1ClipVramTests(unittest.TestCase):
    def test_one_ref_image_reserves_9gb(self):
        img = torch.zeros(1, 480, 864, 3)
        tokens = {"qwen3vl_32b": [[({"type": "image", "data": img}, 1.0), (1, 1.0)]]}
        need = p.clip_activation_bytes(tokens)
        self.assertGreaterEqual(need, 9 * p.GIB)
        self.assertGreater(16 * p.GIB + need, 20 * p.GIB)

    def test_clip_load_without_tokens_still_reserves_9gb(self):
        self.assertGreaterEqual(p.clip_activation_bytes({}), 9 * p.GIB)

    def test_dit_and_lora_load_after_stage1_encode(self):
        src = inspect.getsource(p._render_segment)
        self.assertLess(src.index("_unload_device(enc_dev)"), src.index("_stage1_condition"))
        self.assertLess(src.index("_stage1_condition"), src.index("_load_unet"))
        self.assertLess(src.index("_stage1_condition"), src.index("_load_lora"))

    def test_cuda_oom_detects_wrapped_and_cublas(self):
        self.assertTrue(p._cuda_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")))
        try:
            try:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            except RuntimeError as exc:
                raise RuntimeError("encode failed") from exc
        except RuntimeError as exc:
            self.assertTrue(p._cuda_oom(exc))
        self.assertFalse(p._cuda_oom(ValueError("bad reference video")))


if __name__ == "__main__":
    unittest.main()
