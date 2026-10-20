import unittest
from unittest.mock import patch

import torch
from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

import pipeline_h3 as pipeline
from memory_manager import StageSchedule


class TinyStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


class PatcherSubclass(ModelPatcher):
    pass


class RecordingRegistry:
    def __init__(self, schedule):
        self.schedule = schedule
        self.trims = 0
        self.probes = []

    def trim_all_memory(self):
        self.trims += 1

    def probe_and_plan(self, stage):
        self.probes.append(stage)
        return self.schedule


class ComfyOffloadTests(unittest.TestCase):
    def setUp(self):
        self.schedule = StageSchedule(
            stage="stage2", gpus=[], active_gpus=[], lead_device=torch.device("cpu"),
            weights_by_gpu={}, total_usable_vram_mb=0, required_vram_mb=1,
            stream_from_pinned_ram=True, blocks_per_stream_batch=1,
        )

    def test_comfy_patchers_keep_their_weight_storage_and_forward_hooks(self):
        for patcher_type in (ModelPatcher, PatcherSubclass):
            with self.subTest(patcher_type=patcher_type.__name__):
                model = TinyStack()
                patcher = patcher_type(model, torch.device("cpu"), torch.device("cpu"))
                pointers = [p.data_ptr() for p in model.parameters()]
                forwards = [block.forward for block in model.blocks]

                pipeline._apply_stage_shards(patcher, self.schedule, lambda *a: None, 0.28, "dit")

                self.assertEqual([p.data_ptr() for p in model.parameters()], pointers)
                self.assertEqual([block.forward for block in model.blocks], forwards)
                self.assertFalse(hasattr(model, "_mmh3_block_streamer"))

    def sample(self, model, registry):
        return pipeline._stage2_sample(
            model, [], {}, 1, 1.0, "simple", 1, lambda: None, lambda *a: None, registry,
        )

    def test_native_cuda_oom_does_not_build_a_second_weight_cache(self):
        model = TinyStack()
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        registry = RecordingRegistry(self.schedule)
        oom = torch.cuda.OutOfMemoryError("CUDA out of memory.")
        with patch.object(MiniMaxH3SigmaShift, "execute", side_effect=oom) as execute:
            with self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
                self.sample(patcher, registry)
        self.assertIs(caught.exception, oom)
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(registry.trims, 0)
        self.assertEqual(registry.probes, [])
        self.assertFalse(hasattr(model, "_mmh3_block_streamer"))

    def test_raw_modules_retain_the_existing_single_streaming_retry(self):
        model = TinyStack()
        registry = RecordingRegistry(self.schedule)
        oom = torch.cuda.OutOfMemoryError("CUDA out of memory.")
        retry_result = ValueError("second sampling attempt reached")
        with patch.object(MiniMaxH3SigmaShift, "execute", side_effect=[oom, retry_result]) as execute:
            with self.assertRaises(ValueError) as caught:
                self.sample(model, registry)
        self.assertIs(caught.exception, retry_result)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(registry.trims, 1)
        self.assertEqual(registry.probes, ["stage2_oom"])
        self.assertTrue(hasattr(model, "_mmh3_block_streamer"))


if __name__ == "__main__":
    unittest.main()
