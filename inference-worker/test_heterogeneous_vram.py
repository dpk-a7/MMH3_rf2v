import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_manager import (  # noqa: E402
    ACTIVE_FULL,
    ACTIVE_TIGHT,
    BYPASSED_BUSY,
    MIB,
    build_stage_schedule,
    estimate_stage_memory_mb,
    probe_live_gpus,
)
from parallel_shard import (  # noqa: E402
    apply_proportional_block_sharding,
    compute_capacity_weights,
    compute_proportional_block_counts,
    compute_weighted_split_sizes,
)


def _bytes_gb(gb: float) -> int:
    return int(gb * 1024 * MIB)


class DummyStack(nn.Module):
    def __init__(self, n: int = 40):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(n)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class HeterogeneousVramTests(unittest.TestCase):
    def _patch_mem(self, free_gb):
        totals = [24.0] * len(free_gb)

        def mem_get_info(device_id):
            idx = int(device_id)
            return _bytes_gb(free_gb[idx]), _bytes_gb(totals[idx])

        return patch.object(torch.cuda, "mem_get_info", side_effect=mem_get_info)

    def test_case_a_gpu0_busy_split_50_50(self):
        """GPU0 1.2 GB busy; GPU1/GPU2 22 GB free -> 0% / 50% / 50%."""
        with self._patch_mem([1.2, 22.0, 22.0]):
            gpus = probe_live_gpus(num_devices=3)
        self.assertEqual([g.state for g in gpus], [BYPASSED_BUSY, ACTIVE_FULL, ACTIVE_FULL])
        self.assertEqual([g.role for g in gpus], ["bypassed_busy", "active_full", "active_full"])
        self.assertEqual(gpus[0].weight, 0.0)
        self.assertAlmostEqual(gpus[1].weight, 0.5, places=3)
        self.assertAlmostEqual(gpus[2].weight, 0.5, places=3)
        self.assertEqual(gpus[0].allocated_weight_pct, 0.0)
        self.assertAlmostEqual(gpus[1].allocated_weight_pct, 50.0, places=0)
        self.assertAlmostEqual(gpus[2].allocated_weight_pct, 50.0, places=0)

        counts = compute_proportional_block_counts(40, [g.weight for g in gpus])
        self.assertEqual(counts, [0, 20, 20])

        with self._patch_mem([1.2, 22.0, 22.0]):
            schedule = build_stage_schedule("stage2", required_vram_mb=8000, num_devices=3)
        self.assertFalse(schedule.stream_from_pinned_ram)
        self.assertEqual(str(schedule.lead_device), "cuda:1")
        model = DummyStack(40)
        summary = apply_proportional_block_sharding(model, schedule)
        self.assertEqual(summary["mode"], "proportional_sharding")
        self.assertEqual(len(summary["blocks_by_gpu"].get(0, [])), 0)
        self.assertEqual(len(summary["blocks_by_gpu"].get(1, [])), 20)
        self.assertEqual(len(summary["blocks_by_gpu"].get(2, [])), 20)

    def test_case_b_proportional_81_19(self):
        """GPU0 20 GB, GPU1 6 GB tight, GPU2 1 GB bypassed -> ~81% / ~19% / 0%."""
        with self._patch_mem([20.0, 6.0, 1.0]):
            gpus = probe_live_gpus(num_devices=3)
        self.assertEqual(gpus[0].state, ACTIVE_FULL)
        self.assertEqual(gpus[1].state, ACTIVE_TIGHT)
        self.assertEqual(gpus[2].state, BYPASSED_BUSY)
        self.assertEqual(gpus[2].weight, 0.0)
        self.assertAlmostEqual(gpus[0].allocated_weight_pct, 80.4, delta=2.0)
        self.assertAlmostEqual(gpus[1].allocated_weight_pct, 19.6, delta=2.0)
        counts = compute_proportional_block_counts(100, [g.weight for g in gpus])
        self.assertEqual(counts[2], 0)
        self.assertIn(counts[0], range(79, 83))
        self.assertIn(counts[1], range(18, 22))
        self.assertEqual(sum(counts), 100)

        with self._patch_mem([20.0, 6.0, 1.0]):
            schedule = build_stage_schedule("stage2", required_vram_mb=12000, num_devices=3)
        self.assertEqual(str(schedule.lead_device), "cuda:0")
        self.assertFalse(schedule.stream_from_pinned_ram)
        model = DummyStack(100)
        summary = apply_proportional_block_sharding(model, schedule)
        self.assertEqual(summary["mode"], "proportional_sharding")
        self.assertEqual(len(summary["blocks_by_gpu"].get(2, [])), 0)

    def test_case_c_all_tight_pinned_streaming(self):
        """All GPUs 4.5 GB free -> usable below M_stage, pinned block streaming."""
        required = estimate_stage_memory_mb("stage2")
        with self._patch_mem([4.5, 4.5, 4.5]):
            gpus = probe_live_gpus(num_devices=3)
            schedule = build_stage_schedule("stage2", required_vram_mb=required, num_devices=3)
        self.assertTrue(all(g.state == ACTIVE_TIGHT for g in gpus))
        self.assertLess(schedule.total_usable_vram_mb, required)
        self.assertTrue(schedule.stream_from_pinned_ram)
        model = DummyStack(6)
        x = torch.randn(2, 4)
        summary = apply_proportional_block_sharding(model, schedule)
        self.assertEqual(summary["mode"], "pinned_block_streaming")
        self.assertIsNotNone(summary["streamer"])
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 4))
        self.assertGreaterEqual(summary["streamer"].prefetch_events, 1)
        self.assertGreaterEqual(summary["streamer"].offload_events, 1)

    def test_weighted_split_sizes_head_aligned(self):
        splits = compute_weighted_split_sizes(32, [0.0, 0.5, 0.5], head_dim=8)
        self.assertEqual(splits, [0, 16, 16])
        splits = compute_weighted_split_sizes(96, [0.81, 0.19, 0.0], head_dim=8)
        self.assertEqual(sum(splits), 96)
        self.assertTrue(all(s % 8 == 0 for s in splits))
        self.assertEqual(splits[2], 0)

    def test_capacity_weights_bypass_busy(self):
        weights = compute_capacity_weights([0.0, 20992.0, 20992.0])
        self.assertEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[1], 0.5)
        self.assertAlmostEqual(weights[2], 0.5)


if __name__ == "__main__":
    unittest.main()
