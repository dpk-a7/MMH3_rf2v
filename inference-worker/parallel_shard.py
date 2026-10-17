import contextlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from memory_manager import (
    BYPASSED_BUSY,
    MIN_USABLE_VRAM_MB,
    PinnedBlockStreamer,
    StageSchedule,
    device_scope,
)


class ParallelContext:
    def __init__(self) -> None:
        self.rank: int = 0
        self.world: int = 1
        self.local_rank: int = 0
        self.device: torch.device = torch.device("cpu")
        self.weights: List[float] = [1.0]

    def active(self) -> bool:
        return self.world > 1 and dist.is_available() and dist.is_initialized()


CTX = ParallelContext()


def compute_capacity_weights(
    usable_vram_mb: Sequence[float],
    min_usable_mb: float = MIN_USABLE_VRAM_MB,
) -> List[float]:
    """Compute normalized capacity weights w_i across healthy GPU subset A.

    GPUs with usable_vram_mb < min_usable_mb are bypassed (w_i = 0.0).
    """
    filtered = [float(v) if float(v) >= float(min_usable_mb) else 0.0 for v in usable_vram_mb]
    total = sum(filtered)
    if total <= 0.0:
        return [0.0 for _ in filtered]
    return [v / total for v in filtered]


def compute_proportional_block_counts(
    num_blocks: int,
    weights: Sequence[float],
) -> List[int]:
    """Distribute `num_blocks` across GPUs in proportion to `weights` (0 for bypassed GPUs)."""
    total_blocks = max(0, int(num_blocks))
    n = len(weights)
    if n == 0 or total_blocks == 0:
        return [0] * n

    pos = [max(0.0, float(w)) for w in weights]
    w_sum = sum(pos)
    if w_sum <= 0.0:
        return [0] * n
    norm = [w / w_sum for w in pos]

    raw = [w * total_blocks for w in norm]
    counts = [int(math.floor(r)) for r in raw]
    remainder = total_blocks - sum(counts)

    # Ensure any active GPU with non-zero weight and enough blocks gets at least 1 block if possible
    order = sorted(
        [i for i in range(n) if norm[i] > 0.0],
        key=lambda i: (raw[i] - counts[i], norm[i], -i),
        reverse=True,
    )
    for idx in range(remainder):
        counts[order[idx % len(order)]] += 1

    return counts


def compute_weighted_split_sizes(
    total_dim: int,
    weights: Sequence[float],
    head_dim: int = 1,
) -> List[int]:
    """Allocate slice dimensions proportional to w_i aligned to a multiple of `head_dim`.

    Bypassed GPUs (w_i == 0) receive 0. Sum of returned split_sizes equals `total_dim`.
    """
    total_dim = int(total_dim)
    align = max(1, int(head_dim))
    if total_dim % align != 0:
        raise ValueError(f"total_dim ({total_dim}) must be divisible by head_dim ({align})")
    units = total_dim // align
    unit_counts = compute_proportional_block_counts(units, weights)
    return [c * align for c in unit_counts]


def assign_blocks_to_devices(
    num_blocks: int,
    schedule: StageSchedule,
) -> Tuple[List[torch.device], Dict[int, List[int]]]:
    """Return (device_per_block, blocks_by_gpu_index) proportional to schedule.gpus weights."""
    weights = [g.weight if g.state != BYPASSED_BUSY else 0.0 for g in schedule.gpus]
    counts = compute_proportional_block_counts(num_blocks, weights)
    if sum(counts) == 0:
        lead = schedule.lead_device
        lead_idx = lead.index if lead.type == "cuda" and lead.index is not None else 0
        return [lead] * num_blocks, {lead_idx: list(range(num_blocks))}

    device_per_block: List[torch.device] = []
    blocks_by_gpu: Dict[int, List[int]] = {g.index: [] for g in schedule.gpus}
    cursor = 0
    for gpu, cnt in zip(schedule.gpus, counts):
        for b_idx in range(cursor, cursor + cnt):
            device_per_block.append(gpu.device)
            blocks_by_gpu[gpu.index].append(b_idx)
        cursor += cnt
    return device_per_block, blocks_by_gpu


class ColumnParallelLinear(nn.Module):
    """Heterogeneous Column-Parallel Linear layer with head_dim-aligned proportional split_sizes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Sequence[float],
        rank: int,
        head_dim: int = 1,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.head_dim = max(1, int(head_dim))
        self.split_sizes = compute_weighted_split_sizes(out_features, weights, self.head_dim)
        self.local_out = self.split_sizes[self.rank] if 0 <= self.rank < len(self.split_sizes) else 0
        self.offset = sum(self.split_sizes[: self.rank])
        tkw = {"dtype": dtype or torch.float32, "device": device or torch.device("cpu")}
        self.weight = nn.Parameter(torch.empty(self.local_out, self.in_features, **tkw), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.local_out, **tkw), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weights: Sequence[float],
        rank: int,
        head_dim: int = 1,
    ) -> "ColumnParallelLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            weights=weights,
            rank=rank,
            head_dim=head_dim,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        start, end = layer.offset, layer.offset + layer.local_out
        with torch.no_grad():
            if layer.local_out > 0:
                layer.weight.copy_(linear.weight[start:end])
                if linear.bias is not None and layer.bias is not None:
                    layer.bias.copy_(linear.bias[start:end])
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_out == 0:
            return x.new_zeros((*x.shape[:-1], 0))
        w = self.weight.to(device=x.device, dtype=x.dtype)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class RowParallelLinear(nn.Module):
    """Heterogeneous Row-Parallel Linear layer with head_dim-aligned proportional split_sizes."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Sequence[float],
        rank: int,
        head_dim: int = 1,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        process_group=None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.head_dim = max(1, int(head_dim))
        self.process_group = process_group
        self.split_sizes = compute_weighted_split_sizes(in_features, weights, self.head_dim)
        self.local_in = self.split_sizes[self.rank] if 0 <= self.rank < len(self.split_sizes) else 0
        self.offset = sum(self.split_sizes[: self.rank])
        tkw = {"dtype": dtype or torch.float32, "device": device or torch.device("cpu")}
        self.weight = nn.Parameter(torch.empty(self.out_features, self.local_in, **tkw), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **tkw), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weights: Sequence[float],
        rank: int,
        head_dim: int = 1,
        process_group=None,
    ) -> "RowParallelLinear":
        layer = cls(
            linear.in_features,
            linear.out_features,
            weights=weights,
            rank=rank,
            head_dim=head_dim,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
            process_group=process_group,
        )
        start, end = layer.offset, layer.offset + layer.local_in
        with torch.no_grad():
            if layer.local_in > 0:
                layer.weight.copy_(linear.weight[:, start:end])
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.in_features:
            x_local = x[..., self.offset : self.offset + self.local_in]
        else:
            x_local = x
        if self.local_in == 0:
            out = x.new_zeros((*x.shape[:-1], self.out_features))
        else:
            w = self.weight.to(device=x.device, dtype=x.dtype)
            out = F.linear(x_local, w, None)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.process_group)
        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)
        return out


def weighted_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weights: Sequence[float],
    rank: int,
    head_dim: int,
    group=None,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """Weighted Ulysses Context Parallelism using `dist.all_to_all_single` with uneven split_sizes.

    Args:
        q, k, v: [B, S_local, H_total, D] (or full [B, S_total, H_total, D] before slicing)
        weights: normalized capacity weights per rank
        rank: current rank index
        head_dim: attention head dimension D

    Returns:
        (out_local [B, S_local, H_total * D], input_split_sizes, output_split_sizes)
    """
    b, s_in, h_total, d = q.shape
    head_splits_bytes = compute_weighted_split_sizes(h_total * d, weights, head_dim=head_dim)
    head_splits = [hs // head_dim for hs in head_splits_bytes]
    local_heads = head_splits[rank]

    # Sequence split sizes across ranks
    seq_splits = compute_weighted_split_sizes(s_in, weights, head_dim=1) if s_in > len(weights) else [s_in] * len(weights)
    s_total = sum(seq_splits)

    # input_split_sizes: sending local sequence chunk split by target rank's head allocation
    input_split_sizes = [hs for hs in head_splits]
    # output_split_sizes: receiving sequence chunks from each rank for our `local_heads`
    output_split_sizes = [seq_len for seq_len in seq_splits]

    if dist.is_available() and dist.is_initialized() and len(weights) > 1:
        def _exchange_seq_to_heads(t: torch.Tensor) -> torch.Tensor:
            # Rearrange so head dimension is leading for all_to_all_single
            # t: [B, S_local, H_total, D] -> [H_total, B, S_local, D]
            t_perm = t.permute(2, 0, 1, 3).contiguous()
            recv = torch.empty((s_total, b, local_heads, d), dtype=t.dtype, device=t.device)
            in_elems = [hs * b * t.shape[1] * d for hs in input_split_sizes]
            out_elems = [sl * b * local_heads * d for sl in output_split_sizes]
            dist.all_to_all_single(
                recv.view(-1),
                t_perm.view(-1),
                output_split_sizes=out_elems,
                input_split_sizes=in_elems,
                group=group,
            )
            return recv.permute(1, 2, 0, 3).contiguous()  # [B, local_heads, S_total, D]

        q_h = _exchange_seq_to_heads(q)
        k_h = _exchange_seq_to_heads(k)
        v_h = _exchange_seq_to_heads(v)
        attn = F.scaled_dot_product_attention(q_h, k_h, v_h)  # [B, local_heads, S_total, D]

        # Reverse all_to_all_single: [S_total, B, local_heads, D] -> [H_total, B, S_local, D]
        send_back = attn.permute(2, 0, 1, 3).contiguous()
        s_local = seq_splits[rank]
        recv_back = torch.empty((h_total, b, s_local, d), dtype=q.dtype, device=q.device)
        in_back = [sl * b * local_heads * d for sl in output_split_sizes]
        out_back = [hs * b * s_local * d for hs in input_split_sizes]
        dist.all_to_all_single(
            recv_back.view(-1),
            send_back.view(-1),
            output_split_sizes=out_back,
            input_split_sizes=in_back,
            group=group,
        )
        out = recv_back.permute(1, 2, 0, 3).reshape(b, s_local, h_total * d)
        return out, input_split_sizes, output_split_sizes

    # Single-process / unit-test execution path verifying uneven split_sizes and math
    h_start = sum(head_splits[:rank])
    h_end = h_start + local_heads
    if local_heads == 0:
        return q.new_zeros((b, q.shape[1], 0)), input_split_sizes, output_split_sizes
    q_sub = q[:, :, h_start:h_end, :].permute(0, 2, 1, 3)
    k_sub = k[:, :, h_start:h_end, :].permute(0, 2, 1, 3)
    v_sub = v[:, :, h_start:h_end, :].permute(0, 2, 1, 3)
    attn_sub = F.scaled_dot_product_attention(q_sub, k_sub, v_sub)
    out = attn_sub.permute(0, 2, 1, 3).reshape(b, q.shape[1], local_heads * d)
    return out, input_split_sizes, output_split_sizes


def _move_arg_to_device(val, device: torch.device):
    if isinstance(val, torch.Tensor):
        try:
            return val.to(device, non_blocking=True) if val.device != device else val
        except Exception:
            return val
    if isinstance(val, list):
        return [_move_arg_to_device(x, device) for x in val]
    if isinstance(val, tuple):
        return tuple(_move_arg_to_device(x, device) for x in val)
    if isinstance(val, dict):
        return {k: _move_arg_to_device(v, device) for k, v in val.items()}
    return val


def _wrap_block_for_device(
    block: nn.Module,
    target_device: torch.device,
    return_device: Optional[torch.device] = None,
) -> None:
    """Wrap a transformer block so its inputs hop to `target_device` and optional output returns to `return_device`."""
    if not hasattr(block, "_mmh3_orig_forward"):
        block._mmh3_orig_forward = block.forward
    block._mmh3_target_device = target_device
    block._mmh3_return_device = return_device
    orig_forward = block._mmh3_orig_forward

    def _forward(*args, **kwargs):
        dev = getattr(block, "_mmh3_target_device", target_device)
        ret_dev = getattr(block, "_mmh3_return_device", None)
        moved_args = _move_arg_to_device(args, dev)
        moved_kwargs = _move_arg_to_device(kwargs, dev)
        with device_scope(dev):
            out = orig_forward(*moved_args, **moved_kwargs)
        if ret_dev is not None and ret_dev != dev:
            out = _move_arg_to_device(out, ret_dev)
        return out

    block.forward = _forward
    block._mmh3_device_wrapped = True


def _extract_block_list(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the primary transformer block stack on MiniMaxH3Model or Qwen3-VL CLIP."""
    for attr_path in (
        "blocks",
        "diffusion_model.blocks",
        "model.diffusion_model.blocks",
        "layers",
        "model.layers",
        "cond_stage_model.qwen3vl_32b.transformer.model.layers",
        "cond_stage_model.transformer.model.layers",
    ):
        cur = model
        for part in attr_path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
            return cur
    return None


def apply_proportional_block_sharding(
    model: nn.Module,
    schedule: StageSchedule,
) -> Dict[str, object]:
    """Distribute transformer blocks across healthy GPUs proportional to w_i, or attach PinnedBlockStreamer.

    Returns a summary dictionary describing the applied placement or streaming plan:
      - `mode`: `"proportional_sharding"` or `"pinned_block_streaming"`
      - `lead_device`: elected lead device (`argmax(usable_vram)`)
      - `blocks_by_gpu`: `{gpu_idx: [block_indices]}` (0 blocks for `BYPASSED_BUSY` GPUs)
      - `streamer`: `PinnedBlockStreamer` instance if streaming fallback is active, else `None`
    """
    blocks = _extract_block_list(model)
    lead_device = schedule.lead_device

    if blocks is None or len(blocks) == 0:
        return {
            "mode": "lead_only",
            "lead_device": lead_device,
            "blocks_by_gpu": {},
            "streamer": None,
        }

    num_blocks = len(blocks)

    # Case C: Total usable VRAM across healthy GPUs < M_stage -> Pinned-RAM Double-Buffered Block Streaming
    if schedule.stream_from_pinned_ram:
        streamer = PinnedBlockStreamer(
            blocks=blocks,
            lead_device=lead_device,
            batch_size=schedule.blocks_per_stream_batch,
        )
        setattr(model, "_mmh3_block_streamer", streamer)
        setattr(model, "_mmh3_lead_device", lead_device)
        return {
            "mode": "pinned_block_streaming",
            "lead_device": lead_device,
            "blocks_by_gpu": { g.index: [] for g in schedule.gpus },
            "streamer": streamer,
        }

    # Cases A & B: Capacity-Weighted Proportional Block Sharding across healthy GPU subset A
    setattr(model, "_mmh3_block_streamer", None)
    setattr(model, "_mmh3_lead_device", lead_device)
    device_per_block, blocks_by_gpu = assign_blocks_to_devices(num_blocks, schedule)

    for b_idx, (blk, dev) in enumerate(zip(blocks, device_per_block)):
        try:
            blk.to(dev)
        except Exception:
            pass
        ret_dev = lead_device if b_idx == (num_blocks - 1) else None
        _wrap_block_for_device(blk, dev, return_device=ret_dev)

    # Also move non-block stem/head modules onto lead_device (never hardcoded to cuda:0)
    block_ids = {id(b) for b in blocks}
    for name, child in model.named_children():
        if id(child) in block_ids or child is blocks:
            continue
        try:
            child.to(lead_device)
        except Exception:
            pass

    return {
        "mode": "proportional_sharding",
        "lead_device": lead_device,
        "blocks_by_gpu": blocks_by_gpu,
        "device_per_block": device_per_block,
        "streamer": None,
    }
