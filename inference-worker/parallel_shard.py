import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


class ParallelContext:
    def __init__(self):
        self.rank = 0
        self.world = 1
        self.local_rank = 0
        self.device = torch.device("cpu")

    def active(self):
        return self.world > 1 and dist.is_available() and dist.is_initialized()


CTX = ParallelContext()


def shard_first_dim(tensor, rank, world):
    if world <= 1 or tensor.ndim == 0 or tensor.shape[0] < world or tensor.shape[0] % world != 0:
        return tensor.contiguous()
    return tensor.chunk(world, dim=0)[rank].contiguous()


def sync_error(local):
    """Return the rank-0 error string when any rank failed. Collective when N>1."""
    if not CTX.active():
        return local
    holder = [local if CTX.rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    flag = torch.tensor(
        [0 if local is None else 1],
        device=CTX.device if CTX.device.type == "cuda" else "cpu",
        dtype=torch.int32,
    )
    if CTX.device.type == "cuda":
        flag = flag.to(CTX.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if int(flag.item()) == 0:
        return None
    return holder[0] or local or "inference_failed"


def collective_abort(local_abort):
    if not CTX.active():
        return bool(local_abort)
    flag = torch.tensor(
        [1 if local_abort else 0],
        device=CTX.device if CTX.device.type == "cuda" else "cpu",
        dtype=torch.int32,
    )
    if CTX.device.type == "cuda":
        flag = flag.to(CTX.device)
    dist.broadcast(flag, src=0)
    return int(flag.item()) == 1


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world, rank, weight=None, bias=None):
        super().__init__()
        if out_features % world != 0:
            raise RuntimeError("column_parallel_shape")
        self.world = world
        self.rank = rank
        local_out = out_features // world
        if weight is None:
            self.weight = torch.nn.Parameter(torch.empty(local_out, in_features))
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias = torch.nn.Parameter(torch.zeros(local_out))
        else:
            sliced = weight.detach().contiguous()
            if tuple(sliced.shape) != (local_out, in_features):
                raise RuntimeError("column_parallel_shape")
            self.weight = torch.nn.Parameter(sliced)
            if bias is None:
                self.bias = torch.nn.Parameter(torch.zeros(local_out, dtype=sliced.dtype))
            else:
                self.bias = torch.nn.Parameter(bias.detach().contiguous())

    def forward(self, inputs):
        w = self.weight if self.weight.dtype == inputs.dtype else self.weight.to(inputs.dtype)
        b = self.bias if self.bias.dtype == inputs.dtype else self.bias.to(inputs.dtype)
        return F.linear(inputs, w, b)


class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, world, rank, weight=None, bias=None):
        super().__init__()
        if in_features % world != 0:
            raise RuntimeError("row_parallel_shape")
        self.world = world
        self.rank = rank
        local_in = in_features // world
        if weight is None:
            self.weight = torch.nn.Parameter(torch.empty(out_features, local_in))
            torch.nn.init.xavier_uniform_(self.weight)
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            sliced = weight.detach().contiguous()
            if tuple(sliced.shape) != (out_features, local_in):
                raise RuntimeError("row_parallel_shape")
            self.weight = torch.nn.Parameter(sliced)
            if bias is None:
                self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=sliced.dtype))
            else:
                self.bias = torch.nn.Parameter(bias.detach().contiguous())

    def forward(self, inputs):
        w = self.weight if self.weight.dtype == inputs.dtype else self.weight.to(inputs.dtype)
        b = self.bias if self.bias.dtype == inputs.dtype else self.bias.to(inputs.dtype)
        local = F.linear(inputs, w, None)
        if self.world > 1 and CTX.active():
            dist.all_reduce(local)
        return local + b


def ulysses_attention(q, k, v, attend_fn):
    """Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated."""
    if not CTX.active():
        return attend_fn(q, k, v)
    world = CTX.world
    heads = q.shape[1]
    if heads % world != 0:
        return attend_fn(q, k, v)
    q, k, v, pad = _shard_sequence(q, k, v, world)
    q = _seq_to_head(q, world)
    k = _seq_to_head(k, world)
    v = _seq_to_head(v, world)
    attended = attend_fn(q, k, v)
    attended = _head_to_seq(attended, world)
    attended = _gather_sequence(attended, world)
    if pad:
        attended = attended[:, :, : attended.shape[2] - pad, :]
    return attended


def _shard_sequence(q, k, v, world):
    length = q.shape[2]
    pad = (world - (length % world)) % world
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    local = q.shape[2] // world
    start = CTX.rank * local
    end = start + local
    return (
        q[:, :, start:end, :].contiguous(),
        k[:, :, start:end, :].contiguous(),
        v[:, :, start:end, :].contiguous(),
        pad,
    )


def _seq_to_head(tensor, world):
    # [B, H, S/N, D] -> [B, H/N, S, D]
    batch, heads, local_s, dim = tensor.shape
    tensor = tensor.reshape(batch, world, heads // world, local_s, dim)
    tensor = tensor.permute(1, 0, 2, 3, 4).contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.reshape(batch, heads // world, world * local_s, dim)


def _head_to_seq(tensor, world):
    # [B, H/N, S, D] -> [B, H, S/N, D]
    batch, local_h, length, dim = tensor.shape
    local_s = length // world
    tensor = tensor.reshape(batch, local_h, world, local_s, dim)
    tensor = tensor.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor)
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    return output.reshape(batch, world * local_h, local_s, dim)


def _gather_sequence(tensor, world):
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=2)


def block_sparse_attention(q, k, v, step_percent):
    """Exact sink rows/keys, block-sparse body past 12288 tokens.

    tau = 1.3, start_percent = 0.2, end_percent = 0.9,
    sink_conditioning = exact_kv_and_rows.
    """
    length = q.shape[2]
    if length <= 12288:
        return F.scaled_dot_product_attention(q, k, v)
    if not (0.2 <= float(step_percent) <= 0.9):
        return _chunked_attention(q, k, v)
    sink = min(256, length)
    exact_rows = _chunked_attention(q[:, :, :sink, :], k, v)
    sparse = _block_body(q, k, v, sink=sink, tau=1.3)
    sparse = sparse.clone()
    sparse[:, :, :sink, :] = exact_rows
    return sparse


def _block_body(q, k, v, sink, tau):
    block = 64
    blocks = math.ceil(k.shape[2] / block)
    reps = []
    for index in range(blocks):
        start = index * block
        chunk = k[:, :, start : start + block, :]
        reps.append(chunk.mean(dim=2))
    rep = torch.stack(reps, dim=2)
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, rep.transpose(-1, -2)) * scale
    block_scores = scores.mean(dim=2)
    mean = block_scores.mean(dim=-1, keepdim=True)
    keep = block_scores >= (mean * tau)
    # Sink keys stay exact: the first block is always kept.
    keep[:, :, 0] = True
    max_keep = max(1, min(blocks, max(8, blocks // 8)))
    selected = []
    batch, heads, _, dim = q.shape
    for b in range(batch):
        head_rows = []
        for h in range(heads):
            idx = torch.nonzero(keep[b, h], as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=q.device)
            if idx.numel() > max_keep:
                ranked = block_scores[b, h, idx]
                top = torch.topk(ranked, k=max_keep).indices
                idx = idx[top]
                idx = torch.unique(torch.cat([idx, idx.new_zeros(1)]), sorted=True)
            parts = []
            for block_index in idx.tolist():
                start = int(block_index) * block
                parts.append(k[b, h, start : start + block, :])
            head_rows.append(torch.cat(parts, dim=0))
        width = max(row.shape[0] for row in head_rows)
        padded = []
        for row in head_rows:
            if row.shape[0] < width:
                row = F.pad(row, (0, 0, 0, width - row.shape[0]))
            padded.append(row)
        selected.append(torch.stack(padded, dim=0))
    key_sel = torch.stack(selected, dim=0)
    # Values follow the same block selection.
    value_sel = torch.zeros_like(key_sel)
    for b in range(batch):
        for h in range(heads):
            idx = torch.nonzero(keep[b, h], as_tuple=False).flatten()
            if idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=q.device)
            if idx.numel() > max_keep:
                ranked = block_scores[b, h, idx]
                top = torch.topk(ranked, k=max_keep).indices
                idx = idx[top]
                idx = torch.unique(torch.cat([idx, idx.new_zeros(1)]), sorted=True)
            parts = []
            for block_index in idx.tolist():
                start = int(block_index) * block
                parts.append(v[b, h, start : start + block, :])
            row = torch.cat(parts, dim=0)
            value_sel[b, h, : row.shape[0], :] = row
    # Force the exact sink key span into the front of the selected KV.
    sink_k = k[:, :, :sink, :]
    sink_v = v[:, :, :sink, :]
    key_sel = torch.cat([sink_k, key_sel], dim=2)
    value_sel = torch.cat([sink_v, value_sel], dim=2)
    return F.scaled_dot_product_attention(q, key_sel, value_sel)


_STEP_PERCENT = [0.0]


def note_step_percent(value):
    _STEP_PERCENT[0] = max(0.0, min(1.0, float(value)))


def _dense_weight(linear, target_dtype=torch.bfloat16):
    weight = getattr(linear, "weight", None)
    if weight is None:
        return None, None
    tensor = weight.detach()
    if hasattr(tensor, "dequantize"):
        tensor = tensor.dequantize().detach()
    scale = getattr(linear, "weight_scale", None)
    if scale is None:
        scale = getattr(tensor, "scale", None)
    if scale is not None and not tensor.dtype.is_floating_point:
        tensor = (tensor.to(torch.float32) * scale.detach().to(torch.float32)).to(target_dtype)
    elif not tensor.dtype.is_floating_point:
        tensor = tensor.to(target_dtype)
    if tensor.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        tensor = tensor.to(target_dtype)
    if tensor.dtype == torch.float32 and target_dtype in (torch.bfloat16, torch.float16):
        tensor = tensor.to(target_dtype)
    if tensor.ndim != 2:
        return None, None
    bias = getattr(linear, "bias", None)
    if bias is not None:
        bias = bias.detach().to(dtype=tensor.dtype).contiguous()
    return tensor.contiguous(), bias


def slice_qkv_out(weight, world, rank):
    """Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all three."""
    out, _inn = weight.shape
    if world <= 1:
        return weight.contiguous()
    if out % 3 != 0 or (out // 3) % world != 0:
        raise RuntimeError("column_parallel_shape")
    third = out // 3
    local = third // world
    parts = []
    for block in range(3):
        base = block * third + rank * local
        parts.append(weight[base : base + local])
    return torch.cat(parts, dim=0).contiguous()


def slice_gate_up(weight, world, rank):
    """Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both halves."""
    out, _inn = weight.shape
    if world <= 1:
        return weight.contiguous()
    if out % 2 != 0 or (out // 2) % world != 0:
        raise RuntimeError("column_parallel_shape")
    half = out // 2
    local = half // world
    gate = weight[rank * local : (rank + 1) * local]
    up = weight[half + rank * local : half + (rank + 1) * local]
    return torch.cat([gate, up], dim=0).contiguous()


def slice_row(weight, world, rank):
    _out, inn = weight.shape
    if world <= 1:
        return weight.contiguous()
    if inn % world != 0:
        raise RuntimeError("row_parallel_shape")
    local = inn // world
    return weight[:, rank * local : (rank + 1) * local].contiguous()


def _slice_paired_bias(bias, out_features, world, rank, parts):
    if bias is None:
        return None
    if world <= 1:
        return bias
    if bias.numel() != out_features or out_features % parts != 0:
        return None
    span = out_features // parts
    if span % world != 0:
        return None
    local = span // world
    chunks = []
    for block in range(parts):
        base = block * span + rank * local
        chunks.append(bias[base : base + local])
    return torch.cat(chunks, dim=0).contiguous()


def attach_tensor_parallel(root, world, rank):
    """Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-sliced parallel layers."""
    world = max(1, int(world))
    rank = int(rank)
    for module in root.modules():
        if hasattr(module, "qkv_proj") and hasattr(module, "out_proj") and hasattr(module, "heads"):
            _shard_attention(module, world, rank)
        elif (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
            and hasattr(module, "num_heads")
            and hasattr(module, "num_kv_heads")
        ):
            _shard_llama_attention(module, world, rank)
        elif module.__class__.__name__ == "MLP" and hasattr(module, "fc1") and hasattr(module, "fc2"):
            _shard_swiglu(module, world, rank)
        elif hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
            _shard_llama_swiglu(module, world, rank)


def _shard_attention(attn, world, rank):
    qkv_w, qkv_b = _dense_weight(attn.qkv_proj)
    out_w, out_b = _dense_weight(attn.out_proj)
    if qkv_w is None or out_w is None:
        return
    qkv = ColumnParallelLinear(
        qkv_w.shape[1],
        qkv_w.shape[0],
        world,
        rank,
        weight=slice_qkv_out(qkv_w, world, rank),
        bias=_slice_paired_bias(qkv_b, qkv_w.shape[0], world, rank, 3),
    )
    proj = RowParallelLinear(
        out_w.shape[1],
        out_w.shape[0],
        world,
        rank,
        weight=slice_row(out_w, world, rank),
        bias=out_b,
    )
    attn.qkv_proj = qkv
    attn.out_proj = proj
    attn.o_proj = proj
    if world > 1 and int(attn.heads) % world == 0:
        attn.heads = int(attn.heads) // world


def _shard_swiglu(mlp, world, rank):
    up_w, up_b = _dense_weight(mlp.fc1)
    down_w, down_b = _dense_weight(mlp.fc2)
    if up_w is None or down_w is None:
        return
    gate_up = ColumnParallelLinear(
        up_w.shape[1],
        up_w.shape[0],
        world,
        rank,
        weight=slice_gate_up(up_w, world, rank),
        bias=_slice_paired_bias(up_b, up_w.shape[0], world, rank, 2),
    )
    down = RowParallelLinear(
        down_w.shape[1],
        down_w.shape[0],
        world,
        rank,
        weight=slice_row(down_w, world, rank),
        bias=down_b,
    )
    mlp.fc1 = gate_up
    mlp.gate_up_proj = gate_up
    mlp.fc2 = down
    mlp.down_proj = down

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

    mlp.forward = forward.__get__(mlp, mlp.__class__)


def _shard_llama_attention(attn, world, rank):
    if world <= 1:
        return
    num_heads = int(getattr(attn, "num_heads", 0))
    num_kv_heads = int(getattr(attn, "num_kv_heads", 0))
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % world != 0 or num_kv_heads % world != 0:
        return
    q_w, q_b = _dense_weight(attn.q_proj)
    k_w, k_b = _dense_weight(attn.k_proj)
    v_w, v_b = _dense_weight(attn.v_proj)
    o_w, o_b = _dense_weight(attn.o_proj)
    if q_w is None or k_w is None or v_w is None or o_w is None:
        return
    if q_w.shape[0] % world != 0 or k_w.shape[0] % world != 0 or v_w.shape[0] % world != 0 or o_w.shape[1] % world != 0:
        return
    q_local = q_w.shape[0] // world
    kv_local = k_w.shape[0] // world
    attn.q_proj = ColumnParallelLinear(
        q_w.shape[1],
        q_w.shape[0],
        world,
        rank,
        weight=q_w[rank * q_local : (rank + 1) * q_local].contiguous(),
        bias=q_b[rank * q_local : (rank + 1) * q_local].contiguous() if q_b is not None else None,
    )
    attn.k_proj = ColumnParallelLinear(
        k_w.shape[1],
        k_w.shape[0],
        world,
        rank,
        weight=k_w[rank * kv_local : (rank + 1) * kv_local].contiguous(),
        bias=k_b[rank * kv_local : (rank + 1) * kv_local].contiguous() if k_b is not None else None,
    )
    attn.v_proj = ColumnParallelLinear(
        v_w.shape[1],
        v_w.shape[0],
        world,
        rank,
        weight=v_w[rank * kv_local : (rank + 1) * kv_local].contiguous(),
        bias=v_b[rank * kv_local : (rank + 1) * kv_local].contiguous() if v_b is not None else None,
    )
    attn.o_proj = RowParallelLinear(
        o_w.shape[1],
        o_w.shape[0],
        world,
        rank,
        weight=slice_row(o_w, world, rank),
        bias=o_b,
    )
    attn.num_heads = num_heads // world
    attn.num_kv_heads = num_kv_heads // world


def _shard_llama_swiglu(mlp, world, rank):
    if world <= 1:
        return
    g_w, g_b = _dense_weight(mlp.gate_proj)
    u_w, u_b = _dense_weight(mlp.up_proj)
    d_w, d_b = _dense_weight(mlp.down_proj)
    if g_w is None or u_w is None or d_w is None:
        return
    if g_w.shape[0] != u_w.shape[0] or g_w.shape[0] % world != 0 or d_w.shape[1] % world != 0:
        return
    combined_w = torch.cat([g_w, u_w], dim=0)
    combined_b = torch.cat([g_b, u_b], dim=0) if (g_b is not None and u_b is not None) else None
    gate_up = ColumnParallelLinear(
        combined_w.shape[1],
        combined_w.shape[0],
        world,
        rank,
        weight=slice_gate_up(combined_w, world, rank),
        bias=_slice_paired_bias(combined_b, combined_w.shape[0], world, rank, 2),
    )
    down = RowParallelLinear(
        d_w.shape[1],
        d_w.shape[0],
        world,
        rank,
        weight=slice_row(d_w, world, rank),
        bias=d_b,
    )
    mlp.gate_up_proj = gate_up
    mlp.down_proj = down
    act = getattr(mlp, "activation", F.silu)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(act(gate) * up)

    mlp.forward = forward.__get__(mlp, mlp.__class__)


def attach_ulysses(model_patcher):
    options = model_patcher.model_options.setdefault("transformer_options", {})

    def override(_func, query, key, value, heads, *args, **kwargs):
        skip_reshape = bool(kwargs.get("skip_reshape", False))
        skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))
        if not skip_reshape:
            b, seq, total = query.shape
            dim_head = total // max(1, int(heads))
            query = query.view(b, seq, int(heads), dim_head).permute(0, 2, 1, 3).contiguous()
            key = key.view(b, key.shape[1], int(heads), dim_head).permute(0, 2, 1, 3).contiguous()
            value = value.view(b, value.shape[1], int(heads), dim_head).permute(0, 2, 1, 3).contiguous()
        percent = _STEP_PERCENT[0]

        def attend(qq, kk, vv):
            return block_sparse_attention(qq, kk, vv, percent)

        attended = ulysses_attention(query, key, value, attend)
        if skip_output_reshape:
            return attended
        b, h, seq, dim_head = attended.shape
        return attended.permute(0, 2, 1, 3).contiguous().reshape(b, seq, h * dim_head)

    options["optimized_attention_override"] = override
    inner = getattr(model_patcher, "model", None)
    if inner is not None and hasattr(inner, "model_options"):
        inner.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = override


def _chunked_attention(q, k, v, chunk=1024):
    batch, heads, queries, dim = q.shape
    keys = k.shape[2]
    out = torch.zeros(batch, heads, queries, dim, device=q.device, dtype=q.dtype)
    running_max = torch.full((batch, heads, queries, 1), -1e30, device=q.device, dtype=q.dtype)
    normalizer = torch.zeros(batch, heads, queries, 1, device=q.device, dtype=q.dtype)
    scale = dim ** -0.5
    for start in range(0, keys, chunk):
        key_chunk = k[:, :, start : start + chunk, :]
        value_chunk = v[:, :, start : start + chunk, :]
        scores = torch.matmul(q, key_chunk.transpose(-1, -2)) * scale
        block_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, block_max)
        alpha = torch.exp(running_max - new_max)
        beta = torch.exp(scores - new_max)
        normalizer = normalizer * alpha + beta.sum(dim=-1, keepdim=True)
        out = out * alpha + torch.matmul(beta.to(value_chunk.dtype), value_chunk)
        running_max = new_max
    return out / normalizer.clamp_min(1e-6)
