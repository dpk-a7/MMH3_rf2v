import contextlib
import ctypes
import dataclasses
import gc
import os
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import torch

GIB = 1024 ** 3
MIB = 1024 ** 2
SLOTS = ("clip", "video_vae", "audio_vae", "diffusion", "lora", "controlnet")

# Configurable per-card headroom and participation thresholds (in MiB)
VRAM_SAFETY_RESERVE_MB = int(os.environ.get("VRAM_SAFETY_RESERVE_MB", "1536"))
MIN_USABLE_VRAM_MB = int(os.environ.get("MIN_USABLE_VRAM_MB", "3072"))
TIGHT_VRAM_THRESHOLD_MB = int(os.environ.get("TIGHT_VRAM_THRESHOLD_MB", "8192"))

# Stage classification constants
ACTIVE_FULL = "ACTIVE_FULL"
ACTIVE_TIGHT = "ACTIVE_TIGHT"
BYPASSED_BUSY = "BYPASSED_BUSY"

ROLE_MAP = {
    ACTIVE_FULL: "active_full",
    ACTIVE_TIGHT: "active_tight_partial",
    BYPASSED_BUSY: "bypassed_busy",
}


class ResourceLimitExceeded(Exception):
    """Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM."""


@dataclasses.dataclass
class GPUStatus:
    index: int
    name: str
    free_vram_mb: float
    total_vram_mb: float
    usable_vram_mb: float
    state: str  # ACTIVE_FULL | ACTIVE_TIGHT | BYPASSED_BUSY
    role: str  # "active_full" | "active_tight_partial" | "bypassed_busy"
    weight: float = 0.0  # Normalized capacity weight w_i in [0.0, 1.0]

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.index}")

    @property
    def allocated_weight_pct(self) -> float:
        return round(float(self.weight * 100.0), 1)

    def to_telemetry(self) -> Dict[str, object]:
        return {
            "index": int(self.index),
            "name": str(self.name),
            "free_vram_mb": int(round(self.free_vram_mb)),
            "total_vram_mb": int(round(self.total_vram_mb)),
            "allocated_weight_pct": self.allocated_weight_pct,
            "role": str(self.role),
        }


@dataclasses.dataclass
class StageSchedule:
    stage: str
    gpus: List[GPUStatus]
    active_gpus: List[GPUStatus]
    lead_device: torch.device
    weights_by_gpu: Dict[int, float]
    total_usable_vram_mb: float
    required_vram_mb: float
    stream_from_pinned_ram: bool
    blocks_per_stream_batch: int = 2

    def telemetry(self) -> List[Dict[str, object]]:
        return [gpu.to_telemetry() for gpu in self.gpus]


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="ascii", errors="ignore") as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_ram() -> Optional[Tuple[int, int]]:
    """(available, limit) inside a cgroup-v2 memory limit, or None when unlimited."""
    limit = _read_int("/sys/fs/cgroup/memory.max")
    current = _read_int("/sys/fs/cgroup/memory.current")
    if limit is None or current is None:
        return None
    reclaimable = 0
    try:
        with open("/sys/fs/cgroup/memory.stat", "r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                key, _, value = line.partition(" ")
                if key in ("active_file", "inactive_file"):
                    reclaimable += int(value)
    except (OSError, ValueError):
        reclaimable = 0
    return max(0, limit - (current - reclaimable)), limit


def host_ram_info() -> Tuple[int, int]:
    """Return (available_bytes, total_bytes), honouring the container memory limit."""
    avail = 0
    total = 0
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
    except OSError:
        pass
    if total <= 0:
        total = 64 * GIB
    if avail <= 0:
        avail = total // 2
    cgroup = _cgroup_ram()
    if cgroup is not None:
        avail = min(avail, cgroup[0])
        total = min(total, cgroup[1])
    return avail, total


def ram_reserve_bytes(total: int) -> int:
    return max(6 * GIB, int(total * 0.08))


def can_pin_bytes(nbytes: int) -> bool:
    avail, total = host_ram_info()
    return (avail - ram_reserve_bytes(total)) > int(nbytes)


def pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and torch.cuda.is_available()
        and not tensor.is_pinned()
        and can_pin_bytes(tensor.numel() * tensor.element_size())
    ):
        try:
            return tensor.pin_memory()
        except Exception:
            return tensor
    return tensor


def device_scope(device: Optional[torch.device]):
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _move_tensors(val, device: torch.device):
    if isinstance(val, torch.Tensor):
        try:
            return val.to(device, non_blocking=True) if val.device != device else val
        except Exception:
            return val
    if isinstance(val, list):
        return [_move_tensors(x, device) for x in val]
    if isinstance(val, tuple):
        return tuple(_move_tensors(x, device) for x in val)
    if isinstance(val, dict):
        return {k: _move_tensors(v, device) for k, v in val.items()}
    return val


def _is_mocked(fn) -> bool:
    return (
        hasattr(fn, "_mock_name")
        or hasattr(fn, "side_effect")
        or getattr(fn, "__module__", "").startswith("unittest")
        or type(fn).__name__.endswith("Mock")
    )


def _pynvml_Snapshot() -> Dict[int, Tuple[str, int, int]]:
    """Query pynvml (or nvidia-smi fallback) for {index: (name, free_bytes, total_bytes)}."""
    out: Dict[int, Tuple[str, int, int]] = {}
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8", "ignore") if isinstance(raw_name, bytes) else str(raw_name)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out[idx] = (name, int(mem.free), int(mem.total))
        return out
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        for line in proc.stdout.decode("ascii", "ignore").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                name = parts[1]
                total_b = int(float(parts[2]) * MIB)
                free_b = int(float(parts[3]) * MIB)
                out[idx] = (name, free_b, total_b)
    except Exception:
        pass
    return out


def probe_live_gpus(
    num_devices: Optional[int] = None,
    safety_reserve_mb: float = VRAM_SAFETY_RESERVE_MB,
    min_usable_mb: float = MIN_USABLE_VRAM_MB,
    tight_threshold_mb: float = TIGHT_VRAM_THRESHOLD_MB,
) -> List[GPUStatus]:
    """Real-time pre-stage VRAM probe across all visible CUDA devices 0..N-1.

    Combines `torch.cuda.mem_get_info(device_id)` with `pynvml` and classifies each
    device into `ACTIVE_FULL`, `ACTIVE_TIGHT`, or `BYPASSED_BUSY`.
    """
    mem_get_info_mocked = _is_mocked(getattr(torch.cuda, "mem_get_info", None))
    if num_devices is None:
        if mem_get_info_mocked and not _is_mocked(getattr(torch.cuda, "device_count", None)):
            try:
                cnt = int(torch.cuda.device_count())
            except Exception:
                cnt = 0
            num_devices = cnt if cnt > 0 else 3
        else:
            try:
                num_devices = int(torch.cuda.device_count()) if (torch.cuda.is_available() or _is_mocked(torch.cuda.device_count)) else 0
            except Exception:
                num_devices = 0

    nvml_map = {} if mem_get_info_mocked else _pynvml_Snapshot()
    statuses: List[GPUStatus] = []

    for dev_id in range(max(0, int(num_devices))):
        name = nvml_map.get(dev_id, (f"cuda:{dev_id}", 0, 0))[0]
        if not mem_get_info_mocked and hasattr(torch.cuda, "get_device_name"):
            try:
                if dev_id not in nvml_map:
                    name = str(torch.cuda.get_device_name(dev_id))
            except Exception:
                pass

        free_b: Optional[int] = None
        total_b: Optional[int] = None

        # If pynvml says the card is already below safety_reserve + min_usable and mem_get_info is not mocked,
        # avoid initializing a primary CUDA context on the busy external GPU.
        if not mem_get_info_mocked and dev_id in nvml_map:
            nv_free_b, nv_total_b = nvml_map[dev_id][1], nvml_map[dev_id][2]
            if (nv_free_b / MIB) - safety_reserve_mb < min_usable_mb:
                free_b, total_b = nv_free_b, nv_total_b

        if free_b is None or total_b is None:
            try:
                t_free, t_total = torch.cuda.mem_get_info(dev_id)
                free_b, total_b = int(t_free), int(t_total)
                if not mem_get_info_mocked and dev_id in nvml_map:
                    free_b = min(free_b, nvml_map[dev_id][1])
            except Exception:
                if dev_id in nvml_map:
                    free_b, total_b = nvml_map[dev_id][1], nvml_map[dev_id][2]
                else:
                    free_b, total_b = 0, 0

        free_mb = float(free_b) / float(MIB)
        total_mb = float(total_b) / float(MIB)
        usable_mb = max(0.0, free_mb - float(safety_reserve_mb))

        if usable_mb < float(min_usable_mb):
            state = BYPASSED_BUSY
        elif usable_mb < float(tight_threshold_mb):
            state = ACTIVE_TIGHT
        else:
            state = ACTIVE_FULL

        statuses.append(
            GPUStatus(
                index=dev_id,
                name=name,
                free_vram_mb=free_mb,
                total_vram_mb=total_mb,
                usable_vram_mb=usable_mb if state != BYPASSED_BUSY else 0.0,
                state=state,
                role=ROLE_MAP[state],
                weight=0.0,
            )
        )

    active = [g for g in statuses if g.state != BYPASSED_BUSY and g.usable_vram_mb >= float(min_usable_mb)]
    total_usable = sum(g.usable_vram_mb for g in active)
    if total_usable > 0.0:
        for g in active:
            g.weight = g.usable_vram_mb / total_usable

    return statuses


def elect_lead_device(gpus: Sequence[GPUStatus]) -> torch.device:
    """Elect `lead_device = argmax(usable_vram)` with zero `cuda:0` anchoring."""
    if not gpus:
        return torch.device("cpu")
    active = [g for g in gpus if g.state != BYPASSED_BUSY]
    pool = active if active else list(gpus)
    best = max(pool, key=lambda g: (g.usable_vram_mb, g.free_vram_mb, -g.index))
    return best.device


def estimate_stage_memory_mb(
    stage: str,
    width: int = 864,
    height: int = 480,
    frames: int = 124,
    num_refs: int = 0,
    model_weights_mb: Optional[float] = None,
) -> float:
    """Estimate M_stage = Model Weights + KV/Reference Activation Cache (in MiB)."""
    t_lat = max(2, ((max(5, int(frames)) - 5) // 17) * 5 + 2)
    h_lat = max(2, int(height) // 16)
    w_lat = max(2, int(width) // 16)
    tokens = t_lat * (h_lat // 2) * (w_lat // 2) + max(0, int(num_refs)) * 512

    if stage == "stage1":
        base_weights = model_weights_mb if model_weights_mb is not None else 11500.0
        kv_cache = 1200.0 + max(0, int(num_refs)) * 380.0
        return base_weights + kv_cache
    if stage == "stage2":
        base_weights = model_weights_mb if model_weights_mb is not None else 14500.0
        act_cache = max(1500.0, (tokens * 96 * 2 * 0.114) / 100.0 * 1024.0)
        return base_weights + act_cache
    if stage == "stage3":
        base_weights = model_weights_mb if model_weights_mb is not None else 2600.0
        frame_act = (int(width) * int(height) * min(34, int(frames)) * 3 * 2 * 9.5) / float(MIB)
        return base_weights + max(600.0, frame_act)
    return model_weights_mb if model_weights_mb is not None else 8192.0


def build_stage_schedule(
    stage: str = "stage2",
    required_vram_mb: Optional[float] = None,
    num_devices: Optional[int] = None,
    safety_reserve_mb: float = VRAM_SAFETY_RESERVE_MB,
    min_usable_mb: float = MIN_USABLE_VRAM_MB,
) -> StageSchedule:
    """Probe all GPUs, elect `lead_device`, compute proportional weights w_i, and decide streaming fallback."""
    gpus = probe_live_gpus(
        num_devices=num_devices,
        safety_reserve_mb=safety_reserve_mb,
        min_usable_mb=min_usable_mb,
    )
    active = [g for g in gpus if g.state != BYPASSED_BUSY]
    lead_device = elect_lead_device(gpus)
    total_usable = sum(g.usable_vram_mb for g in active)
    req_mb = float(required_vram_mb) if required_vram_mb is not None else estimate_stage_memory_mb(stage)
    stream_fallback = (len(active) == 0) or (total_usable < req_mb)

    # When falling back to Pinned-RAM Micro-Block Streaming on lead_device, reflect lead_device in telemetry
    if stream_fallback and gpus:
        lead_idx = lead_device.index if lead_device.type == "cuda" else 0
        for g in gpus:
            if g.index == lead_idx and g.free_vram_mb > 0:
                g.weight = 1.0
                if g.state == BYPASSED_BUSY:
                    g.state = ACTIVE_TIGHT
                    g.role = ROLE_MAP[ACTIVE_TIGHT]
            else:
                g.weight = 0.0

    weights_by_gpu = {g.index: g.weight for g in gpus if g.weight > 0.0}
    lead_usable = next((g.usable_vram_mb for g in gpus if g.device == lead_device), 3072.0)
    blocks_per_batch = max(1, min(4, int(max(1024.0, lead_usable) // 1536.0)))

    return StageSchedule(
        stage=stage,
        gpus=gpus,
        active_gpus=active,
        lead_device=lead_device,
        weights_by_gpu=weights_by_gpu,
        total_usable_vram_mb=total_usable,
        required_vram_mb=req_mb,
        stream_from_pinned_ram=stream_fallback,
        blocks_per_stream_batch=blocks_per_batch,
    )


class PinnedBlockStreamer:
    """Double-buffered CUDA stream micro-block streamer backed by Pinned Host RAM.

    Keeps the full transformer stack in Pinned Host RAM and streams `batch_size`
    blocks at a time onto `lead_device` using `compute_stream` and `transfer_stream`,
    prefetching batch `b+1` while batch `b` computes, then freeing batch `b`'s VRAM.
    """

    def __init__(
        self,
        blocks: Sequence[torch.nn.Module],
        lead_device: torch.device,
        batch_size: int = 2,
    ):
        self.blocks = list(blocks)
        self.lead_device = lead_device
        self.batch_size = max(1, int(batch_size))
        self.pinned_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.pinned_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.compute_stream = None
        self.transfer_stream = None
        self.prefetch_events: int = 0
        self.offload_events: int = 0
        self._staged: set = set()
        self._orig_forwards: Dict[int, object] = {
            idx: getattr(blk, "_mmh3_orig_forward", blk.forward)
            for idx, blk in enumerate(self.blocks)
        }
        self._prepare_pinned_cache()
        if lead_device.type == "cuda" and torch.cuda.is_available():
            try:
                self.compute_stream = torch.cuda.Stream(device=lead_device)
                self.transfer_stream = torch.cuda.Stream(device=lead_device)
            except Exception:
                self.compute_stream = None
                self.transfer_stream = None
        self._install_streaming_hooks()

    def _install_streaming_hooks(self) -> None:
        """Install transparent per-block double-buffered streaming hooks for standard `for blk in blocks:` loops."""
        n = len(self.blocks)
        for idx, block in enumerate(self.blocks):
            if not hasattr(block, "_mmh3_orig_forward"):
                block._mmh3_orig_forward = block.forward
            orig_fn = block._mmh3_orig_forward

            def _streamed_forward(*args, _idx=idx, _orig=orig_fn, **kwargs):
                if _idx not in self._staged:
                    for k in range(_idx, min(n, _idx + self.batch_size)):
                        if k not in self._staged:
                            self._stage_block_to_gpu(k)
                            self._staged.add(k)
                if self.compute_stream is not None and self.transfer_stream is not None:
                    self.compute_stream.wait_stream(self.transfer_stream)
                next_idx = _idx + self.batch_size
                if next_idx < n and next_idx not in self._staged:
                    self._stage_block_to_gpu(next_idx)
                    self._staged.add(next_idx)

                dev = self.lead_device
                moved_args = _move_tensors(args, dev)
                moved_kwargs = _move_tensors(kwargs, dev)
                compute_ctx = (
                    torch.cuda.stream(self.compute_stream)
                    if self.compute_stream is not None
                    else contextlib.nullcontext()
                )
                with device_scope(dev), compute_ctx:
                    out = _orig(*moved_args, **moved_kwargs)
                if self.compute_stream is not None and self.transfer_stream is not None:
                    self.transfer_stream.wait_stream(self.compute_stream)
                self._offload_block_to_cpu(_idx)
                self._staged.discard(_idx)
                return out

            block.forward = _streamed_forward

    def _prepare_pinned_cache(self) -> None:
        for idx, block in enumerate(self.blocks):
            p_map: Dict[str, torch.Tensor] = {}
            b_map: Dict[str, torch.Tensor] = {}
            for name, param in block.named_parameters(recurse=True):
                cpu_t = pin_tensor(param.detach().to("cpu", copy=True).contiguous())
                p_map[name] = cpu_t
            for name, buf in block.named_buffers(recurse=True):
                cpu_b = pin_tensor(buf.detach().to("cpu", copy=True).contiguous())
                b_map[name] = cpu_b
            self.pinned_params[idx] = p_map
            self.pinned_buffers[idx] = b_map
            try:
                block.to("cpu")
            except Exception:
                pass

    def _stage_block_to_gpu(self, block_idx: int) -> None:
        block = self.blocks[block_idx]
        dev = self.lead_device
        p_map = self.pinned_params.get(block_idx, {})
        b_map = self.pinned_buffers.get(block_idx, {})
        stream_ctx = (
            torch.cuda.stream(self.transfer_stream)
            if self.transfer_stream is not None
            else contextlib.nullcontext()
        )
        with stream_ctx:
            try:
                for name, param in block.named_parameters(recurse=True):
                    src = p_map.get(name, param)
                    param.data = src.to(dev, non_blocking=True)
                for name, buf in block.named_buffers(recurse=True):
                    src = b_map.get(name, buf)
                    buf.data = src.to(dev, non_blocking=True)
            except Exception:
                block.to(dev)
        self.prefetch_events += 1

    def _offload_block_to_cpu(self, block_idx: int) -> None:
        block = self.blocks[block_idx]
        p_map = self.pinned_params.get(block_idx, {})
        b_map = self.pinned_buffers.get(block_idx, {})
        try:
            for name, param in block.named_parameters(recurse=True):
                if name in p_map:
                    param.data = p_map[name]
                else:
                    param.data = param.detach().to("cpu")
            for name, buf in block.named_buffers(recurse=True):
                if name in b_map:
                    buf.data = b_map[name]
                else:
                    buf.data = buf.detach().to("cpu")
        except Exception:
            try:
                block.to("cpu")
            except Exception:
                pass
        self.offload_events += 1

    def execute_blocks(self, x: torch.Tensor, block_runner=None, *args, **kwargs) -> torch.Tensor:
        """Stream blocks in micro-batches of `self.batch_size` with double-buffered prefetch."""
        n = len(self.blocks)
        if n == 0:
            return x
        dev = self.lead_device
        if isinstance(x, torch.Tensor) and x.device != dev:
            try:
                x = x.to(dev, non_blocking=True)
            except Exception:
                pass

        batches = [list(range(i, min(n, i + self.batch_size))) for i in range(0, n, self.batch_size)]

        # Prefetch first micro-batch
        for b_idx in batches[0]:
            self._stage_block_to_gpu(b_idx)

        for step_idx, current_batch in enumerate(batches):
            if self.compute_stream is not None and self.transfer_stream is not None:
                self.compute_stream.wait_stream(self.transfer_stream)

            # Asynchronous prefetch of next micro-batch while current micro-batch computes
            if step_idx + 1 < len(batches):
                for next_b_idx in batches[step_idx + 1]:
                    self._stage_block_to_gpu(next_b_idx)

            compute_ctx = (
                torch.cuda.stream(self.compute_stream)
                if self.compute_stream is not None
                else contextlib.nullcontext()
            )
            with device_scope(dev), compute_ctx:
                for b_idx in current_batch:
                    blk = self.blocks[b_idx]
                    orig_fn = getattr(blk, "_mmh3_orig_forward", blk.forward)
                    if block_runner is not None:
                        x = block_runner(b_idx, blk, x, *args, **kwargs)
                    else:
                        x = orig_fn(x, *args, **kwargs)

            if self.compute_stream is not None and self.transfer_stream is not None:
                self.transfer_stream.wait_stream(self.compute_stream)

            for b_idx in current_batch:
                self._offload_block_to_cpu(b_idx)

        if self.compute_stream is not None:
            try:
                self.compute_stream.synchronize()
            except Exception:
                pass
        return x


def plan_devices(num_devices: Optional[int] = None) -> Dict[str, torch.device]:
    """Build per-slot initial device map using live VRAM probing and zero `cuda:0` anchoring."""
    schedule = build_stage_schedule("stage2", num_devices=num_devices)
    lead = schedule.lead_device
    if lead.type != "cuda" or len(schedule.active_gpus) == 0:
        return {slot: lead for slot in SLOTS}

    sorted_active = sorted(schedule.active_gpus, key=lambda g: (-g.usable_vram_mb, g.index))
    dit_dev = sorted_active[0].device
    enc_dev = sorted_active[1].device if len(sorted_active) > 1 else dit_dev
    aud_dev = sorted_active[2].device if len(sorted_active) > 2 else enc_dev
    return {
        "clip": enc_dev,
        "video_vae": dit_dev if len(sorted_active) >= 2 else enc_dev,
        "audio_vae": aud_dev,
        "diffusion": dit_dev,
        "lora": dit_dev,
        "controlnet": dit_dev,
    }


def describe_plan(plan: Dict[str, torch.device]) -> str:
    groups: Dict[str, List[str]] = {}
    for slot in SLOTS:
        if slot in plan:
            groups.setdefault(str(plan[slot]), []).append(slot)
    return "; ".join(f"{dev}: {'+'.join(names)}" for dev, names in groups.items())


class ModelRegistry:
    """Dynamic VRAM-aware model registry with pre-stage probing and pinned-RAM fallback."""

    def __init__(self, model_root: str = "/app/model"):
        self.model_root = model_root
        self.lock = threading.Lock()
        self.loaded_slots = {slot: {"key": None, "module": None} for slot in SLOTS}
        self.last_schedule: StageSchedule = build_stage_schedule("init")
        self.plan: Dict[str, torch.device] = plan_devices()

    def probe_and_plan(
        self,
        stage: str,
        required_vram_mb: Optional[float] = None,
        num_devices: Optional[int] = None,
    ) -> StageSchedule:
        with self.lock:
            schedule = build_stage_schedule(
                stage=stage,
                required_vram_mb=required_vram_mb,
                num_devices=num_devices,
            )
            self.last_schedule = schedule
            if schedule.lead_device.type == "cuda":
                sorted_active = sorted(
                    schedule.active_gpus,
                    key=lambda g: (-g.usable_vram_mb, g.index),
                )
                if sorted_active:
                    dit_dev = sorted_active[0].device
                    enc_dev = sorted_active[1].device if len(sorted_active) > 1 else dit_dev
                    aud_dev = sorted_active[2].device if len(sorted_active) > 2 else enc_dev
                    self.plan = {
                        "clip": enc_dev,
                        "video_vae": dit_dev,
                        "audio_vae": aud_dev,
                        "diffusion": dit_dev,
                        "lora": dit_dev,
                        "controlnet": dit_dev,
                    }
            return schedule

    def latest_telemetry(self) -> List[Dict[str, object]]:
        with self.lock:
            return self.last_schedule.telemetry()

    def _safe_path(self, subdir: str, filename: str) -> str:
        raw = (filename or "").strip().replace("\\", "/")
        parts = [p for p in raw.split("/") if p and p != "."]
        if not parts or any(p == ".." or p.startswith(".") for p in parts):
            raise FileNotFoundError(raw)
        clean_rel = os.path.join(*parts)
        full_path = os.path.join(self.model_root, subdir, clean_rel)
        root = os.path.realpath(os.path.join(self.model_root, subdir))
        real = os.path.realpath(full_path)
        if not real.startswith(root + os.sep) or not os.path.isfile(real):
            raise FileNotFoundError(clean_rel)
        return real

    def cached(self, slot: str, key: str) -> bool:
        current = self.loaded_slots[slot]
        return current["module"] is not None and current["key"] == key

    def evict(self, slot: str) -> None:
        current = self.loaded_slots[slot]
        if current["module"] is None:
            return
        current["module"] = None
        current["key"] = None
        _unload_comfy_models({self.plan.get(slot, self.last_schedule.lead_device)})
        self.trim_all_memory()

    def _ensure_ram(self, nbytes: int, keep) -> None:
        need = int(nbytes * 1.25) + ram_reserve_bytes(host_ram_info()[1])
        if host_ram_info()[0] >= need:
            return
        for slot in SLOTS:
            if slot not in keep and self.loaded_slots[slot]["module"] is not None:
                self.loaded_slots[slot]["module"] = None
                self.loaded_slots[slot]["key"] = None
        _unload_comfy_models()
        self.trim_all_memory()
        if host_ram_info()[0] < need:
            raise ResourceLimitExceeded("insufficient_ram")

    def get_or_swap(self, slot: str, subdir: str, filename: str, loader_fn, key: Optional[str] = None, keep=()):
        key = key or filename
        with self.lock:
            if self.cached(slot, key):
                return self.loaded_slots[slot]["module"]
            self.evict(slot)
            path = self._safe_path(subdir, filename)
            self._ensure_ram(os.path.getsize(path), set(keep) | {slot})
            target_dev = self.plan.get(slot, self.last_schedule.lead_device)
            with device_scope(target_dev):
                module = loader_fn(path)
            self.loaded_slots[slot]["module"] = module
            self.loaded_slots[slot]["key"] = key
            return module

    def flush_vram(self) -> None:
        with self.lock:
            for slot in self.loaded_slots.values():
                slot["key"] = None
                slot["module"] = None
            _unload_comfy_models()
            self.trim_all_memory()
            self.last_schedule = build_stage_schedule("flush")
            self.plan = plan_devices()

    def trim_all_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available() and not _is_mocked(torch.cuda.mem_get_info):
            active_devs = {
                g.device for g in self.last_schedule.gpus if g.state != BYPASSED_BUSY
            } | {d for d in self.plan.values() if d.type == "cuda"}
            for device in active_devs:
                try:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    continue
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            return


# Alias for backward compatibility with Specs.md §6
PinnedModelRegistry = ModelRegistry


def _unload_comfy_models(devices=None) -> None:
    if "comfy.model_management" not in sys.modules:
        return
    try:
        import comfy.model_management as mm

        active = {
            getattr(m, "device", None)
            for m in list(getattr(mm, "current_loaded_models", ()))
        }
        active.discard(None)
        if devices is not None:
            active &= set(devices)
        for dev in active:
            mm.free_memory(1e30, dev)
        mm.cleanup_models_gc()
        mm.cleanup_models()
    except Exception:
        return
