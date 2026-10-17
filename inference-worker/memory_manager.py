import contextlib
import ctypes
import gc
import os
import subprocess
import threading

import torch

GIB = 1024 ** 3
SLOTS = ("clip", "video_vae", "audio_vae", "diffusion", "lora", "controlnet")


class ResourceLimitExceeded(Exception):
    """Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM."""


def _read_int(path):
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


def _cgroup_ram():
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


def host_ram_info():
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


def ram_reserve_bytes(total):
    return max(6 * GIB, int(total * 0.08))


def device_scope(device):
    if device is not None and device.type == "cuda":
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _nvidia_smi_free():
    """Physical index -> (free_bytes, total_bytes) without creating CUDA contexts."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:
        return {}
    rows = {}
    for line in proc.stdout.decode("ascii", "ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                rows[int(parts[0])] = (int(float(parts[2])) * 1024 ** 2, int(float(parts[1])) * 1024 ** 2)
            except ValueError:
                continue
    return rows


def _visible_gpu_memory():
    """[(torch_index, free_bytes, total_bytes)] for every visible CUDA device."""
    count = torch.cuda.device_count()
    phys = _nvidia_smi_free()
    visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    mapping = list(range(count))
    if visible:
        try:
            mapping = [int(v) for v in visible.split(",") if v.strip()][:count]
        except ValueError:
            phys = {}
    rows = []
    for index in range(count):
        if phys and index < len(mapping) and mapping[index] in phys:
            free_b, total_b = phys[mapping[index]]
        else:
            free_b, total_b = torch.cuda.mem_get_info(index)
        rows.append((index, free_b, total_b))
    return rows


def plan_devices():
    """One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next ones.

    GPUs already occupied by external workloads (free < max(MMH3_MIN_FREE_VRAM_GB, 30%))
    are skipped unless every GPU is busy, in which case the emptiest one is used.
    """
    cpu = torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return {slot: cpu for slot in SLOTS}
    try:
        min_free = float(os.environ.get("MMH3_MIN_FREE_VRAM_GB", "6")) * GIB
    except ValueError:
        min_free = 6 * GIB
    rows = _visible_gpu_memory()
    usable = [r for r in rows if r[1] >= max(min_free, 0.30 * r[2])]
    if not usable:
        usable = [max(rows, key=lambda r: r[1])]
    usable.sort(key=lambda r: -r[1])
    dit = torch.device(f"cuda:{usable[0][0]}")
    enc = torch.device(f"cuda:{usable[1][0]}") if len(usable) > 1 else dit
    aud = torch.device(f"cuda:{usable[2][0]}") if len(usable) > 2 else enc
    return {
        "clip": enc,
        "video_vae": enc,
        "audio_vae": aud,
        "diffusion": dit,
        "lora": dit,
        "controlnet": dit,
    }


def describe_plan(plan):
    groups = {}
    for slot in SLOTS:
        groups.setdefault(str(plan[slot]), []).append(slot)
    return "; ".join(f"{dev}: {'+'.join(names)}" for dev, names in groups.items())


class ModelRegistry:
    """Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency."""

    def __init__(self, model_root="/app/model"):
        self.model_root = model_root
        self.lock = threading.Lock()
        self.loaded_slots = {slot: {"key": None, "module": None} for slot in SLOTS}
        self.plan = plan_devices()

    def _safe_path(self, subdir, filename):
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

    def cached(self, slot, key):
        current = self.loaded_slots[slot]
        return current["module"] is not None and current["key"] == key

    def evict(self, slot):
        current = self.loaded_slots[slot]
        if current["module"] is None:
            return
        current["module"] = None
        current["key"] = None
        _unload_comfy_models()
        self.trim_all_memory()

    def _ensure_ram(self, nbytes, keep):
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

    def get_or_swap(self, slot, subdir, filename, loader_fn, key=None, keep=()):
        key = key or filename
        with self.lock:
            if self.cached(slot, key):
                return self.loaded_slots[slot]["module"]
            self.evict(slot)
            path = self._safe_path(subdir, filename)
            self._ensure_ram(os.path.getsize(path), set(keep) | {slot})
            with device_scope(self.plan[slot]):
                module = loader_fn(path)
            self.loaded_slots[slot]["module"] = module
            self.loaded_slots[slot]["key"] = key
            return module

    def flush_vram(self):
        with self.lock:
            for slot in self.loaded_slots.values():
                slot["key"] = None
                slot["module"] = None
            _unload_comfy_models()
            self.trim_all_memory()
            self.plan = plan_devices()

    def trim_all_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            for device in {d for d in self.plan.values() if d.type == "cuda"}:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            return


def _unload_comfy_models():
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
    except Exception:
        return
