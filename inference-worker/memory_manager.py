import ctypes
import gc
import os
import threading

import torch

from parallel_shard import CTX, shard_first_dim


class PinnedModule(torch.nn.Module):
    """Full checkpoint resident in pinned host RAM, moved stage by stage."""

    def __init__(self, tensors):
        super().__init__()
        self.weights = {}
        for key, value in tensors.items():
            cpu = value.detach().to("cpu").contiguous()
            self.weights[str(key)] = _pin(cpu)
        self._fingerprint = None

    def fingerprint(self):
        if self._fingerprint is not None:
            return self._fingerprint
        acc = torch.zeros(256, dtype=torch.float64)
        for key, tensor in self.weights.items():
            flat = tensor.detach().reshape(-1)
            if flat.numel() == 0:
                continue
            step = max(1, flat.numel() // 65536)
            sample = flat[::step][:65536].to(torch.float64)
            slot = _stable_slot(key)
            acc[slot] += sample.mean()
            if sample.numel() > 1:
                acc[(slot + 1) % 256] += sample.std(unbiased=False)
        self._fingerprint = acc.to(torch.float32)
        return self._fingerprint

    def move_to(self, device):
        target = torch.device(device)
        moved = {}
        for key, tensor in self.weights.items():
            if target.type == "cpu":
                cpu = tensor.detach().to("cpu", non_blocking=True).contiguous()
                moved[key] = _pin(cpu)
            else:
                moved[key] = tensor.detach().to(target, non_blocking=True).contiguous()
        self.weights = moved
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class PinnedModelRegistry:
    def __init__(self, model_root="/app/model"):
        self.model_root = model_root
        self.lock = threading.Lock()
        self.loaded_slots = {
            "diffusion": {"filename": None, "module": None},
            "video_vae": {"filename": None, "module": None},
            "audio_vae": {"filename": None, "module": None},
            "clip": {"filename": None, "module": None},
            "lora": {"filename": None, "module": None, "weights": None},
            "controlnet": {"filename": None, "module": None},
        }
        self.mempool = None
        pool_cls = getattr(torch.cuda, "MemPool", None)
        if pool_cls is not None and torch.cuda.is_available():
            self.mempool = pool_cls()

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

    def get_or_swap(self, slot, subdir, filename, loader_fn):
        with self.lock:
            current = self.loaded_slots[slot]
            if current["filename"] != filename or current.get("module") is None:
                if current.get("module") is not None:
                    current["module"] = None
                    current["weights"] = None
                    self.trim_all_memory()
                path = self._safe_path(subdir, filename)
                module = loader_fn(path)
                current["module"] = module
                current["filename"] = filename
                if slot == "lora":
                    current["weights"] = module
            return current["module"]

    def activate(self, slots, device):
        with self.lock:
            for slot in slots:
                module = self.loaded_slots[slot].get("module")
                if module is not None:
                    module.move_to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def offload(self, slots):
        with self.lock:
            for slot in slots:
                module = self.loaded_slots[slot].get("module")
                if module is not None:
                    module.move_to("cpu")
        self.trim_all_memory()

    def flush_vram(self):
        with self.lock:
            for slot in self.loaded_slots.values():
                slot["filename"] = None
                slot["module"] = None
                slot["weights"] = None
        self.trim_all_memory()

    @staticmethod
    def trim_all_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            return


def load_pinned(path):
    tensors = read_tensors(path)
    if CTX.world > 1:
        sharded = {}
        for key, value in tensors.items():
            sharded[key] = shard_first_dim(value, CTX.rank, CTX.world)
        tensors = sharded
    return PinnedModule(tensors)


def read_tensors(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors import safe_open

        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key).contiguous()
        if not tensors:
            raise FileNotFoundError(os.path.basename(path))
        return tensors
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    except Exception:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
        blob = blob["state_dict"]
    if not isinstance(blob, dict):
        raise FileNotFoundError(os.path.basename(path))
    tensors = {}
    for key, value in blob.items():
        if torch.is_tensor(value):
            tensors[str(key)] = value.detach().cpu().contiguous()
    if not tensors:
        raise FileNotFoundError(os.path.basename(path))
    return tensors


def _pin(tensor):
    if torch.cuda.is_available() and tensor.device.type == "cpu" and not tensor.is_pinned():
        if can_pin_bytes(tensor.numel() * tensor.element_size()):
            try:
                return tensor.pin_memory()
            except RuntimeError:
                return tensor
    return tensor


def host_ram_info():
    """Return (available_bytes, total_bytes) from /proc/meminfo with safe fallback."""
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
        total = 64 * (1024 ** 3)
    if avail <= 0:
        avail = total // 2
    return avail, total


def can_pin_bytes(nbytes):
    """ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned."""
    avail, total = host_ram_info()
    reserve = max(8 * (1024 ** 3), int(total * 0.12))
    return (avail - reserve) > int(nbytes)


def vram_budget_bytes(device=None):
    """ComfyUI-style usable VRAM budget for model weights on `device`."""
    if not torch.cuda.is_available():
        return 0
    try:
        dev = device if device is not None else torch.cuda.current_device()
        if isinstance(dev, torch.device) and dev.type != "cuda":
            return 0
        free_b, total_b = torch.cuda.mem_get_info(dev)
    except Exception:
        return 0
    reserve_mb = os.environ.get("MMH3_RESERVE_VRAM_MB")
    if reserve_mb is not None:
        try:
            reserve = max(512 * (1024 ** 2), int(float(reserve_mb) * (1024 ** 2)))
        except ValueError:
            reserve = int(max(1200 * (1024 ** 2), min(3072 * (1024 ** 2), total_b * 0.15)))
    else:
        reserve = int(max(1200 * (1024 ** 2), min(3072 * (1024 ** 2), total_b * 0.15)))
    return max(64 * (1024 ** 2), int(free_b - reserve))


def _stable_slot(key):
    total = 2166136261
    for byte in key.encode("utf-8", "ignore"):
        total ^= byte
        total = (total * 16777619) & 0xFFFFFFFF
    return total % 256
