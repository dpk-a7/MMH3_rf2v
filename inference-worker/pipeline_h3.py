import contextlib
import math
import os
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F

from memory_manager import (
    GIB,
    TIGHT_VRAM_THRESHOLD_MB,
    ResourceLimitExceeded,
    describe_plan,
    device_scope,
    estimate_stage_memory_mb,
    host_ram_info,
    ram_reserve_bytes,
)
from parallel_shard import apply_proportional_block_sharding

_COMFY_READY = False
CPU = torch.device("cpu")


class JobAborted(Exception):
    """The caller aborted the running generation."""


class BadReference(Exception):
    """A reference image, video, or audio clip failed validation."""


class MuxFailed(Exception):
    """Bitexact FFmpeg muxing did not produce an output file."""


def _ephemeral_free_bytes():
    try:
        stat = os.statvfs(_ephemeral_root())
        return int(stat.f_bavail * stat.f_frsize)
    except OSError:
        return 0


def check_host_gate(width, height, frames):
    """Reject before allocating when decoded frames would not fit in RAM or /app/ephemeral."""
    avail, total = host_ram_info()
    frame_bytes = int(width) * int(height) * int(frames) * 3 * 4
    if avail - ram_reserve_bytes(total) < frame_bytes * 2 + 2 * GIB:
        raise ResourceLimitExceeded("insufficient_ram")
    if _ephemeral_free_bytes() < max(512 * 1024 ** 2, frame_bytes // 16):
        raise ResourceLimitExceeded("insufficient_storage")


def check_vram_gate(model, positive, latent, device):
    """Activations cannot be streamed like weights: they must fit on the DiT GPU.

    Uses ComfyUI's own batch-1 estimate on the flat-packed AV latent plus reference conditioning.
    """
    if device.type != "cuda":
        return
    import comfy.model_management as mm
    import comfy.sampler_helpers as helpers

    samples = latent.get("samples")
    if samples is None:
        return
    if hasattr(samples, "unbind"):
        packed_shape = [1, 1, sum(int(t.numel()) for t in samples.unbind())]
    elif hasattr(samples, "shape"):
        packed_shape = list(samples.shape)
    else:
        return
    try:
        conds = {"positive": helpers.convert_cond(positive)}
        _full, minimum = helpers.estimate_memory(model, packed_shape, conds)
        need = int(minimum + mm.minimum_inference_memory())
    except Exception:
        return
    free_b, _total_b = torch.cuda.mem_get_info(device)
    if need > free_b + torch.cuda.memory_reserved(device):
        raise ResourceLimitExceeded("insufficient_vram")


def _force_dtype(vae, dtype):
    import comfy.model_management as mm

    vae.vae_dtype = dtype
    vae.first_stage_model.to(dtype=dtype)
    mm.archive_model_dtypes(vae.first_stage_model)
    return vae


def _patch_psutil_cgroup():
    try:
        import psutil

        if getattr(psutil, "_mmh3_patched", False):
            return
        orig = psutil.virtual_memory

        def _virtual_memory():
            vm = orig()
            avail, total = host_ram_info()
            if total < vm.total or avail < vm.available:
                return vm._replace(total=min(vm.total, total), available=min(vm.available, avail))
            return vm

        psutil.virtual_memory = _virtual_memory
        psutil._mmh3_patched = True
    except Exception:
        return


def _patch_minimax_model():
    import importlib

    try:
        mb = importlib.import_module("comfy.model_base")
        cls = getattr(mb, "MiniMaxH3", None)
        if cls is not None and not getattr(cls, "_mmh3_patched", False):
            cls.memory_usage_factor_conds = ("minimax_cond",)

            def _extra_conds_shapes(self, **kwargs):
                elems = 0
                cross = kwargs.get("cross_attn")
                if hasattr(cross, "shape") and cross.ndim >= 2:
                    elems += int(cross.shape[1]) * 96
                for r in kwargs.get("minimax_refs") or ():
                    lat = r.get("latent")
                    if hasattr(lat, "numel"):
                        elems += int(lat.numel())
                    elems += int(r.get("ref_audio_t") or 0) * 2 * 96
                for kf in kwargs.get("minimax_keyframes") or ():
                    lat = kf.get("latent")
                    if hasattr(lat, "numel"):
                        elems += int(lat.numel())
                return {"minimax_cond": [1, 1, elems]} if elems > 0 else {}

            cls.extra_conds_shapes = _extra_conds_shapes
            cls._mmh3_patched = True
    except Exception:
        pass
    try:
        mm_dit = importlib.import_module("comfy.ldm.minimax.model")
        dit_cls = getattr(mm_dit, "MiniMaxH3Model", None)
        if dit_cls is not None and not getattr(dit_cls, "_mmh3_cn_patched", False):
            orig_forward = dit_cls._forward

            def _forward_with_control(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
                control = kwargs.get("control")
                if isinstance(control, dict) and (control.get("output") or control.get("input")):
                    res_list = (control.get("output") or []) + (control.get("input") or [])
                    transformer_options = transformer_options.copy()
                    pr = transformer_options.get("patches_replace", {}).copy()
                    dit_pr = pr.get("dit", {}).copy()
                    n_blocks = len(self.blocks)
                    stride = max(1, n_blocks // max(1, len(res_list)))
                    for idx, res in enumerate(res_list):
                        blk_idx = min(n_blocks - 1, idx * stride)
                        if res is not None and ("double_block", blk_idx) not in dit_pr:
                            def _inject(args, wrapper_extra, _r=res):
                                out = wrapper_extra["original_block"](args)
                                img = out["img"]
                                delta = _r.reshape(-1, _r.shape[-1]) if _r.ndim > 2 else _r
                                if delta.shape[-1] == img.shape[-1] and delta.shape[0] <= img.shape[0]:
                                    img[-delta.shape[0]:].add_(delta.to(device=img.device, dtype=img.dtype))
                                return out
                            dit_pr[("double_block", blk_idx)] = _inject
                    pr["dit"] = dit_pr
                    transformer_options["patches_replace"] = pr
                return orig_forward(self, x, timestep, context, transformer_options=transformer_options, minimax_payload=minimax_payload, **kwargs)

            dit_cls._forward = _forward_with_control
            dit_cls._mmh3_cn_patched = True
    except Exception:
        return


def boot_comfy(model_root):
    global _COMFY_READY
    if _COMFY_READY:
        return
    root = os.environ.get("COMFYUI_ROOT", "/opt/comfyui")
    if root not in sys.path:
        sys.path.insert(0, root)
    import logging

    logging.disable(logging.CRITICAL)
    _patch_psutil_cgroup()
    import comfy.model_management as mm
    import comfy.utils
    import folder_paths

    _patch_minimax_model()
    # ComfyUI sizes its pin budget from host RAM; pinned pages cannot be reclaimed, so cap it to the container.
    if mm.MAX_PINNED_MEMORY > 0:
        mm.MAX_PINNED_MEMORY = min(mm.MAX_PINNED_MEMORY, host_ram_info()[1] * 0.6)
    comfy.utils.PROGRESS_BAR_ENABLED = False
    for sub in ("output", "temp", "input", "user"):
        os.makedirs(f"/tmp/comfyui/{sub}", exist_ok=True)
    folder_paths.set_output_directory("/tmp/comfyui/output")
    folder_paths.temp_directory = "/tmp/comfyui/temp"
    folder_paths.input_directory = "/tmp/comfyui/input"
    folder_paths.user_directory = "/tmp/comfyui/user"
    mapping = {
        "diffusion_models": "diffussion",
        "unet": "diffussion",
        "text_encoders": "clip",
        "clip": "clip",
        "vae": "vae",
        "loras": "lora",
        "controlnet": "controlnet",
        "embeddings": "clip",
    }
    for name, subdir in mapping.items():
        path = os.path.join(model_root, subdir)
        os.makedirs(path, exist_ok=True)
        try:
            folder_paths.add_model_folder_path(name, path)
        except Exception:
            current = folder_paths.folder_names_and_paths.get(name)
            exts = current[1] if current else {".safetensors"}
            folder_paths.folder_names_and_paths[name] = ([path], exts)
    _COMFY_READY = True


def _rel_model_name(path, subdir):
    marker = f"{os.sep}{subdir}{os.sep}"
    if marker in path:
        return path.split(marker, 1)[1]
    return os.path.basename(path)


def clip_activation_bytes(tokens, floor_bytes=9 * GIB):
    """Activation budget ComfyUI must keep free before Qwen3-VL encode.

    CLIPLoader otherwise fully pins ~16GB weights on a 20GB card, which leaves
    too little room for a reference-image vision pass.
    """
    seq = 0
    vis_blocks = 0
    items = tokens.get("qwen3vl_32b", ()) if isinstance(tokens, dict) else tokens
    for batch in items or ():
        for tok in batch:
            entry = tok[0] if isinstance(tok, tuple) else tok
            if isinstance(entry, dict) and hasattr(entry.get("data"), "shape"):
                shape = entry["data"].shape
                h, w = int(shape[-3]), int(shape[-2])
                vis_blocks += 1
                seq += max(64, (h // 32) * (w // 32))
            else:
                seq += 1
    need = int(3.2 * GIB + seq * 262_144 + vis_blocks * 32 * 1024 ** 2)
    return max(need, int(floor_bytes))


def _load_clip(path):
    from nodes import CLIPLoader

    clip = CLIPLoader().load_clip(_rel_model_name(path, "clip"), type="minimax")[0]

    def _estimate_clip_memory(tokens, device=None):
        return clip_activation_bytes(tokens, getattr(clip, "_mmh3_activation_floor", 9 * GIB))

    if hasattr(clip, "cond_stage_model"):
        clip.cond_stage_model.memory_estimation_function = _estimate_clip_memory
    return clip


def _load_vae(path, dtype):
    from nodes import VAELoader

    return _force_dtype(VAELoader().load_vae(_rel_model_name(path, "vae"))[0], dtype)


# ComfyUI's H3 factor (0.114) asked for 22.3 GiB of activations at 1280x736x362 (~99k tokens) and the job was
# refused on a 20 GB card, yet Stage 2 peaked at 10.8 GiB there (RTX 4000 Ada, FastH3 + VSA). 0.075 keeps ~30% headroom.
H3_MEMORY_USAGE_FACTOR = 0.075


def _load_unet(path):
    from nodes import UNETLoader

    model = UNETLoader().load_unet(_rel_model_name(path, "diffussion"), "default")[0]
    model.model.memory_usage_factor = H3_MEMORY_USAGE_FACTOR
    return model


def _load_lora(path, base, strength):
    from nodes import LoraLoaderModelOnly

    return LoraLoaderModelOnly().load_lora_model_only(base, _rel_model_name(path, "lora"), strength)[0]


def _load_controlnet(path):
    from nodes import ControlNetLoader

    return ControlNetLoader().load_controlnet(_rel_model_name(path, "controlnet"))[0]


def _unload_device(device):
    import comfy.model_management as mm

    mm.free_memory(1e30, device)
    mm.soft_empty_cache()


def _unwrap_nn(obj):
    cur = obj
    for attr in ("model", "diffusion_model", "cond_stage_model", "first_stage_model"):
        nxt = getattr(cur, attr, None)
        if nxt is not None:
            cur = nxt
    return cur


def _bind_module_device(obj, device):
    inner = _unwrap_nn(obj)
    if inner is not None and hasattr(inner, "to"):
        try:
            inner.to(device)
        except Exception:
            pass
    if hasattr(obj, "device"):
        try:
            obj.device = device
        except Exception:
            pass
    patcher = getattr(obj, "patcher", None) or getattr(obj, "model_patcher", None)
    if patcher is not None:
        for attr in ("load_device", "offload_device", "current_device"):
            if hasattr(patcher, attr):
                try:
                    setattr(patcher, attr, device)
                except Exception:
                    pass


def _comfy_manages_weights(module):
    from comfy.model_patcher import ModelPatcher

    return isinstance(module, ModelPatcher)


def _apply_stage_shards(module, schedule, report, frac, label):
    if _comfy_manages_weights(module):
        # ComfyUI owns placement, pinning, and LoRA patches for these weights.
        # Installing a second block streamer conflicts with its low-VRAM loader.
        lead = str(module.load_device)
        report(f"schedule_{label}", frac, f"{label}: ComfyUI weight offload on {lead}")
        return {"mode": "comfy_offload", "lead_device": lead, "blocks_by_gpu": {}, "streamer": None}
    summary = apply_proportional_block_sharding(_unwrap_nn(module), schedule)
    mode = summary.get("mode")
    lead = summary.get("lead_device")
    if mode == "pinned_block_streaming":
        report(f"schedule_{label}", frac, f"{label}: pinned block streaming on {lead}")
    elif mode == "proportional_sharding":
        blocks = summary.get("blocks_by_gpu") or {}
        parts = [f"gpu{idx}:{len(idxs)}" for idx, idxs in sorted(blocks.items())]
        report(f"schedule_{label}", frac, f"{label}: proportional shards {' '.join(parts)} lead={lead}")
    return summary


def _tighten_video_vae(vae, usable_mb):
    if usable_mb >= TIGHT_VRAM_THRESHOLD_MB:
        return
    chunk = 2 if usable_mb >= 4096 else 1
    for obj in (vae, getattr(vae, "first_stage_model", None)):
        if obj is None:
            continue
        for name in ("tokens_chunk_size", "decode_chunk_size"):
            cur = getattr(obj, name, None)
            if isinstance(cur, int):
                setattr(obj, name, min(cur, chunk))


def _failure_text(exc):
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return " ".join(text.split())[:180]


def _cuda_oom(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur).lower()
        if (
            type(cur).__name__ == "OutOfMemoryError"
            or "out of memory" in text
            or "cublas_status_alloc_failed" in text
            or "cudamalloc" in text
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _count_refs(payload):
    if payload.get("task") == "i2v":
        return sum(1 for path in (payload.get("keyframes") or {}).values() if path)
    refs = payload.get("references") or {}
    return len(refs.get("images") or []) + len(refs.get("videos") or []) + len(refs.get("audios") or [])


def _at(result, index):
    return result[index]


@torch.inference_mode()
def run_job(payload, registry, progress, abort_event):
    def report(stage, frac, detail):
        if progress is not None:
            progress(stage, max(0.0, min(1.0, float(frac))), detail)

    def checkpoint():
        if abort_event is not None and abort_event.is_set():
            raise JobAborted()

    checkpoint()
    root = payload.get("model_root") or registry.model_root
    registry.model_root = root
    models = payload["models"]
    mode = payload.get("mode", "normal")
    width = int(payload["width"])
    height = int(payload["height"])
    first_frame = (payload.get("keyframes") or {}).get("first")
    if payload.get("task") == "i2v" and first_frame:
        width, height = _first_frame_canvas(first_frame, width * height)
    frames = int(payload["frames"])
    fps = int(payload.get("fps", 24))
    seed = int(payload.get("seed", 0))
    steps = max(1, int(payload["steps"]))
    denoise = float(payload.get("denoise", 1.0))
    scheduler = payload.get("scheduler", "simple")
    ref_mode = payload.get("ref_image_size", "match")
    cn_cfg = payload.get("controlnet_config") or {}
    strength = float(payload.get("lora_strength", 1.0))
    num_refs = _count_refs(payload)
    req = estimate_stage_memory_mb("stage1", width, height, frames, num_refs)
    schedule = registry.probe_and_plan("stage1", required_vram_mb=req)
    plan = registry.plan
    enc_dev = plan["clip"]
    dit_dev = plan["diffusion"]
    lead = schedule.lead_device

    roles = " ".join(f"gpu{g.index}:{g.role}/{g.allocated_weight_pct:.0f}%" for g in schedule.gpus)
    report(
        "preflight_gate",
        0.02,
        f"Checking RAM/storage headroom ({width}x{height}, {frames}f) lead={lead} {roles} - {describe_plan(plan)}",
    )
    check_host_gate(width, height, frames)

    report("init_comfy", 0.03, "Initializing headless ComfyUI runtime")
    if lead.type == "cuda":
        torch.cuda.set_device(lead)
    boot_comfy(root)
    import comfy.model_management as mm

    mm.interrupt_current_processing(False)
    try:
        return _run_job_inner(
            payload, registry, report, checkpoint, models, mode, width, height, frames, fps,
            seed, steps, denoise, scheduler, ref_mode, cn_cfg, strength, plan, enc_dev, dit_dev,
            num_refs,
        )
    except mm.InterruptProcessingException as exc:
        raise JobAborted() from exc


def _run_job_inner(
    payload, registry, report, checkpoint, models, mode, width, height, frames, fps,
    seed, steps, denoise, scheduler, ref_mode, cn_cfg, strength, plan, enc_dev, dit_dev,
    num_refs,
):
    keep = set()

    def load(slot, subdir, name, loader, frac, label, key=None):
        verb = "Reusing cached" if registry.cached(slot, key or name) else "Loading"
        report(f"load_{slot}", frac, f"{verb} {label} ({name}) for {plan[slot]}")
        obj = registry.get_or_swap(slot, subdir, name, loader, key=key, keep=keep)
        keep.add(slot)
        checkpoint()
        return obj

    clip = load("clip", "clip", models["clip"], _load_clip, 0.04, "text encoder")
    video_vae = load(
        "video_vae", "vae", models["video_vae"], lambda p: _load_vae(p, torch.float16), 0.07, "video VAE (FP16)"
    )
    audio_vae = load(
        "audio_vae", "vae", models["audio_vae"], lambda p: _load_vae(p, torch.float32), 0.08, "audio VAE (FP32)"
    )
    control = None
    if cn_cfg.get("enabled"):
        if not models.get("controlnet") or not cn_cfg.get("path"):
            raise BadReference()
        control = load("controlnet", "controlnet", models["controlnet"], _load_controlnet, 0.16, "ControlNet")

    # CLIPLoader pins the full ~16GB encoder. Drop it so encode reloads with the 9GB activation floor.
    report("stage1_encode", 0.18, f"Releasing full Qwen pin on {enc_dev} before vision encode")
    _unload_device(enc_dev)
    registry.trim_all_memory()
    report("stage1_encode", 0.19, f"Stage 1: encoding prompt & references on {enc_dev}")

    def _encode_stage1():
        with device_scope(enc_dev):
            if payload.get("task") == "i2v":
                cond, lat = _stage1_i2v(clip, video_vae, payload, width, height, frames)
            else:
                cond, lat = _stage1_condition(clip, video_vae, audio_vae, payload, width, height, frames, ref_mode)
            if control is not None:
                report("stage1_controlnet", 0.22, "Applying ControlNet conditioning")
                cond = _apply_controlnet(cond, control, video_vae, cn_cfg, frames, height, width)
        return cond, lat

    try:
        positive, latent = _encode_stage1()
    except ValueError as exc:
        raise BadReference() from exc
    except Exception as exc:
        if not _cuda_oom(exc):
            report("stage1_encode", 0.19, _failure_text(exc))
            raise
        report(
            "stage1_encode",
            0.19,
            "Vision encode ran out of VRAM; streaming Qwen weights from RAM and retrying",
        )
        clip._mmh3_activation_floor = 12 * GIB
        _unload_device(enc_dev)
        registry.trim_all_memory()
        try:
            positive, latent = _encode_stage1()
        except ValueError as exc:
            raise BadReference() from exc
    checkpoint()

    req2 = estimate_stage_memory_mb("stage2", width, height, frames, num_refs)
    schedule2 = registry.probe_and_plan("stage2", required_vram_mb=req2)
    dit_dev = schedule2.lead_device
    plan = registry.plan
    if enc_dev == dit_dev:
        report("stage1_offload", 0.24, f"Offloading Stage 1 encoders from {enc_dev}")
        _unload_device(enc_dev)
    else:
        report("stage1_offload", 0.24, f"Encoders stay resident on {enc_dev}; DiT has {dit_dev} to itself")
    if not registry.cached("diffusion", models["diffusion"]):
        registry.evict("lora")
    runner = load("diffusion", "diffussion", models["diffusion"], _load_unet, 0.10, "DiT backbone")
    if mode == "turbo":
        base = runner
        runner = load(
            "lora",
            "lora",
            models["lora"],
            lambda p: _load_lora(p, base, strength),
            0.13,
            "4-step Turbo LoRA",
            key=f"{models['diffusion']}|{models['lora']}|{strength:.4f}",
        )
    else:
        registry.evict("lora")
    if _comfy_manages_weights(runner):
        # A cached patcher keeps the GPU it was loaded for; ComfyUI samples there, so gate that GPU.
        dit_dev = runner.load_device
    report("stage2_load", 0.27, f"Stage 2: checking activation memory on {dit_dev}")
    _apply_stage_shards(runner, schedule2, report, 0.28, "dit")
    check_vram_gate(runner, positive, latent, dit_dev)
    with device_scope(dit_dev):
        sampled = _stage2_sample(
            runner, positive, latent, steps, denoise, scheduler, seed, checkpoint, report, registry,
            task=payload.get("task", "ref2v"),
        )
    checkpoint()
    if enc_dev == dit_dev:
        report("stage2_offload", 0.86, f"Offloading Stage 2 DiT from {dit_dev}")
        _unload_device(dit_dev)

    req3 = estimate_stage_memory_mb("stage3", width, height, frames, num_refs)
    schedule3 = registry.probe_and_plan("stage3", required_vram_mb=req3)
    healthy = sorted(schedule3.active_gpus, key=lambda g: (-g.usable_vram_mb, g.index))
    if not healthy:
        video_dev = audio_dev = schedule3.lead_device
        sequential = True
        usable_mb = 0.0
    elif len(healthy) >= 2:
        video_dev = healthy[0].device
        audio_dev = healthy[1].device
        sequential = False
        usable_mb = healthy[0].usable_vram_mb
    else:
        video_dev = audio_dev = healthy[0].device
        sequential = True
        usable_mb = healthy[0].usable_vram_mb
    report(
        "stage3_decode",
        0.90,
        f"Stage 3: video VAE (FP16) on {video_dev}, audio VAE (FP32) on {audio_dev}"
        + (" sequential" if sequential else ""),
    )
    check_host_gate(width, height, frames)
    frames_rgb, waveform = _stage3_decode(
        video_vae, audio_vae, sampled, video_dev, audio_dev, sequential, usable_mb
    )
    checkpoint()

    report("stage3_mux", 0.97, "Muxing MP4 with FFmpeg")
    try:
        _mux(frames_rgb, waveform, width, height, fps, payload["output_path"])
    except Exception as exc:
        raise MuxFailed() from exc
    del frames_rgb, waveform, sampled
    registry.trim_all_memory()
    report("stage3_mux", 1.0, "ready")


def _first_frame_canvas(path, area):
    """I2V stretches the first frame onto the canvas, so keep its aspect at the requested pixel area (32 px grid)."""
    from PIL import Image

    try:
        with Image.open(_require_media(path)) as image:
            w, h = image.size
    except Exception as exc:
        raise BadReference() from exc
    scale = math.sqrt(area / float(w * h))
    return max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)


def _stage1_i2v(clip, video_vae, payload, width, height, frames):
    """t2va / fl2va: both keyframes optional; the node stretches the first and cover-crops the last."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo

    keyframes = payload.get("keyframes") or {}
    frames_in = {
        f"{key}_frame": _load_image(keyframes[key], width, height, "max").permute(1, 2, 0).unsqueeze(0).contiguous()
        for key in ("first", "last")
        if keyframes.get(key)
    }
    result = MiniMaxH3ImageToVideo.execute(
        clip=clip, vae=video_vae, prompt=payload.get("prompt") or "", width=width, height=height, length=frames,
        **frames_in,
    )
    return _at(result, 0), _at(result, 1)


def _stage1_condition(clip, video_vae, audio_vae, payload, width, height, frames, ref_mode):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    references = payload.get("references") or {}
    ref_images, ref_videos, ref_video_audios, ref_audios = _reference_pack(
        references, width, height, frames, ref_mode
    )
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=clip,
        vae=video_vae,
        audio_vae=audio_vae,
        prompt=payload.get("prompt") or "",
        width=width,
        height=height,
        length=frames,
        ref_image_size=ref_mode if ref_mode in {"match", "max"} else "match",
        ref_images=ref_images or None,
        ref_videos=ref_videos or None,
        ref_video_audios=ref_video_audios or None,
        ref_audios=ref_audios or None,
    )
    return _at(result, 0), _at(result, 1)


def _reference_pack(references, width, height, frames, ref_mode):
    """Pack reference latents in prompt order: images, then each soundtrack before its video, then standalone audio.

    Tensors stay on the CPU; ComfyUI's VAE encode moves each batch to the VAE's own GPU.
    """
    images = {}
    for index, path in enumerate((references.get("images") or [])[:9]):
        chw = _load_image(path, width, height, ref_mode)
        images[f"ref_image_{index}"] = chw.permute(1, 2, 0).unsqueeze(0).contiguous()
    videos = {}
    video_audios = {}
    paired = references.get("video_audios") or []
    for index, path in enumerate((references.get("videos") or [])[:3]):
        clip = _load_video_native(path, frames)
        videos[f"ref_video_{index}"] = clip.permute(0, 2, 3, 1).contiguous()
        clip_seconds = max(0.25, float(clip.shape[0]) / 24.0)
        audio_path = paired[index] if index < len(paired) and paired[index] else None
        if audio_path:
            wave = _load_audio_full(audio_path, required=True, max_seconds=clip_seconds)
        else:
            wave = _load_audio_full(path, required=False, max_seconds=clip_seconds)
        if wave is not None:
            video_audios[f"ref_video_audio_{index}"] = _audio_payload(wave)
    audios = {}
    max_audio_sec = max(15.0, float(frames) / 24.0)
    for index, path in enumerate((references.get("audios") or [])[:3]):
        audios[f"ref_audio_{index}"] = _audio_payload(
            _load_audio_full(path, required=True, max_seconds=max_audio_sec)
        )
    return images, videos, video_audios, audios


def _audio_payload(wave):
    stereo = wave if wave.shape[-1] == 2 else wave[:, :1].repeat(1, 2)
    return {"waveform": stereo.transpose(0, 1).unsqueeze(0).contiguous().float(), "sample_rate": 48000}


def _apply_controlnet(positive, control_net, vae, cn_cfg, frames, height, width):
    import comfy.utils
    from nodes import ControlNetApplyAdvanced

    path = cn_cfg["path"]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".webm", ".mov", ".mkv"}:
        image = _load_video(path, width, height, frames).permute(0, 2, 3, 1).contiguous()
    else:
        image = _load_image(path, width, height, "exact").permute(1, 2, 0).unsqueeze(0).contiguous()
    positive, _negative = ControlNetApplyAdvanced().apply_controlnet(
        positive,
        positive,
        control_net,
        image,
        float(cn_cfg.get("strength", 0.8)),
        float(cn_cfg.get("start_percent", 0.0)),
        float(cn_cfg.get("end_percent", 0.85)),
        vae=vae,
    )
    for entry in positive:
        cnet = entry[1].get("control") if len(entry) > 1 and isinstance(entry[1], dict) else None
        if cnet is not None and not getattr(cnet, "_mmh3_packed_wrapped", False):
            orig_get_control = cnet.get_control

            def _get_control(x_noisy, t, cond, batched_number, transformer_options, _orig=orig_get_control):
                if getattr(x_noisy, "ndim", 0) == 3 and isinstance(cond, dict) and "latent_shapes" in cond:
                    x_noisy = comfy.utils.unpack_latents(x_noisy, cond["latent_shapes"])[0]
                try:
                    return _orig(x_noisy, t, cond, batched_number, transformer_options)
                except Exception:
                    return None

            cnet.get_control = _get_control
            cnet._mmh3_packed_wrapped = True
    return positive


def _is_fasth3(model):
    """FastH3 checkpoints are VSA-trained and ship its coarse-attention gates."""
    try:
        blocks = model.get_model_object("diffusion_model").blocks
    except AttributeError:
        return False
    return bool(blocks) and getattr(blocks[0].attn, "to_gate_compress", None) is not None


def _has_control(positive):
    return any(len(c) > 1 and isinstance(c[1], dict) and c[1].get("control") is not None for c in positive or ())


# ComfyUI's FastH3 templates per task: (video shift, VSA start_percent, sampler).
# OmniRef (ref2v) runs VSA from the first step with euler; FastH3 V2 (i2v/t2v) shifts 10 and goes sparse after 20%.
FASTH3_RECIPES = {"ref2v": (12.0, 0.0, "euler"), "i2v": (10.0, 0.2, "res_multistep")}


def _patch_attention(model, positive, task="ref2v"):
    """Same model patches as ComfyUI's H3 templates; returns (model, sampler name)."""
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend

    fast = _is_fasth3(model)
    shift, vsa_start, sampler = FASTH3_RECIPES[task] if fast else (12.0, None, "res_multistep")
    model = _at(MiniMaxH3SigmaShift.execute(model, shift, 3.0), 0)
    # Comfy Kitchen INT8 attention; the node itself falls back to PyTorch attention where the kernel is missing.
    model = _at(ModelAttentionBackend.execute(model, "comfy kitchen attention"), 0)
    if fast and not _has_control(positive):
        from comfy_extras.nodes_sparse_attention import BlockSparseAttention

        # FastH3 is trained against VSA at keep 10% and must never run plain block-sparse attention.
        # Skipped under ControlNet: VSA claims every block patch the control injection needs (dense is safe).
        model = _at(BlockSparseAttention.execute(model, {"selection": "vsa", "keep_percent": 10.0}, vsa_start, 1.0), 0)
    if not _has_control(positive):
        _install_chunked_blocks(model)
    return model, sampler


# A DiT block holds ~100 KB of activations per token (fused qkv, attention copies, the 2x14336 MLP input), so a
# 1920x1088x362 clip (~219k tokens) needs ~22 GB inside one block. Above this size blocks run in token chunks
# with the same math; smaller jobs keep ComfyUI's own block untouched.
CHUNK_ABOVE_TOKENS = 120_000
CHUNK_TOKENS = 16_384


def _seg_slice(segments, i, j):
    """Modulation segments restricted to rows [i, j), re-based to the chunk."""
    out = []
    for a, b, row in segments:
        lo, hi = max(a, i), min(b, j)
        if lo < hi:
            out.append((lo - i, hi - i, row[lo - a:hi - a] if torch.is_tensor(row) and row.ndim else row))
    return out


def _chunk_qkv(block, x, segments, rope_freqs, shift, scale, i, j):
    """Normed, rotated q/k/v for rows [i, j) as [1, heads, rows, dim]."""
    import comfy.model_management as mm
    import comfy.quant_ops
    from comfy.ldm.minimax.model import _mod_scale_shift

    attn = block.attn
    heads, dim = attn.heads, attn.head_dim
    h = _mod_scale_shift(block.norm1(x[i:j]), shift, scale, _seg_slice(segments, i, j))
    q, k, v = attn.qkv_proj(h).split(heads * dim, dim=-1)
    del h
    q, k = q.view(1, j - i, heads, dim), k.view(1, j - i, heads, dim)
    if rope_freqs is not None:
        qw = mm.cast_to(attn.q_norm.weight, device=x.device)
        kw = mm.cast_to(attn.k_norm.weight, device=x.device)
        comfy.quant_ops.ck.rms_rope_split_half_(
            q, k, rope_freqs[:, i:j], qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q, k = attn.q_norm(q), attn.k_norm(k)
    return q.transpose(1, 2), k.transpose(1, 2), v.view(1, j - i, heads, dim).transpose(1, 2)


def _attend(q, k, v):
    """Comfy Kitchen INT8 attention (unequal q/k lengths are supported) as the H3 templates use; else PyTorch SDPA."""
    import comfy_kitchen

    if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and comfy_kitchen.int8_attention_is_available(q.device):
        return comfy_kitchen.int8_attention(q.contiguous(), k, v)
    return F.scaled_dot_product_attention(q, k, v)


def _chunked_attention(block, x, segments, rope_freqs, shift, scale, gate, spans):
    """Dense self-attention over all keys with only K/V held for the whole sequence; outputs gate into x.

    Each chunk's queries are rebuilt from its rows just before those rows are updated (one extra qkv projection,
    small next to dense attention), so nothing reads a row after it changed.
    """
    from comfy.ldm.minimax.model import _mod_gate

    k_all = v_all = None
    for i, j in spans:
        _q, k, v = _chunk_qkv(block, x, segments, rope_freqs, shift, scale, i, j)
        if k_all is None:  # the projection's dtype, not the residual stream's
            k_all, v_all = (k.new_empty(k.shape[:2] + (x.shape[0], k.shape[-1])) for _ in range(2))
        k_all[:, :, i:j] = k
        v_all[:, :, i:j] = v
    for i, j in spans:
        q, _k, _v = _chunk_qkv(block, x, segments, rope_freqs, shift, scale, i, j)
        out = _attend(q, k_all, v_all)
        out = block.attn.out_proj(out[0].transpose(0, 1).reshape(j - i, -1))
        _mod_gate(x[i:j], gate, out, _seg_slice(segments, i, j))


# ponytail: one pinned buffer for the latest residual shape (~2.3 GB at 1080p 15 s), kept for the next block/job.
_PARKED = {}


@contextlib.contextmanager
def _parked(t):
    """Hold a CUDA tensor's data in pinned RAM for the duration; its storage is freed and restored in place."""
    storage = t.untyped_storage()
    nbytes = storage.nbytes()
    if not t.is_cuda or nbytes != t.numel() * t.element_size():
        yield
        return
    key = (tuple(t.shape), t.dtype)
    if key not in _PARKED:
        _PARKED.clear()
        _PARKED[key] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host = _PARKED[key]
    host.copy_(t)
    storage.resize_(0)
    try:
        yield
    finally:
        storage.resize_(nbytes)
        t.copy_(host)


def _chunked_block(block, x, t_emb, segments, rope_freqs, transformer_options, attention=None):
    """DiTBlock.forward over token chunks. A VSA `attention` already builds q/k/v in slices, so it gets the full h
    while the residual stream waits in pinned RAM (the caller still references it, so it cannot simply be freed)."""
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

    n = x.shape[0]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    spans = [(i, min(n, i + CHUNK_TOKENS)) for i in range(0, n, CHUNK_TOKENS)]
    if attention is None:
        _chunked_attention(block, x, segments, rope_freqs, shift_msa, scale_msa, gate_msa, spans)
    else:
        h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, segments)
        with _parked(x):
            out = attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
            del h
        x = _mod_gate(x, gate_msa, out, segments)
        del out
    for i, j in spans:
        seg = _seg_slice(segments, i, j)
        h = _mod_scale_shift(block.norm2(x[i:j]), shift_mlp, scale_mlp, seg)
        _mod_gate(x[i:j], gate_mlp, block.mlp(h), seg)
    return x


def _install_chunked_blocks(model):
    """Wrap every H3 block, and any VSA patch already on it, so long sequences take _chunked_block."""
    try:
        blocks = model.get_model_object("diffusion_model").blocks
    except AttributeError:
        return
    dit = model.model_options["transformer_options"].get("patches_replace", {}).get("dit", {})
    for index, block in enumerate(blocks):
        model.set_model_patch_replace(_chunked_patch(block, dit.get(("double_block", index))), "dit", "double_block", index)


def _chunked_patch(block, previous):
    def run(args):
        return {"img": _chunked_block(block, args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                      args["transformer_options"], args.get("attention"))}

    def patch(args, extra):
        if args["img"].shape[0] > CHUNK_ABOVE_TOKENS:
            extra = {**extra, "original_block": run}
        return previous(args, extra) if previous is not None else extra["original_block"](args)

    return patch


def _stage2_sample(model, positive, latent, steps, denoise, scheduler, seed, checkpoint, report, registry=None,
                   task="ref2v"):
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )

    def _once():
        checkpoint()
        shifted_model, sampler_name = _patch_attention(model, positive, task)
        guider = _at(BasicGuider.execute(shifted_model, positive), 0)
        sampler = _at(KSamplerSelect.execute(sampler_name), 0)
        sigmas = _at(BasicScheduler.execute(shifted_model, scheduler, steps, denoise), 0)
        noise = _at(RandomNoise.execute(int(seed) & 0xFFFFFFFFFFFFFFFF), 0)
        _hook_progress(guider, steps, report, checkpoint)
        report("stage2_denoise", 0.30, f"step 0/{steps}")
        sampled = _at(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent), 0)
        report("stage2_denoise", 0.84, f"step {steps}/{steps}")
        return sampled

    try:
        return _once()
    except Exception as exc:
        if registry is None or not _cuda_oom(exc) or _comfy_manages_weights(model):
            raise
        report("stage2_oom_recover", 0.30, "CUDA OOM — trimming and resuming via pinned block streaming")
        registry.trim_all_memory()
        schedule = registry.probe_and_plan("stage2_oom")
        schedule.stream_from_pinned_ram = True
        apply_proportional_block_sharding(_unwrap_nn(model), schedule)
        return _once()


def _hook_progress(guider, steps, report, checkpoint):
    original = guider.sample

    def sample(*args, **kwargs):
        callback = kwargs.get("callback")

        def wrapped(step, *rest, **kw):
            checkpoint()
            done = int(step) + 1
            report(
                "stage2_denoise",
                0.30 + 0.54 * (done / max(1, steps)),
                f"step {done}/{steps}",
            )
            if callback is not None:
                callback(step, *rest, **kw)

        kwargs["callback"] = wrapped
        return original(*args, **kwargs)

    guider.sample = sample


def _stage3_decode(video_vae, audio_vae, sampled, video_dev=None, audio_dev=None, sequential=True, usable_mb=0.0):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    from nodes import VAEDecode

    if video_dev is None:
        video_dev = torch.device("cpu")
    if audio_dev is None:
        audio_dev = video_dev
    _tighten_video_vae(video_vae, usable_mb)
    _bind_module_device(video_vae, video_dev)
    if not sequential:
        _bind_module_device(audio_vae, audio_dev)
    separated = LTXVSeparateAVLatent.execute(sampled)
    with device_scope(video_dev):
        images = VAEDecode().decode(video_vae, _at(separated, 0))[0]
    frames_rgb = images.float().clamp(0, 1).cpu()
    del images
    if sequential:
        _unload_device(video_dev)
        _bind_module_device(audio_vae, audio_dev)
    with device_scope(audio_dev):
        waveform = _decode_audio(audio_vae, _at(separated, 1)).cpu()
    return frames_rgb, waveform


def _decode_audio(vae, audio_latent):
    from comfy_extras.nodes_audio import VAEDecodeAudio

    audio = _at(VAEDecodeAudio.execute(vae, audio_latent), 0)
    return _waveform_48k(audio)


def _waveform_48k(audio):
    wave = audio["waveform"].float()
    if wave.ndim == 3:
        wave = wave[0]
    if wave.shape[0] <= 8 and wave.shape[0] < wave.shape[-1]:
        wave = wave.transpose(0, 1)
    if wave.shape[-1] == 1:
        wave = wave.repeat(1, 2)
    elif wave.shape[-1] > 2:
        wave = wave[:, :2]
    rate = int(audio.get("sample_rate") or 48000)
    if rate != 48000 and wave.shape[0] > 1:
        import torchaudio

        wave = torchaudio.functional.resample(wave.transpose(0, 1).contiguous(), rate, 48000).transpose(0, 1)
    return wave.contiguous().float()


def _ephemeral_root():
    return os.path.realpath(os.environ.get("EPHEMERAL_DIR", "/app/ephemeral"))


def _require_media(path):
    if not path or not isinstance(path, str):
        raise BadReference()
    root = _ephemeral_root()
    real = os.path.realpath(path)
    if not real.startswith(root + os.sep) or not os.path.isfile(real):
        raise BadReference()
    return real


def _load_image(path, width, height, ref_mode):
    from PIL import Image

    real = _require_media(path)
    try:
        image = Image.open(real).convert("RGB")
    except Exception as exc:
        raise BadReference() from exc
    array = torch.from_numpy(np.asarray(image)).float() / 255.0
    return _resize_image(array, width, height, ref_mode)


def _resize_image(image, gen_w, gen_h, ref_mode):
    height, width = image.shape[:2]
    if ref_mode == "exact":
        new_w, new_h = int(gen_w), int(gen_h)
    else:
        if ref_mode == "max":
            short = min(height, width)
            scale = 1.0 if short <= 2048 else 2048.0 / float(short)
        else:
            scale = min(1.0, math.sqrt((gen_w * gen_h) / max(1, width * height)))
        new_w = max(32, int(round((width * scale) / 32.0) * 32))
        new_h = max(32, int(round((height * scale) / 32.0) * 32))
    tensor = image.permute(2, 0, 1).unsqueeze(0)
    if new_h == height and new_w == width:
        return tensor[0].contiguous()
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return tensor[0]


def _load_video(path, width, height, target_frames):
    frames = _load_video_native(path, target_frames)
    return F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)


def _ffmpeg_threads():
    cores = os.cpu_count() or 8
    return str(max(2, min(16, cores // 4)))


def _load_video_native(path, target_frames):
    real = _require_media(path)
    probed = _ffprobe_size(real)
    if probed is None:
        raise BadReference()
    src_w, src_h = probed
    max_frames = max(5, int(target_frames))
    check_host_gate(src_w, src_h, max_frames)
    raw = _ffmpeg_raw(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            _ffmpeg_threads(),
            "-i",
            real,
            "-an",
            "-vf",
            "fps=24",
            "-frames:v",
            str(max_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_size = src_w * src_h * 3
    if frame_size <= 0 or len(raw) < frame_size * 5:
        raise BadReference()
    count = min(max_frames, len(raw) // frame_size)
    keep = _largest_aligned(count)
    if keep < 5:
        raise BadReference()
    frames = torch.frombuffer(bytearray(memoryview(raw)[: keep * frame_size]), dtype=torch.uint8).reshape(
        keep, src_h, src_w, 3
    )
    frames = frames.float().div_(255.0)
    return frames.permute(0, 3, 1, 2).contiguous()


def _largest_aligned(count):
    raw = max(5, count)
    rem = (5 - (raw % 17)) % 17
    aligned = raw + rem
    if aligned <= count:
        return aligned
    value = count
    while value >= 5 and ((value - 5) % 17) != 0:
        value -= 1
    return value


def _load_audio_full(path, required, max_seconds=15.0):
    try:
        real = _require_media(path)
    except BadReference:
        if required:
            raise
        return None
    try:
        raw = _ffmpeg_raw(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                real,
                "-vn",
                "-t",
                f"{max(0.25, float(max_seconds)):.3f}",
                "-f",
                "f32le",
                "-ac",
                "2",
                "-ar",
                "48000",
                "pipe:1",
            ]
        )
    except BadReference:
        # FFmpeg exits non-zero when a reference video has no audio stream.
        if required:
            raise
        return None
    if len(raw) < 8:
        if required:
            raise BadReference()
        return None
    usable = len(raw) - (len(raw) % 8)
    return torch.frombuffer(bytearray(raw[:usable]), dtype=torch.float32).reshape(-1, 2).clone()


def _ffprobe_size(path):
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    text = proc.stdout.decode("ascii", "ignore").strip()
    if "x" not in text:
        return None
    width_s, height_s = text.split("x", 1)
    try:
        width = int(width_s)
        height = int(height_s)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _ffmpeg_raw(cmd):
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError as exc:
        raise BadReference() from exc
    if proc.returncode != 0:
        raise BadReference()
    return proc.stdout


def _mux(frames_rgb, waveform, width, height, fps, output_path):
    parent = os.path.realpath(os.path.dirname(output_path))
    outputs = os.path.realpath(os.path.join(_ephemeral_root(), "outputs"))
    if parent != outputs:
        raise MuxFailed()
    os.makedirs(outputs, exist_ok=True)
    video = (frames_rgb.clamp(0, 1) * 255.0).round().to(torch.uint8).contiguous()
    if video.shape[1] != height or video.shape[2] != width or video.shape[-1] != 3:
        video = F.interpolate(
            video.permute(0, 3, 1, 2).float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        video = video.permute(0, 2, 3, 1).round().to(torch.uint8).contiguous()
    audio = waveform.float().contiguous()
    target_samples = max(1, int(round(video.shape[0] / 24.0 * 48000)))
    if audio.ndim != 2 or audio.shape[1] != 2:
        audio = torch.zeros(target_samples, 2)
    elif audio.shape[0] > target_samples:
        audio = audio[:target_samples].contiguous()
    elif audio.shape[0] < target_samples:
        pad = torch.zeros(target_samples - audio.shape[0], 2, dtype=audio.dtype)
        audio = torch.cat([audio, pad], dim=0).contiguous()
    audio_read, audio_write = os.pipe()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        "24",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-i",
        "pipe:3",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags",
        "+bitexact",
        "-c:v",
        "libx264",
        "-threads",
        _ffmpeg_threads(),
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
    ]
    if int(fps) != 24:
        cmd.extend(["-vf", f"minterpolate=fps={int(fps)}:mi_mode=mci"])
    cmd.append(output_path)

    def _preexec():
        os.dup2(audio_read, 3)
        if audio_read != 3:
            os.close(audio_read)

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=_preexec,
            pass_fds=(3,),
        )
    except OSError as exc:
        os.close(audio_read)
        os.close(audio_write)
        raise MuxFailed() from exc
    os.close(audio_read)

    def _write_audio():
        try:
            view = audio.numpy().tobytes()
            os.write(audio_write, view)
        finally:
            os.close(audio_write)

    writer = threading.Thread(target=_write_audio, daemon=True)
    writer.start()
    try:
        proc.stdin.write(video.numpy().tobytes())
        proc.stdin.close()
    except Exception:
        proc.kill()
        writer.join(timeout=5)
        _secure_unlink(output_path)
        raise MuxFailed()
    writer.join()
    err = proc.stderr.read() if proc.stderr is not None else b""
    code = proc.wait()
    if code != 0 or not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        _ = err
        _secure_unlink(output_path)
        raise MuxFailed()


def _secure_unlink(path):
    if not path or not os.path.isfile(path):
        return
    size = os.path.getsize(path)
    with open(path, "r+b") as handle:
        remaining = size
        zeros = b"\x00" * 65536
        while remaining > 0:
            chunk = zeros if remaining >= len(zeros) else zeros[:remaining]
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    os.remove(path)
