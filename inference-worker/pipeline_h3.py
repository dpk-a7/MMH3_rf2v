import math
import os
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.nn.functional as F

from parallel_shard import (
    CTX,
    attach_tensor_parallel,
    attach_ulysses,
    collective_abort,
    note_step_percent,
    sync_error,
)

_COMFY_READY = False


class JobAborted(Exception):
    """The caller aborted the running generation."""


class BadReference(Exception):
    """A reference image, video, or audio clip failed validation."""


class MuxFailed(Exception):
    """Bitexact FFmpeg muxing did not produce an output file."""


class ComfyPinned:
    """ComfyUI model object moved with the 3-stage pinned-RAM registry."""

    def __init__(self, obj):
        self.obj = obj
        self.parallel = False
        self.marker = ""
        self.move_to("cpu")

    def move_to(self, device):
        target = device if isinstance(device, torch.device) else torch.device(device)
        if hasattr(self.obj, "device"):
            self.obj.device = target
        patcher = _patcher(self.obj)
        if patcher is not None:
            if target.type == "cpu":
                _call_patcher(patcher, "unpatch_model", target)
            else:
                patcher.load_device = target
                _call_patcher(patcher, "patch_model", target)
            _pin_tree(patcher)
        else:
            for module in _iter_modules(self.obj):
                module.to(target)
                if target.type == "cpu":
                    _pin_module(module)
        if target.type != "cpu":
            for module in _iter_modules(self.obj):
                module.to(target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _call_patcher(patcher, name, device):
    method = getattr(patcher, name)
    try:
        method(device_to=device, unpatch_weights=True) if name == "unpatch_model" else method(device_to=device)
    except TypeError:
        try:
            method(device)
        except TypeError:
            method()


def _patcher(obj):
    if hasattr(obj, "patch_model") and hasattr(obj, "unpatch_model"):
        return obj
    inner = getattr(obj, "patcher", None)
    if inner is not None and hasattr(inner, "patch_model"):
        return inner
    return None


def _iter_modules(obj):
    seen = set()
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, torch.nn.Module):
            if id(current) in seen:
                continue
            seen.add(id(current))
            yield current
        if current is None or isinstance(current, (str, bytes, int, float)):
            continue
        for name in ("patcher", "model", "cond_stage_model", "first_stage_model", "diffusion_model"):
            child = getattr(current, name, None)
            if child is not None and not isinstance(child, (str, bytes, int, float)):
                stack.append(child)


def _pin_tree(obj):
    for module in _iter_modules(obj):
        _pin_module(module)


def _pin_module(module):
    if not torch.cuda.is_available():
        return
    for param in module.parameters():
        if param.device.type == "cpu" and param.is_floating_point() and not param.is_pinned():
            param.data = param.data.pin_memory()


def _dit(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "diffusion_model"):
        return inner.diffusion_model
    if hasattr(model, "diffusion_model"):
        return model.diffusion_model
    return model


def _force_dtype(pinned, dtype):
    if hasattr(pinned.obj, "vae_dtype"):
        pinned.obj.vae_dtype = dtype
    for module in _iter_modules(pinned.obj):
        module.to(dtype=dtype)


def boot_comfy(model_root):
    global _COMFY_READY
    if _COMFY_READY:
        return
    root = os.environ.get("COMFYUI_ROOT", "/opt/comfyui")
    if root not in sys.path:
        sys.path.insert(0, root)
    import logging

    logging.disable(logging.CRITICAL)
    import comfy.utils
    import folder_paths

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


def _load_clip(path):
    from nodes import CLIPLoader

    clip = CLIPLoader().load_clip(_rel_model_name(path, "clip"), type="minimax", device="cpu")[0]
    return ComfyPinned(clip)


def _load_vae(path):
    from nodes import VAELoader

    vae = VAELoader().load_vae(_rel_model_name(path, "vae"))[0]
    return ComfyPinned(vae)


def _load_unet(path):
    from nodes import UNETLoader

    model = UNETLoader().load_unet(_rel_model_name(path, "diffussion"), "default")[0]
    return ComfyPinned(model)


def _load_controlnet(path):
    from nodes import ControlNetLoader

    control = ControlNetLoader().load_controlnet(_rel_model_name(path, "controlnet"))[0]
    return ComfyPinned(control)


def _at(result, index):
    return result[index]


@torch.inference_mode()
def run_job(payload, registry, progress, abort_event):
    def report(stage, frac, detail):
        if CTX.rank == 0 and progress is not None:
            progress(stage, max(0.0, min(1.0, float(frac))), detail)

    def raise_for(code):
        if code == "aborted":
            raise JobAborted()
        if code == "model_missing":
            raise FileNotFoundError(code)
        if code == "bad_reference":
            raise BadReference()
        if code == "mux_failed":
            raise MuxFailed()
        raise RuntimeError(code or "inference_failed")

    def checkpoint():
        local = bool(abort_event is not None and abort_event.is_set()) if CTX.rank == 0 else False
        if collective_abort(local):
            raise JobAborted()

    checkpoint()
    root = payload.get("model_root") or registry.model_root
    registry.model_root = root
    models = payload["models"]
    mode = payload.get("mode", "normal")
    width = int(payload["width"])
    height = int(payload["height"])
    frames = int(payload["frames"])
    fps = int(payload.get("fps", 24))
    seed = int(payload.get("seed", 0))
    steps = max(1, int(payload["steps"]))
    denoise = float(payload.get("denoise", 1.0))
    scheduler = payload.get("scheduler", "simple")
    ref_mode = payload.get("ref_image_size", "match")
    device = CTX.device
    world = CTX.world if CTX.active() else 1
    cn_cfg = payload.get("controlnet_config") or {}
    strength = float(payload.get("lora_strength", 1.0))
    identity = "|".join(
        [
            models.get("diffusion") or "",
            mode,
            models.get("lora") or "",
            f"{strength:.4f}",
            models.get("controlnet") or "",
            "1" if cn_cfg.get("enabled") else "0",
        ]
    )
    if registry.loaded_slots["diffusion"].get("identity") != identity:
        for slot in ("diffusion", "lora", "controlnet"):
            registry.loaded_slots[slot]["filename"] = None
            registry.loaded_slots[slot]["module"] = None

    err = None
    loaded = {}
    try:
        boot_comfy(root)
        loaded["clip"] = registry.get_or_swap("clip", "clip", models["clip"], _load_clip)
        loaded["video_vae"] = registry.get_or_swap("video_vae", "vae", models["video_vae"], _load_vae)
        loaded["audio_vae"] = registry.get_or_swap("audio_vae", "vae", models["audio_vae"], _load_vae)
        loaded["diffusion"] = registry.get_or_swap(
            "diffusion", "diffussion", models["diffusion"], _load_unet
        )
        active_diffusion = loaded["diffusion"]
        if mode == "turbo":
            base = loaded["diffusion"]
            lora_marker = f"{models.get('diffusion')}|{models.get('lora')}|{strength:.4f}"
            lora_slot = registry.loaded_slots["lora"]
            if lora_slot.get("module") is not None and getattr(lora_slot["module"], "marker", None) != lora_marker:
                lora_slot["filename"] = None
                lora_slot["module"] = None
                lora_slot["weights"] = None
                registry.trim_all_memory()

            def _load_lora(path, _base=base, _strength=strength, _marker=lora_marker):
                from nodes import LoraLoaderModelOnly

                patched = LoraLoaderModelOnly().load_lora_model_only(
                    _base.obj, _rel_model_name(path, "lora"), _strength
                )[0]
                pinned = ComfyPinned(patched)
                pinned.marker = _marker
                return pinned

            loaded["lora"] = registry.get_or_swap("lora", "lora", models["lora"], _load_lora)
            active_diffusion = loaded["lora"]
        else:
            registry.loaded_slots["lora"]["filename"] = None
            registry.loaded_slots["lora"]["module"] = None
            registry.loaded_slots["lora"]["weights"] = None
        if cn_cfg.get("enabled") and models.get("controlnet"):
            loaded["controlnet"] = registry.get_or_swap(
                "controlnet", "controlnet", models["controlnet"], _load_controlnet
            )
        registry.loaded_slots["diffusion"]["identity"] = identity
    except FileNotFoundError:
        err = "model_missing"
    except BadReference:
        err = "bad_reference"
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)

    report("stage1_load", 0.05, "clip_vae")
    err = None
    try:
        registry.activate(["clip", "video_vae", "audio_vae"], device)
        _force_dtype(loaded["video_vae"], torch.float16)
        _force_dtype(loaded["audio_vae"], torch.float32)
        if world > 1 and not loaded["clip"].parallel:
            attach_tensor_parallel(loaded["clip"].obj, world, CTX.rank)
            loaded["clip"].parallel = True
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)
    checkpoint()

    positive = None
    latent = None
    err = None
    try:
        positive, latent = _stage1_condition(
            loaded, payload, width, height, frames, ref_mode, device
        )
        if "controlnet" in loaded and cn_cfg.get("path"):
            registry.activate(["controlnet"], device)
            positive = _apply_controlnet(
                positive, loaded["controlnet"].obj, loaded["video_vae"].obj, cn_cfg, frames, height, width, device
            )
            registry.offload(["controlnet"])
    except BadReference:
        err = "bad_reference"
    except ValueError:
        err = "bad_reference"
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)
    report("stage1_encode", 0.2, "condition")
    err = None
    try:
        registry.offload(["clip", "video_vae", "audio_vae"])
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)
    report("stage1_offload", 0.24, "pinned")
    checkpoint()

    report("stage2_load", 0.28, "dit")
    stage2 = ["lora" if mode == "turbo" else "diffusion"]
    if "controlnet" in loaded:
        stage2.append("controlnet")
    err = None
    try:
        registry.activate(stage2, device)
        runner = active_diffusion
        if not runner.parallel:
            if world > 1:
                attach_tensor_parallel(_dit(runner.obj), world, CTX.rank)
            attach_ulysses(runner.obj)
            _watch_steps(_dit(runner.obj))
            runner.parallel = True
        _reset_steps(_dit(runner.obj), steps)
    except BadReference:
        err = "bad_reference"
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)

    sampled = None
    err = None
    try:
        sampled = _stage2_sample(
            active_diffusion.obj, positive, latent, steps, denoise, scheduler, seed, checkpoint, report
        )
    except JobAborted:
        err = "aborted"
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)
    err = None
    try:
        registry.offload(stage2)
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)
    report("stage2_offload", 0.86, "latent")
    checkpoint()

    report("stage3_decode", 0.9, "vae")
    err = None
    frames_rgb = None
    waveform = None
    try:
        frames_rgb, waveform = _stage3_decode(registry, loaded, sampled, device, world)
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        if world >= 2 and CTX.active():
            torch.distributed.barrier()
        raise_for(err)
    if world >= 2 and CTX.active():
        if CTX.rank == 0:
            waveform = _recv_waveform(1, device)
        elif CTX.rank == 1:
            _send_waveform(waveform, 0)
        # Ranks >= 2 are already waiting here; ranks 0 and 1 join after the waveform exchange.
        torch.distributed.barrier()

    err = None
    try:
        registry.offload(["video_vae", "audio_vae"])
    except Exception:
        err = "inference_failed"
    err = sync_error(err)
    if err:
        raise_for(err)

    mux_error = None
    if CTX.rank == 0:
        try:
            _mux(frames_rgb.detach().cpu(), waveform.detach().cpu(), width, height, fps, payload["output_path"])
        except Exception:
            mux_error = "mux_failed"
    if CTX.active():
        torch.distributed.barrier()
        holder = [mux_error]
        torch.distributed.broadcast_object_list(holder, src=0)
        mux_error = holder[0]
    if mux_error:
        raise MuxFailed()
    report("stage3_mux", 1.0, "ready")
    registry.trim_all_memory()


def _watch_steps(diffusion):
    if getattr(diffusion, "_mmh3_wrapped", False):
        return
    original = diffusion.forward

    def forward(*args, **kwargs):
        seen = int(getattr(diffusion, "_mmh3_seen", 0)) + 1
        total = max(1, int(getattr(diffusion, "_mmh3_steps", 1)))
        diffusion._mmh3_seen = seen
        note_step_percent(seen / total)
        return original(*args, **kwargs)

    diffusion.forward = forward
    diffusion._mmh3_wrapped = True


def _reset_steps(diffusion, steps):
    diffusion._mmh3_seen = 0
    diffusion._mmh3_steps = max(1, int(steps))


def _stage1_condition(loaded, payload, width, height, frames, ref_mode, device):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    references = payload.get("references") or {}
    ref_images, ref_videos, ref_video_audios, ref_audios = _reference_pack(
        references, width, height, frames, ref_mode, device
    )
    result = MiniMaxH3ReferenceToVideo.execute(
        clip=loaded["clip"].obj,
        vae=loaded["video_vae"].obj,
        audio_vae=loaded["audio_vae"].obj,
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


def _reference_pack(references, width, height, frames, ref_mode, device):
    """Pack reference latents in prompt order: images, then each soundtrack before its video, then standalone audio.

    MiniMaxH3ReferenceToVideo encodes images and videos with the video VAE and
    both audio groups with the audio VAE, and stores those latents on the
    positive conditioning plus the joint AV latent.
    """
    images = {}
    for index, path in enumerate((references.get("images") or [])[:9]):
        chw = _load_image(path, width, height, ref_mode).to(device)
        images[f"ref_image_{index}"] = chw.permute(1, 2, 0).unsqueeze(0).contiguous()
    videos = {}
    video_audios = {}
    paired = references.get("video_audios") or []
    for index, path in enumerate((references.get("videos") or [])[:3]):
        clip = _load_video_native(path, frames).to(device)
        videos[f"ref_video_{index}"] = clip.permute(0, 2, 3, 1).contiguous()
        audio_path = paired[index] if index < len(paired) and paired[index] else None
        wave = None
        if audio_path:
            wave = _load_audio_full(audio_path, required=True)
        else:
            wave = _load_audio_full(path, required=False)
        if wave is not None:
            video_audios[f"ref_video_audio_{index}"] = _audio_payload(wave.to(device))
    audios = {}
    for index, path in enumerate((references.get("audios") or [])[:3]):
        wave = _load_audio_full(path, required=True).to(device)
        audios[f"ref_audio_{index}"] = _audio_payload(wave)
    return images, videos, video_audios, audios


def _audio_payload(wave):
    stereo = wave if wave.shape[-1] == 2 else wave[:, :1].repeat(1, 2)
    return {"waveform": stereo.transpose(0, 1).unsqueeze(0).contiguous().float(), "sample_rate": 48000}


def _apply_controlnet(positive, control_net, vae, cn_cfg, frames, height, width, device):
    from nodes import ControlNetApplyAdvanced

    path = cn_cfg["path"]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mp4", ".webm", ".mov", ".mkv"}:
        clip = _load_video(path, width, height, frames).to(device)
        image = clip.permute(0, 2, 3, 1).contiguous()
    else:
        chw = _load_image(path, width, height, "match").to(device)
        image = chw.permute(1, 2, 0).unsqueeze(0).contiguous()
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
    return positive


def _stage2_sample(model, positive, latent, steps, denoise, scheduler, seed, checkpoint, report):
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    checkpoint()
    shifted_model = _at(MiniMaxH3SigmaShift.execute(model, 12.0, 3.0), 0)
    guider = _at(BasicGuider.execute(shifted_model, positive), 0)
    sampler = _at(KSamplerSelect.execute("res_multistep"), 0)
    sigmas = _at(BasicScheduler.execute(shifted_model, scheduler, steps, denoise), 0)
    noise = _at(RandomNoise.execute(int(seed) & 0xFFFFFFFFFFFFFFFF), 0)
    _hook_progress(guider, steps, report, checkpoint)
    report("stage2_denoise", 0.30, f"step 0/{steps}")
    sampled = _at(SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent), 0)
    report("stage2_denoise", 0.84, f"step {steps}/{steps}")
    return sampled


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


def _stage3_decode(registry, loaded, sampled, device, world):
    from comfy_extras.nodes_lt import LTXVSeparateAVLatent
    from nodes import VAEDecode

    separated = LTXVSeparateAVLatent.execute(sampled)
    video_latent = _at(separated, 0)
    audio_latent = _at(separated, 1)
    frames_rgb = None
    waveform = None
    if world == 1 or not CTX.active():
        registry.activate(["video_vae"], device)
        _force_dtype(loaded["video_vae"], torch.float16)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            images = VAEDecode().decode(loaded["video_vae"].obj, video_latent)[0]
        frames_rgb = images.float().clamp(0, 1)
        registry.offload(["video_vae"])
        registry.activate(["audio_vae"], device)
        _force_dtype(loaded["audio_vae"], torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            waveform = _decode_audio(loaded["audio_vae"].obj, audio_latent)
        return frames_rgb, waveform
    if CTX.rank == 0:
        registry.activate(["video_vae"], device)
        _force_dtype(loaded["video_vae"], torch.float16)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            images = VAEDecode().decode(loaded["video_vae"].obj, video_latent)[0]
        frames_rgb = images.float().clamp(0, 1)
    elif CTX.rank == 1:
        registry.activate(["audio_vae"], device)
        _force_dtype(loaded["audio_vae"], torch.float32)
        with torch.autocast(device_type=device.type, enabled=False):
            waveform = _decode_audio(loaded["audio_vae"].obj, audio_latent)
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


def _send_waveform(waveform, dst):
    data = waveform.detach().to(dtype=torch.float32).contiguous()
    if data.ndim != 2:
        data = data.reshape(-1, 2)
    payload = data.to(CTX.device)
    meta = torch.tensor([payload.shape[0], payload.shape[1], 1], dtype=torch.int64, device=payload.device)
    torch.distributed.send(meta, dst=dst)
    torch.distributed.send(payload, dst=dst)


def _recv_waveform(src, device):
    meta = torch.zeros(3, dtype=torch.int64, device=device)
    torch.distributed.recv(meta, src=src)
    payload = torch.empty(int(meta[0].item()), int(meta[1].item()), dtype=torch.float32, device=device)
    torch.distributed.recv(payload, src=src)
    return payload


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
    if ref_mode == "max":
        short = min(height, width)
        scale = 1.0 if short <= 2048 else 2048.0 / float(short)
    else:
        scale = math.sqrt((gen_w * gen_h) / max(1, width * height))
    new_w = max(32, int(round((width * scale) / 32.0) * 32))
    new_h = max(32, int(round((height * scale) / 32.0) * 32))
    tensor = image.permute(2, 0, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return tensor[0]


def _load_video(path, width, height, target_frames):
    frames = _load_video_native(path, target_frames)
    return F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)


def _load_video_native(path, target_frames):
    real = _require_media(path)
    probed = _ffprobe_size(real)
    if probed is None:
        raise BadReference()
    src_w, src_h = probed
    raw = _ffmpeg_raw(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            real,
            "-an",
            "-vf",
            "fps=24",
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
    count = len(raw) // frame_size
    frames = torch.frombuffer(bytearray(raw[: count * frame_size]), dtype=torch.uint8).reshape(count, src_h, src_w, 3)
    frames = frames.float() / 255.0
    if count > target_frames:
        frames = frames[:target_frames]
    keep = _largest_aligned(int(frames.shape[0]))
    if keep < 5:
        raise BadReference()
    frames = frames[:keep]
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


def _load_audio_full(path, required):
    try:
        real = _require_media(path)
    except BadReference:
        if required:
            raise
        return None
    raw = _ffmpeg_raw(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            real,
            "-vn",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            "48000",
            "pipe:1",
        ]
    )
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
    if audio.ndim != 2 or audio.shape[1] != 2:
        audio = torch.zeros(max(1, int(video.shape[0] / 24.0 * 48000)), 2)
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
