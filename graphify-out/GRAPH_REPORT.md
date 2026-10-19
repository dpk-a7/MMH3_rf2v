# Graph Report - /home/deepak/DeepakProj/MMH3  (2026-10-19)

## Corpus Check
- 24 files · ~60,459 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 414 nodes · 743 edges · 69 communities detected
- Extraction: 74% EXTRACTED · 26% INFERRED · 0% AMBIGUOUS · INFERRED: 194 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]

## God Nodes (most connected - your core abstractions)
1. `ResourceLimitExceeded` - 31 edges
2. `PinnedBlockStreamer` - 22 edges
3. `_run_job_inner()` - 21 edges
4. `run()` - 18 edges
5. `ModelRegistry` - 18 edges
6. `BadReference` - 16 edges
7. `StageSchedule` - 16 edges
8. `run_job()` - 14 edges
9. `generate_video()` - 13 edges
10. `build_stage_schedule()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `known_aspect()` --calls--> `generate_video()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/math.rs → /home/deepak/DeepakProj/MMH3/backend-rust/src/api.rs
- `run()` --calls--> `serve()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/main.rs → /home/deepak/DeepakProj/MMH3/inference-worker/worker_daemon.py
- `Reject before allocating when decoded frames would not fit in RAM or /app/epheme` --uses--> `ResourceLimitExceeded`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py
- `Activation budget ComfyUI must keep free before Qwen3-VL encode.      CLIPLoader` --uses--> `ResourceLimitExceeded`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py
- `Pack reference latents in prompt order: images, then each soundtrack before its` --uses--> `ResourceLimitExceeded`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (53): AppState, AvailableModels, cgroup_available_ram_bytes(), check_system_gates(), classify_gpu_roles(), ControlnetConfig, detect_gpus(), dispatch() (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (40): build_stage_schedule(), device(), elect_lead_device(), estimate_stage_memory_mb(), PinnedBlockStreamer, probe_live_gpus(), Real-time pre-stage VRAM probe across all visible CUDA devices 0..N-1.      Comb, Elect `lead_device = argmax(usable_vram)` with zero `cuda:0` anchoring. (+32 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (28): open(), can_pin_bytes(), _cgroup_ram(), describe_plan(), device_scope(), GPUStatus, host_ram_info(), _is_mocked() (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (40): basename(), _apply_controlnet(), _apply_stage_shards(), _at(), _audio_payload(), _bind_module_device(), boot_comfy(), _count_refs() (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (22): flushAll(), generate(), handleProgress(), onControlFile(), onPick(), pullOutput(), replaceSlot(), uploadSlot() (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (22): flush_all(), connect_retry(), GpuInfo, IpcClient, IpcFrame, ProgressEvent, read_frame(), sanitize_code() (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (27): Exception, Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM., ResourceLimitExceeded, BadReference, check_vram_gate(), JobAborted, MuxFailed, The caller aborted the running generation. (+19 more)

### Community 7 - "Community 7"
Cohesion: 0.2
Nodes (16): encode_envelope(), envelope(), handshake(), sealed_error(), session_from_headers(), both_sides_derive_the_same_session_key(), CryptoError, decode_public_key() (+8 more)

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (2): EphemeralStorage, restrict_dir()

### Community 9 - "Community 9"
Cohesion: 0.31
Nodes (6): aspect_components(), compute_aligned_frame_length(), compute_dimensions(), dimensions_stay_on_grid(), frame_length_is_congruent_five_mod_seventeen(), known_aspect()

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (3): clip_activation_bytes(), Activation budget ComfyUI must keep free before Qwen3-VL encode.      CLIPLoader, Stage1ClipVramTests

### Community 11 - "Community 11"
Cohesion: 0.5
Nodes (0): 

### Community 12 - "Community 12"
Cohesion: 0.67
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): (available, limit) inside a cgroup-v2 memory limit, or None when unlimited.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Return (available_bytes, total_bytes), honouring the container memory limit.

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Physical index -> (free_bytes, total_bytes) without creating CUDA contexts.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): [(torch_index, free_bytes, total_bytes)] for every visible CUDA device.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): [(torch_index, free_bytes, total_bytes)] for every visible CUDA device.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): (available, limit) inside a cgroup-v2 memory limit, or None when unlimited.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Return (available_bytes, total_bytes), honouring the container memory limit.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Physical index -> (free_bytes, total_bytes) without creating CUDA contexts.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): [(torch_index, free_bytes, total_bytes)] for every visible CUDA device.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Return the rank-0 error string when any rank failed. Collective when N>1.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Replace DiT and Qwen3-VL QKV / O projections and SwiGLU/FFN linears with checkpo

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Verify host RAM, GPU VRAM, and /app/ephemeral tmpfs headroom before allocating.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Full checkpoint resident in pinned host RAM, moved stage by stage.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Return (available_bytes, total_bytes) from /proc/meminfo with safe fallback.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): ComfyUI-style usable VRAM budget for model weights on `device`.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): The caller aborted the running generation.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): The caller aborted the running generation.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): A reference image, video, or audio clip failed validation.

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Bitexact FFmpeg muxing did not produce an output file.

## Knowledge Gaps
- **96 isolated node(s):** `GpuInfo`, `ProgressEvent`, `IpcFrame`, `CryptoError`, `ModelCatalogResponse` (+91 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 13`** (2 nodes): `PanicFlushButton.tsx`, `PanicFlushButton()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `tailwind.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `vite-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `(available, limit) inside a cgroup-v2 memory limit, or None when unlimited.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Return (available_bytes, total_bytes), honouring the container memory limit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Physical index -> (free_bytes, total_bytes) without creating CUDA contexts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `[(torch_index, free_bytes, total_bytes)] for every visible CUDA device.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `[(torch_index, free_bytes, total_bytes)] for every visible CUDA device.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `(available, limit) inside a cgroup-v2 memory limit, or None when unlimited.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Return (available_bytes, total_bytes), honouring the container memory limit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Physical index -> (free_bytes, total_bytes) without creating CUDA contexts.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `[(torch_index, free_bytes, total_bytes)] for every visible CUDA device.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `One model per GPU: DiT on the emptiest card, text encoder + VAEs on the next one`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Caches one ComfyUI model object per slot; ComfyUI itself manages VRAM residency.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Return the rank-0 error string when any rank failed. Collective when N>1.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Replace DiT and Qwen3-VL QKV / O projections and SwiGLU/FFN linears with checkpo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Verify host RAM, GPU VRAM, and /app/ephemeral tmpfs headroom before allocating.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Full checkpoint resident in pinned host RAM, moved stage by stage.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Return (available_bytes, total_bytes) from /proc/meminfo with safe fallback.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `ComfyUI-style usable VRAM budget for model weights on `device`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `The caller aborted the running generation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `The caller aborted the running generation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `A reference image, video, or audio clip failed validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Bitexact FFmpeg muxing did not produce an output file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run()` connect `Community 0` to `Community 8`, `Community 2`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.118) - this node is a cross-community bridge._
- **Why does `open()` connect `Community 2` to `Community 0`, `Community 3`, `Community 5`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `_mux()` connect `Community 3` to `Community 2`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 27 inferred relationships involving `ResourceLimitExceeded` (e.g. with `Runs generate and flush commands strictly one at a time.` and `JobAborted`) actually correct?**
  _`ResourceLimitExceeded` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `PinnedBlockStreamer` (e.g. with `ParallelContext` and `ColumnParallelLinear`) actually correct?**
  _`PinnedBlockStreamer` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `_run_job_inner()` (e.g. with `.trim_all_memory()` and `estimate_stage_memory_mb()`) actually correct?**
  _`_run_job_inner()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `run()` (e.g. with `security_headers()` and `.new()`) actually correct?**
  _`run()` has 10 INFERRED edges - model-reasoned connections that need verification._