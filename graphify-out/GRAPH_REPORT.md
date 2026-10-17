# Graph Report - /home/deepak/DeepakProj/MMH3  (2026-10-17)

## Corpus Check
- 21 files · ~47,007 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 294 nodes · 505 edges · 53 communities detected
- Extraction: 80% EXTRACTED · 20% INFERRED · 0% AMBIGUOUS · INFERRED: 99 edges (avg confidence: 0.75)
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

## God Nodes (most connected - your core abstractions)
1. `run_job()` - 20 edges
2. `run()` - 18 edges
3. `ResourceLimitExceeded` - 16 edges
4. `generate_video()` - 12 edges
5. `BadReference` - 12 edges
6. `ModelRegistry` - 12 edges
7. `open()` - 11 edges
8. `envelope()` - 9 edges
9. `_load_video_native()` - 9 edges
10. `dispatch()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `known_aspect()` --calls--> `generate_video()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/math.rs → /home/deepak/DeepakProj/MMH3/backend-rust/src/api.rs
- `run()` --calls--> `serve()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/main.rs → /home/deepak/DeepakProj/MMH3/inference-worker/worker_daemon.py
- `Reject before allocating when decoded frames would not fit in RAM or /app/epheme` --uses--> `ResourceLimitExceeded`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py
- `toBytes()` --calls--> `_mux()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/frontend/src/cryptoClient.ts → /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py
- `flushAll()` --calls--> `emptySlots()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/frontend/src/App.tsx → /home/deepak/DeepakProj/MMH3/frontend/src/components/ReferenceRack.tsx

## Communities

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (50): basename(), Exception, describe_plan(), device_scope(), Pre-flight VRAM/RAM/tmpfs safety gate blocked the job before OOM., ResourceLimitExceeded, _apply_controlnet(), _at() (+42 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (38): AppState, AvailableModels, cgroup_available_ram_bytes(), check_system_gates(), dispatch(), download_output(), ensure_model_file(), EnvelopeFile (+30 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (22): flushAll(), generate(), handleProgress(), onControlFile(), onPick(), pullOutput(), replaceSlot(), uploadSlot() (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (17): ControlnetConfig, detect_gpus(), fallback_gpus(), list_models(), router(), scan_safetensors(), sweep_sessions(), bind_addr() (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (18): open(), _cgroup_ram(), host_ram_info(), ModelRegistry, _nvidia_smi_free(), plan_devices(), ram_reserve_bytes(), [(torch_index, free_bytes, total_bytes)] for every visible CUDA device. (+10 more)

### Community 5 - "Community 5"
Cohesion: 0.2
Nodes (16): encode_envelope(), envelope(), handshake(), sealed_error(), session_from_headers(), both_sides_derive_the_same_session_key(), CryptoError, decode_public_key() (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.23
Nodes (15): client_thread(), coordinator(), emit(), error_frame(), init_runtime(), _is_oom(), main(), mark_abort() (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (8): flush_all(), connect_retry(), IpcClient, IpcFrame, ProgressEvent, read_frame(), sanitize_code(), write_frame()

### Community 8 - "Community 8"
Cohesion: 0.31
Nodes (6): aspect_components(), compute_aligned_frame_length(), compute_dimensions(), dimensions_stay_on_grid(), frame_length_is_congruent_five_mod_seventeen(), known_aspect()

### Community 9 - "Community 9"
Cohesion: 0.5
Nodes (0): 

### Community 10 - "Community 10"
Cohesion: 0.67
Nodes (0): 

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (0): 

### Community 12 - "Community 12"
Cohesion: 1.0
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
Nodes (1): Return the rank-0 error string when any rank failed. Collective when N>1.

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Replace DiT and Qwen3-VL QKV / O projections and SwiGLU/FFN linears with checkpo

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Verify host RAM, GPU VRAM, and /app/ephemeral tmpfs headroom before allocating.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Full checkpoint resident in pinned host RAM, moved stage by stage.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Return (available_bytes, total_bytes) from /proc/meminfo with safe fallback.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): ComfyUI-style usable VRAM budget for model weights on `device`.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): The caller aborted the running generation.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): A reference image, video, or audio clip failed validation.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Bitexact FFmpeg muxing did not produce an output file.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): The caller aborted the running generation.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): A reference image, video, or audio clip failed validation.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Bitexact FFmpeg muxing did not produce an output file.

## Knowledge Gaps
- **63 isolated node(s):** `ProgressEvent`, `IpcFrame`, `CryptoError`, `GpuInfo`, `ModelCatalogResponse` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 11`** (2 nodes): `PanicFlushButton.tsx`, `PanicFlushButton()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `tailwind.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `postcss.config.js`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `vite-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Return the rank-0 error string when any rank failed. Collective when N>1.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Replace DiT and Qwen3-VL QKV / O projections and SwiGLU/FFN linears with checkpo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Verify host RAM, GPU VRAM, and /app/ephemeral tmpfs headroom before allocating.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry and ComfyUI-styl`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Full checkpoint resident in pinned host RAM, moved stage by stage.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Return (available_bytes, total_bytes) from /proc/meminfo with safe fallback.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `ComfyUI-style usable VRAM budget for model weights on `device`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `The caller aborted the running generation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `A reference image, video, or audio clip failed validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Bitexact FFmpeg muxing did not produce an output file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `The caller aborted the running generation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `A reference image, video, or audio clip failed validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Bitexact FFmpeg muxing did not produce an output file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run()` connect `Community 3` to `Community 0`, `Community 1`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Why does `open()` connect `Community 4` to `Community 0`, `Community 1`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `flushAll()` connect `Community 2` to `Community 5`, `Community 7`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_job()` (e.g. with `coordinator()` and `describe_plan()`) actually correct?**
  _`run_job()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `run()` (e.g. with `security_headers()` and `.new()`) actually correct?**
  _`run()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `ResourceLimitExceeded` (e.g. with `Runs generate and flush commands strictly one at a time.` and `JobAborted`) actually correct?**
  _`ResourceLimitExceeded` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `generate_video()` (e.g. with `known_aspect()` and `.new()`) actually correct?**
  _`generate_video()` has 5 INFERRED edges - model-reasoned connections that need verification._