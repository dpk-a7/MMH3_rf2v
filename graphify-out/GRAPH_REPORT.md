# Graph Report - /home/deepak/DeepakProj/MMH3  (2026-10-17)

## Corpus Check
- 22 files · ~45,882 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 325 nodes · 634 edges · 37 communities detected
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 118 edges (avg confidence: 0.74)
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

## God Nodes (most connected - your core abstractions)
1. `run_job()` - 26 edges
2. `run()` - 18 edges
3. `PinnedModelRegistry` - 18 edges
4. `RowParallelLinear` - 15 edges
5. `ColumnParallelLinear` - 14 edges
6. `_split_bounds()` - 12 edges
7. `BadReference` - 12 edges
8. `generate_video()` - 11 edges
9. `attach_tensor_parallel()` - 11 edges
10. `_shard_diffusion_on_cpu()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `known_aspect()` --calls--> `generate_video()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/math.rs → /home/deepak/DeepakProj/MMH3/backend-rust/src/api.rs
- `run()` --calls--> `serve()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/main.rs → /home/deepak/DeepakProj/MMH3/inference-worker/worker_daemon.py
- `A reference image, video, or audio clip failed validation.` --uses--> `PinnedModelRegistry`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py
- `Pack reference latents in prompt order: images, then each soundtrack before its` --uses--> `PinnedModelRegistry`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py → /home/deepak/DeepakProj/MMH3/inference-worker/memory_manager.py
- `toBytes()` --calls--> `_mux()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/frontend/src/cryptoClient.ts → /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (48): AppState, AvailableModels, ControlnetConfig, detect_gpus(), dispatch(), download_output(), ensure_model_file(), EnvelopeFile (+40 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (47): trim_all_memory(), attach_ulysses(), _apply_controlnet(), _at(), _audio_payload(), BadReference, boot_comfy(), _call_patcher() (+39 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (35): attach_tensor_parallel(), _block_body(), block_sparse_attention(), _chunked_attention(), collective_abort(), ColumnParallelLinear, _dense_weight(), _gather_sequence() (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (22): flushAll(), generate(), handleProgress(), onControlFile(), onPick(), pullOutput(), replaceSlot(), uploadSlot() (+14 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (19): encode_envelope(), envelope(), handshake(), sealed_error(), session_from_headers(), both_sides_derive_the_same_session_key(), CryptoError, decode_public_key() (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (16): basename(), Exception, PinnedModelRegistry, ComfyPinned, JobAborted, _load_clip(), _load_controlnet(), _load_unet() (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (13): client_thread(), coordinator(), emit(), error_frame(), follower(), init_parallel(), main(), mark_abort() (+5 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (11): can_pin_bytes(), load_pinned(), _pin(), PinnedModule, Full checkpoint resident in pinned host RAM, moved stage by stage., ComfyUI-style pin budget guard: keep >= max(8 GB, 12% of RAM) headroom unpinned., ComfyUI-style usable VRAM budget for model weights on `device`., read_tensors() (+3 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (9): flush_all(), connect_retry(), IpcClient, IpcFrame, ProgressEvent, read_frame(), sanitize_code(), write_frame() (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.36
Nodes (2): EphemeralStorage, restrict_dir()

### Community 10 - "Community 10"
Cohesion: 0.31
Nodes (6): aspect_components(), compute_aligned_frame_length(), compute_dimensions(), dimensions_stay_on_grid(), frame_length_is_congruent_five_mod_seventeen(), known_aspect()

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
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): The caller aborted the running generation.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): A reference image, video, or audio clip failed validation.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Bitexact FFmpeg muxing did not produce an output file.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): ComfyUI model object moved with the 3-stage pinned-RAM registry.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

## Knowledge Gaps
- **48 isolated node(s):** `ProgressEvent`, `IpcFrame`, `CryptoError`, `GpuInfo`, `ModelCatalogResponse` (+43 more)
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
- **Thin community `Community 19`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Replace DiT and Qwen3-VL QKV / O projections and SwiGLU linears with checkpoint-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `The caller aborted the running generation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `A reference image, video, or audio clip failed validation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Bitexact FFmpeg muxing did not produce an output file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `ComfyUI model object moved with the 3-stage pinned-RAM registry.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `basename()` connect `Community 5` to `Community 0`, `Community 7`?**
  _High betweenness centrality (0.158) - this node is a cross-community bridge._
- **Why does `run()` connect `Community 0` to `Community 9`, `Community 5`, `Community 6`, `Community 1`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `_mux()` connect `Community 1` to `Community 3`, `Community 5`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `run_job()` (e.g. with `coordinator()` and `follower()`) actually correct?**
  _`run_job()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `run()` (e.g. with `security_headers()` and `.new()`) actually correct?**
  _`run()` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `PinnedModelRegistry` (e.g. with `Return (csv_devices, count) skipping GPUs heavily occupied by external workloads` and `JobAborted`) actually correct?**
  _`PinnedModelRegistry` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `RowParallelLinear` (e.g. with `JobAborted` and `BadReference`) actually correct?**
  _`RowParallelLinear` has 6 INFERRED edges - model-reasoned connections that need verification._