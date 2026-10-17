# Graph Report - /home/deepak/DeepakProj/MMH3  (2026-10-17)

## Corpus Check
- 22 files · ~44,395 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 298 nodes · 557 edges · 28 communities detected
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 94 edges (avg confidence: 0.76)
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

## God Nodes (most connected - your core abstractions)
1. `run_job()` - 25 edges
2. `run()` - 17 edges
3. `ColumnParallelLinear` - 13 edges
4. `RowParallelLinear` - 13 edges
5. `generate_video()` - 11 edges
6. `BadReference` - 10 edges
7. `envelope()` - 9 edges
8. `basename()` - 9 edges
9. `_shard_attention()` - 8 edges
10. `_shard_swiglu()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `known_aspect()` --calls--> `generate_video()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/math.rs → /home/deepak/DeepakProj/MMH3/backend-rust/src/api.rs
- `run()` --calls--> `serve()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/backend-rust/src/main.rs → /home/deepak/DeepakProj/MMH3/inference-worker/worker_daemon.py
- `attach_ulysses()` --calls--> `run_job()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/inference-worker/parallel_shard.py → /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py
- `toBytes()` --calls--> `_mux()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/frontend/src/cryptoClient.ts → /home/deepak/DeepakProj/MMH3/inference-worker/pipeline_h3.py
- `flushAll()` --calls--> `emptySlots()`  [INFERRED]
  /home/deepak/DeepakProj/MMH3/frontend/src/App.tsx → /home/deepak/DeepakProj/MMH3/frontend/src/components/ReferenceRack.tsx

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (48): AppState, AvailableModels, ControlnetConfig, detect_gpus(), dispatch(), download_output(), ensure_model_file(), EnvelopeFile (+40 more)

### Community 1 - "Community 1"
Cohesion: 0.1
Nodes (39): Exception, trim_all_memory(), _apply_controlnet(), _at(), _audio_payload(), BadReference, boot_comfy(), _decode_audio() (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.1
Nodes (33): attach_tensor_parallel(), attach_ulysses(), _block_body(), block_sparse_attention(), _chunked_attention(), collective_abort(), ColumnParallelLinear, _dense_weight() (+25 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (22): flushAll(), generate(), handleProgress(), onControlFile(), onPick(), pullOutput(), replaceSlot(), uploadSlot() (+14 more)

### Community 4 - "Community 4"
Cohesion: 0.2
Nodes (17): encode_envelope(), envelope(), handshake(), sealed_error(), session_from_headers(), both_sides_derive_the_same_session_key(), CryptoError, decode_public_key() (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (13): client_thread(), coordinator(), emit(), error_frame(), follower(), init_parallel(), main(), mark_abort() (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.23
Nodes (13): basename(), _call_patcher(), ComfyPinned, _iter_modules(), _load_clip(), _load_controlnet(), _load_unet(), _load_vae() (+5 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (9): flush_all(), connect_retry(), IpcClient, IpcFrame, ProgressEvent, read_frame(), sanitize_code(), write_frame() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.21
Nodes (8): load_pinned(), _pin(), PinnedModelRegistry, PinnedModule, Full checkpoint resident in pinned host RAM, moved stage by stage., read_tensors(), _stable_slot(), shard_first_dim()

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
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Pack reference latents in prompt order: images, then each soundtrack before its

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start

## Knowledge Gaps
- **41 isolated node(s):** `ProgressEvent`, `IpcFrame`, `CryptoError`, `GpuInfo`, `ModelCatalogResponse` (+36 more)
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
- **Thin community `Community 19`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Column-slice a stacked [Q; K; V] matrix so each rank owns the same heads of all`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Column-slice a SwiGLU gate||up matrix so each rank owns matching rows of both ha`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Replace DiT qkv / out projections and SwiGLU linears with checkpoint-sliced para`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Pack reference latents in prompt order: images, then each soundtrack before its`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Sequence-parallel attention. q/k/v are [B, H, S, D] and replicated.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Exact sink rows/keys, block-sparse body past 12288 tokens.      tau = 1.3, start`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `basename()` connect `Community 6` to `Community 0`, `Community 8`?**
  _High betweenness centrality (0.156) - this node is a cross-community bridge._
- **Why does `run_job()` connect `Community 1` to `Community 8`, `Community 2`, `Community 5`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Why does `run()` connect `Community 0` to `Community 9`, `Community 5`, `Community 1`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `run_job()` (e.g. with `coordinator()` and `follower()`) actually correct?**
  _`run_job()` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `run()` (e.g. with `security_headers()` and `.new()`) actually correct?**
  _`run()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `ColumnParallelLinear` (e.g. with `JobAborted` and `BadReference`) actually correct?**
  _`ColumnParallelLinear` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `RowParallelLinear` (e.g. with `JobAborted` and `BadReference`) actually correct?**
  _`RowParallelLinear` has 6 INFERRED edges - model-reasoned connections that need verification._