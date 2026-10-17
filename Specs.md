### ADDENDUM: Single-Endpoint Blind Envelope Routing
- Apart from the initial key-exchange (`POST /api/v1/crypto/handshake`) and static UI asset serving (`GET /`), consolidate ALL API actions (`list_models`, `upload_media`, `generate_video`, `job_status`, `download_output`, `flush_all`) into a SINGLE unified endpoint: `POST /api/v1/envelope` (plus `GET /api/v1/ws` for encrypted binary WebSocket frames).
- The client must place the target action name, headers, metadata, and binary file buffers inside a single MessagePack structure, pad it to a 64 KB block boundary (to prevent file-size traffic analysis), and encrypt the entire MessagePack buffer with AES-256-GCM before sending it to `POST /api/v1/envelope`.
Transport Layer (In-Memory Ephemeral TLS 1.3 via rustls + rcgen): On startup, the Rust backend generates an ephemeral Ed25519 / ECDSA P-256 TLS 1.3 certificate purely in RAM (never touching disk). All HTTP and WebSocket traffic entering and exiting port 8443 uses encrypted HTTPS and WSS (TLS_AES_256_GCM_SHA384).

Application Payload Layer (WebCrypto X25519 + AES-256-GCM): On page load, the browser UI and Rust backend perform an ephemeral Diffie-Hellman (X25519) key exchange. Every uploaded reference image/video/audio, prompt JSON, and streamed output video chunk is encrypted in the browser with AES-256-GCM before entering the Docker container and encrypted by Rust before exiting to the browser.
---

# `AGENT_BLUEPRINT.md` — MiniMax-H3 Reference-to-Video (Ref2VA) Rust + Multi-GPU Application

## 1. System Objective & Architectural Overview

Build a strictly non-logging, memory-managed, multi-GPU **Reference-to-Video-and-Audio (Ref2VA)** web application powered by **MiniMax-H3**. The system translates the ComfyUI `MiniMaxH3ReferenceToVideo` workflow (`46a303cbccf9.json`) into a standalone, containerized production service with a **Rust (`axum` + `tokio`) backend**, a **distributed PyTorch/CUDA (`cu130`) inference worker pool** communicating over an in-memory Unix Domain Socket (`/dev/shm/minimax_h3.sock`), and a **reactive Web UI**.

```
+-----------------------------------------------------------------------------------+
|                              DOCKER CONTAINER (logging: none)                     |
|                                                                                   |
|  +-----------------------+     HTTP / WS (No Logs)     +-----------------------+  |
|  |   Web UI (Frontend)   | <-------------------------> |  Rust Backend (Axum)  |  |
|  |  - Ref Media Tagger   |                             |  - Zero-Log Middleware|  |
|  |  - Prompt <Tag> Editor|                             |  - 24h TTL Reaper     |  |
|  |  - 1-Click Media Flush|                             |  - 1-Click Purge API  |  |
|  +-----------------------+                             +-----------+-----------+  |
|                                                                    |              |
|                                         In-Memory Unix Socket      | /dev/shm     |
|                                         (Protobuf / MessagePack)   v              |
|  +-----------------------------------------------------------------------------+  |
|  |        Multi-GPU Inference Engine (1..N GPUs, single worker process)        |  |
|  |  - Staged Pinned-RAM Offloader (Phase 1: Encode -> Phase 2: DiT -> Phase 3) |  |
|  |  - Whole-Model GPU Placement + ComfyUI model_management (lowvram stream)   |  |
|  |  - Bitexact FFmpeg Muxer (stdin pipe -> /app/ephemeral/outputs, no metadata)|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+

```

---

## 2. Strict Directory Layout & Model Mapping

> **CRITICAL AGENT INSTRUCTION**: Respect the exact folder spelling `model/diffussion/` (with double `s`) as required by the host volume mount specification.

```text
minimax-h3-ref2v/
├── docker-compose.yml
├── Dockerfile
├── model/                                  # Mounted Read-Only (:ro) into container
│   ├── diffussion/                         # Exact spelling required
│   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   ├── lora/
│   │   └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
│   ├── controlnet/
│   │   └── minimax_h3_fun_controlnet_union.safetensors
│   ├── clip/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   └── vae/
│       ├── minimax_h3_video_vae_fp16.safetensors
│       └── minimax_h3_audio_vae_fp32.safetensors
├── backend-rust/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                         # Axum server, zero-log enforcement
│       ├── api.rs                          # REST + WebSocket progress & 1-click flush
│       ├── ipc.rs                          # Unix Domain Socket client to GPU Coordinator
│       ├── storage.rs                      # Ephemeral tmpfs manager, 24h TTL reaper, secure unlink
│       └── math.rs                         # Resolution grid & 17k+5 frame alignment math
├── inference-worker/
│   ├── worker_daemon.py                    # Single-process GPU daemon listening on /dev/shm
│   ├── memory_manager.py                   # Model cache, RAM gates, per-GPU placement planner
│   └── pipeline_h3.py                      # Exact node execution for MiniMaxH3ReferenceToVideo
└── frontend/
    ├── package.json
    └── src/
        ├── App.tsx                         # Main studio interface
        ├── components/
        │   ├── ReferenceRack.tsx           # 9 Images, 3 Videos (+Audio), 3 Standalone Audios
        │   ├── PromptTagEditor.tsx         # Auto-inserts <Picture N>, <Video N>, <Audio N>
        │   ├── GenerationControls.tsx      # Duration, Aspect, MP, FPS, Turbo LoRA, ControlNet
        │   └── PanicFlushButton.tsx        # 1-Click immediate purge of all stored/temp media

```

---

## 3. Exact Mathematical & Workflow Specifications (From `46a303cbccf9.json`

)

The coding agent must implement the exact mathematical transformations and tensor routing defined in the ComfyUI workflow:

### 3.1. Resolution & Aspect Ratio Calculation (Node `115`: `ResolutionSelector`)



MiniMax-H3 requires spatial dimensions rounded to a **multiple of 32**.

* **Inputs**:
* `aspect_ratio`: `"16:9"`, `"9:16"`, `"1:1"`, `"4:3"`, `"3:4"`, `"21:9"` (Default: `"16:9"`)


* `megapixels` ($M$): `0.2` to `2.0` (Default: `0.4` $\rightarrow 864 \times 480$; native ceiling is `1.0` $\rightarrow 1344 \times 768$)




* **Formula**:
Given aspect ratio $W_r : H_r$ and target pixel area $P = M \times 1024^2$:

$$\text{width} = \max\left(32, \text{round}\left(\frac{\sqrt{P \cdot (W_r / H_r)}}{32}\right) \times 32\right)$$


$$\text{height} = \max\left(32, \text{round}\left(\frac{\sqrt{P \cdot (H_r / W_r)}}{32}\right) \times 32\right)$$



| Megapixels ($M$) | Aspect Ratio | Computed Output ($\text{multiple}=32$) |
| --- | --- | --- |
| `0.2` | `16:9` | $608 \times 352$<br> |
| `0.4` (Default) | `16:9` | $864 \times 480$<br> |
| `0.7` | `16:9` | $1152 \times 640$<br> |
| `1.0` (Native Full) | `16:9` | $1344 \times 768$<br> |

### 3.2. Duration to Frame Length Alignment (Node `131` & `132`)



MiniMax-H3's 3D causal Video+Audio VAE operates at a native internal clock of **24 FPS** and requires the total frame count (`length`) to satisfy $L \equiv 5 \pmod{17}$ with $L \ge 5$.

* **Input**: `duration_seconds` ($a \in [1.0, 15.0]$, default `5.0`).


* **Exact Formula**:

$$F_{\text{raw}} = \max(5, \text{round}(a \times 24))$$


$$\text{length} = F_{\text{raw}} + \big(5 - (F_{\text{raw}} \bmod 17)\big) \bmod 17$$



*(Example: $a = 5.0\text{s} \implies F_{\text{raw}} = 120 \implies 120 + (5 - (120 \bmod 17)) \bmod 17 = 120 + 4 = 124\text{ frames}$.)*


### 3.3. Multi-Modal Reference Conditioning (Node `136`: `MiniMaxH3ReferenceToVideo`)



* **Reference Slots & Strict Ordering**:
1. `ref_images`: Up to **9** images (`ref_image_0` … `ref_image_8`). Addressed in prompt as `<Picture 1>` … `<Picture 9>`.


2. `ref_videos`: Up to **3** videos (`ref_video_0` … `ref_video_2`). Addressed in prompt as `<Video 1>` … `<Video 3>`.


* *Validation*: Each reference video is resampled to 24 FPS and **must contain $\ge 5$ frames** ($\sim 0.2\text{s}$). Trim frames exceeding `length` and align remaining frames to $17k + 5$.


3. `ref_video_audios`: Up to **3** paired audio tracks (`ref_video_audio_0` … `ref_video_audio_2`) extracted from or attached to `ref_video_N`. Placed immediately before its corresponding `<Video N>` token in the multimodal conditioning stream.


4. `ref_audios`: Up to **3** standalone audio clips (`ref_audio_0` … `ref_audio_2`). Addressed in prompt as `<Audio 1>` … `<Audio 3>`.




* **`ref_image_size` Mode**:
* `"match"` (Default): Downscales reference images (preserving aspect ratio) to match the target generation pixel area ($\text{width} \times \text{height}$) for fast sampling.


* `"max"`: Preserves up to a `2048px` short edge for maximum character/identity lock (increases reference token sequence length across every sampling step).




* **Encoders**:
* Text + Multimodal Vision/Audio tokens encoded via `model/clip/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (`type="minimax"`).


* Reference visual latents encoded via `model/vae/minimax_h3_video_vae_fp16.safetensors` (FP16).


* Reference audio latents encoded via `model/vae/minimax_h3_audio_vae_fp32.safetensors` (FP32).


* Outputs: `positive` conditioning + joint packed Audio-Video `LATENT`.





## 3.4. (Updated) Dynamic Model Selection, Normal/Turbo Mode Switch & Steps Slider

This section maps the switching logic in Nodes `141`–`146` and model loaders in Nodes `119`, `120`, `127`, `128`, and `145` into user-configurable controls with canonical defaults:

### 1. Dynamic Model Selector Mapping

The backend recursively scans `/app/model/` (and any nested subdirectories inside each category subfolder) on startup and whenever the user clicks **`↻ Rescan Subfolders`**, exposing every model weight file (`.safetensors`, `.sft`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`, `.onnx`) in each subfolder while pre-selecting the workflow defaults (or automatically snapping to the first discovered file in that subfolder if the default is not present on disk):

* **6-Slot Top-Level Model Selector Rack**: All 6 model categories are directly selectable in the top `Model & Sampling Controls` panel, displaying the live count of discovered files (`model/<subfolder>/ · N found`) and marking any absent preset with `(not found in folder)`:

| Model Category | Target Directory | Workflow Default File | Loader Details |
| --- | --- | --- | --- |
| **Diffusion (UNet/DiT)** | `model/diffussion/` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors`<br> | Loaded via `UNETLoader` (`weight_dtype = "default"`)

 |
| **Video VAE** | `model/vae/` | `minimax_h3_video_vae_fp16.safetensors`<br> | Loaded via `VAELoader` in FP16 for video latents (auto-matches `*video*` when scanning)

 |
| **Audio VAE** | `model/vae/` | `minimax_h3_audio_vae_fp32.safetensors`<br> | Loaded via `VAELoader` in FP32 for stereo audio latents (auto-matches `*audio*` when scanning)

 |
| **CLIP / Text Encoder** | `model/clip/` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`<br> | Loaded via `CLIPLoader` (`type = "minimax"`)

 |
| **Turbo / Style LoRA** | `model/lora/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`<br> | Selectable in top rack and Turbo panel; loaded via `LoraLoaderModelOnly` (`strength_model = 1.0`)

 |
| **ControlNet** | `model/controlnet/` | *None (Disabled by default)* | Selectable in top rack (`None (Disabled)` or any discovered file) and in the ControlNet drawer |

### 2. Generation Mode (`Normal` vs. `Turbo`) & Interactive Steps Slider

Instead of locking the step count behind a static boolean, the mode switch (`generation_mode: "normal" | "turbo"`) acts as an intelligent preset linked to an interactive **Steps Slider**:

* **Normal Mode (`generation_mode = "normal"`)**:
* Maps to `switch = false` on Node `141` (`If/Else Switch (model)`) and Node `142` (`If/Else Switch (Steps)`).


* Bypasses `LoraLoaderModelOnly` and routes the base diffusion model directly to `BasicGuider`.


* Automatically sets the **Steps Slider** to **`20`** (Node `143` `Int (Full)`), while allowing the user to slide between **`1` and `50` steps**.




* **Turbo Mode (`generation_mode = "turbo"`)**:
* Maps to `switch = true` on Nodes `141` and `142`.


* Patches the selected diffusion model through `LoraLoaderModelOnly` using the selected LoRA from `model/lora/` (defaulting to `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` at `strength_model = 1.0`).


* Automatically snaps the **Steps Slider** to **`4`** (Node `144` `Int (Lightning LoRA)`), while allowing the user to slide between **`1` and `20` steps** (e.g., bumping to 6 or 8 steps for complex multi-reference scenes).





### 3. Canonical Factory Defaults (`DEFAULT_SETTINGS`)

Clicking **"↺ Reset to Defaults"** restores the exact baseline state defined in the ComfyUI reference workflow:

| Parameter | Default Value | Workflow Source |
| --- | --- | --- |
| `diffusion_model` | `"minimax_h3_ref2va_pruned_int8_convrot.safetensors"` | Node `127` (`UNETLoader`)

 |
| `video_vae_model` | `"minimax_h3_video_vae_fp16.safetensors"` | Node `119` (`VAELoader`)

 |
| `audio_vae_model` | `"minimax_h3_audio_vae_fp32.safetensors"` | Node `120` (`VAELoader`)

 |
| `clip_model` | `"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"` | Node `128` (`CLIPLoader`)

 |
| `lora_model` | `"minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"` | Node `145` (`LoraLoaderModelOnly`)

 |
| `generation_mode` | `"normal"` (`switch = false`) | Node `146` (`PrimitiveBoolean`)

 |
| `steps` | `20` (Normal) / `4` (Turbo) | Nodes `143` & `144` (`PrimitiveInt`)

 |
| `lora_strength` | `1.0` | Node `145` (`strength_model`)

 |
| `aspect_ratio` | `"16:9 (Widescreen)"` | Node `115` (`ResolutionSelector`)

 |
| `megapixels` | `0.4` ($864 \times 480$, `multiple = 32`) | Nodes `115` & `140`<br> |
| `duration_sec` | `5.0` ($124\text{ frames}$) | Nodes `131`, `132` & `136`<br> |
| `fps` | `24` (`bit_depth = 8`) | Node `130` (`CreateVideo`)

 |
| `ref_image_size` | `"match"` | Node `136` (`MiniMaxH3ReferenceToVideo`)

 |
| `sampler_name` | `"res_multistep"` | Node `123` (`KSamplerSelect`)

 |
| `scheduler` | `"simple"` (`denoise = 1.0`) | Node `124` (`BasicScheduler`)

 |
| `controlnet.enabled` | `false` | Disabled by default |






### 3.5. Joint AV Sampling & Dual-VAE Decoding (Nodes `121`–`130`)



* **Sampler**: `res_multistep` (`KSamplerSelect`).


* **Scheduler**: `BasicScheduler` with `scheduler` selectable between `"simple"` (default), `"beta"`, or `"normal"` (recommended for heavy multi-reference prompts), `denoise = 1.0`.


* **Joint Decode**:
* `SamplerCustomAdvanced` outputs a packed audio-video `LATENT`.


* **Video Branch**: `VAEDecode` (strictly non-tiled standard decode; tiled decoding corrupts packed AV latents) using `minimax_h3_video_vae_fp16.safetensors` (FP16) $\rightarrow$ RGB `IMAGE` sequence.


* **Audio Branch**: `VAEDecodeAudio` using `minimax_h3_audio_vae_fp32.safetensors` (strictly **FP32** to prevent phase/timbre distortion) $\rightarrow$ Stereo `AUDIO` waveform.




* **Video Creation & FPS Handling**:
* Native frames (24 FPS) + Stereo Audio are piped to FFmpeg (`CreateVideo`). If the user requests a different output `fps` (e.g., `30` or `60`), apply motion-compensated frame rate conversion (`minterpolate` or `framerate` filter) during muxing while keeping the stereo audio track duration locked.





---

## 4. Memory Management & Multi-GPU ($1 \dots N$) Scaling Architecture

Because `qwen3vl_32b_minimax_h3_nvfp4_awq` (~18 GB) + `minimax_h3_ref2va_pruned_int8_convrot` (~22–28 GB) + Dual VAEs + ControlNet + high-resolution reference tokens can easily exceed single-card VRAM, the worker daemon must implement **Two-Tier Memory Orchestration**:

### 4.1. Tier 1: 3-Stage Pinned-Memory Offloading (Works on 1 to $N$ GPUs)

Pre-allocate a pinned host memory pool (`torch.cuda.MemPool` / `pin_memory=True`) in system RAM so PCIe transfers run at full DMA bandwidth (~25–50 GB/s):

1. **Stage 1 (Multimodal Conditioning & Latent Init)**:
* Load `qwen3vl_32b_minimax_h3_nvfp4_awq` + `video_vae` (FP16) + `audio_vae` (FP32) onto GPU(s).


* Encode prompt and all reference images/videos/audios into `positive` conditioning tensors and initialize the empty joint AV `latent`.


* **Offload** (when the encoders share a GPU with the DiT; see §4.2): unload `qwen3vl_32b`, `video_vae`, and `audio_vae` from that GPU back to host RAM through ComfyUI `model_management.free_memory`, then `torch.cuda.empty_cache()` and `libc.malloc_trim(0)`.


2. **Stage 2 (DiT Joint Audio-Video Denoising)**:
* Stream `minimax_h3_ref2va_pruned_int8_convrot` (+ fused/patched Turbo LoRA + ControlNet if active) from Pinned Host RAM to GPU(s).


* Attention uses ComfyUI's native optimized (exact) attention; no approximate block-sparse override is installed.
* Before sampling, reject with `insufficient_vram` if `model.memory_required(latent_shape)` (activation memory, which cannot be streamed) exceeds the DiT GPU's reachable VRAM.
* Run `res_multistep` sampling steps.


* **Offload**: Move DiT & ControlNet back to Pinned Host RAM; keep only the denoised `LATENT` (~50–200 MB) in VRAM; run `torch.cuda.empty_cache()`.


3. **Stage 3 (Dual-Modality VAE Decode & Zero-Copy Mux)**:
* Reload `video_vae` (FP16) and `audio_vae` (FP32) to GPU(s).


* Decode video frames in temporal micro-batches and decode stereo audio in FP32.
* Stream raw frames and PCM audio over `stdin` pipe to FFmpeg (`-map_metadata -1 -fflags +bitexact`) and flush VRAM.



### 4.2. Tier 2: Multi-GPU ($1 \dots N$) Whole-Model Placement

A **single** worker process (no `torchrun`, no NCCL) owns every visible GPU. Weights stay in their shipped compressed formats (int8 DiT, nvfp4 text encoder); they are never sharded or dequantized. Every checkpoint is loaded exactly once into host RAM, one at a time.

* **Placement planner** (`memory_manager.plan_devices`, run at worker start and after every flush): reads free/total VRAM per GPU from `nvidia-smi` (no CUDA context is created on busy cards). GPUs with free VRAM below `max(MMH3_MIN_FREE_VRAM_GB (default 6), 30 % of total)` are skipped, so cards occupied by other workloads are left alone. Remaining GPUs are sorted by free VRAM:
  * GPU #1 (emptiest): `minimax_h3_ref2va` DiT + Turbo LoRA + ControlNet (Stage 2).
  * GPU #2: `qwen3vl_32b` text encoder + video VAE (FP16) + audio VAE (FP32) (Stages 1 and 3).
* **When only one GPU qualifies**: all models share it; Stage 1 encoders are unloaded before Stage 2 and the DiT is unloaded before Stage 3.
* **When $N \ge 2$**: encoders stay resident on their own GPU between jobs, and the DiT has a GPU to itself.
* Each model is loaded while its target GPU is the current CUDA device, so ComfyUI binds `load_device` to it natively. ComfyUI's `model_management.load_models_gpu` then decides per model whether to load fully or stream weights in `lowvram` mode, using pinned host RAM, and evicts models on that GPU when another needs the space. ComfyUI's VAE decode falls back to tiled decoding on OOM.
* `CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps PyTorch device indices identical to `nvidia-smi`.

### 4.3. Resource Guardrails (checked before the limit is hit)

* **Container**: `mem_limit`/`memswap_limit` (`MMH3_MEM_LIMIT`, default `100g`) so an overrun is contained in the container; tmpfs sizes are fixed (`/tmp` 4G, `/dev/shm` 16G, `/app/ephemeral` 32G). The worker is restarted automatically by the entrypoint if it ever exits.
* **Rust backend**: `upload_media` and `generate_video` check `statvfs` free space on `/app/ephemeral` and cgroup-aware available RAM (`insufficient_storage` / `insufficient_ram`), and reject with `server_busy` when the queue is full.
* **Worker, per model**: before reading a checkpoint, require `file_size × 1.25 + max(6 GB, 8 % RAM)` available RAM (cgroup-aware). If short, evict cached models not needed by the current job, then fail with `insufficient_ram` instead of swapping or being OOM-killed.
* **Worker, per job**: host RAM for decoded frames is checked before Stage 1 and again before Stage 3; activation VRAM is checked before Stage 2 (§4.1); a runtime CUDA OOM is reported as `insufficient_vram`, and memory is trimmed.
* **Progress**: every stage (`preflight_gate`, `load_*`, `stage1_*`, `stage2_denoise` step k/N, `stage3_*`) is pushed over WSS and also exposed through the `job_status` envelope action for polling.







---

## 5. Backend Implementation Blueprint (`backend-rust/`)

### 5.1. `Cargo.toml`

```toml
[package]
name = "minimax-h3-backend"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = { version = "0.8", features = ["multipart", "ws", "macros"] }
tokio = { version = "1.43", features = ["full"] }
tower-http = { version = "0.6", features = ["cors", "fs", "limit"] } # NO trace feature!
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rmp-serde = "1.3" # MessagePack for fast UDS IPC
uuid = { version = "1.12", features = ["v4", "serde"] }
dashmap = "6.1"
nvml-wrapper = "0.10"
zeroize = "1.8"

```

### 5.2. Core Math, Zero-Log & Auto-Flush Engine (`src/main.rs` & `src/storage.rs`)

Instruct the coding agent to implement the following Rust architecture:

```rust
// src/math.rs
pub fn compute_dimensions(aspect_ratio: &str, megapixels: f64, multiple: u32) -> (u32, u32) {
    let (w_r, h_r) = match aspect_ratio {
        "16:9" => (16.0, 9.0),
        "9:16" => (9.0, 16.0),
        "1:1"  => (1.0, 1.0),
        "4:3"  => (4.0, 3.0),
        "3:4"  => (3.0, 4.0),
        "21:9" => (21.0, 9.0),
        _      => (16.0, 9.0),
    };
    let total_pixels = megapixels.clamp(0.1, 2.5) * 1024.0 * 1024.0;
    let m = multiple as f64;
    let width = (((total_pixels * (w_r / h_r)).sqrt() / m).round() * m).max(m) as u32;
    let height = (((total_pixels * (h_r / w_r)).sqrt() / m).round() * m).max(m) as u32;
    (width, height)
}

/// Exact translation of Node 131 ComfyMathExpression:
/// max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
pub fn compute_aligned_frame_length(duration_sec: f64) -> u32 {
    let raw = ((duration_sec * 24.0).round() as i64).max(5);
    let rem = (5 - (raw % 17)).rem_euclid(17);
    (raw + rem) as u32
}

```

```rust
// src/storage.rs
use std::{path::{Path, PathBuf}, time::{Duration, SystemTime}};
use tokio::{fs, io::AsyncWriteExt};

const TTL_SECONDS: u64 = 86_400; // Strictly 1 Day (24 hours)

pub struct EphemeralStorage {
    pub uploads_dir: PathBuf,
    pub outputs_dir: PathBuf,
}

impl EphemeralStorage {
    /// Overwrites file bytes with zeros before unlinking from tmpfs/disk
    pub async fn secure_remove_file(path: &Path) {
        if let Ok(meta) = fs::metadata(path).await {
            if meta.is_file() {
                if let Ok(mut file) = fs::OpenOptions::new().write(true).open(path).await {
                    let zeros = vec![0u8; 65536];
                    let mut remaining = meta.len();
                    while remaining > 0 {
                        let to_write = remaining.min(zeros.len() as u64) as usize;
                        if file.write_all(&zeros[..to_write]).await.is_err() { break; }
                        remaining -= to_write as u64;
                    }
                    let _ = file.sync_all().await;
                }
                let _ = fs::remove_file(path).await;
            }
        }
    }

    /// Background reaper running every 60 seconds to auto-flush files > 24h old
    pub async fn run_ttl_reaper(self: std::sync::Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            let now = SystemTime::now();
            for dir in [&self.uploads_dir, &self.outputs_dir] {
                if let Ok(mut entries) = fs::read_dir(dir).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        if let Ok(meta) = entry.metadata().await {
                            if let Ok(modified) = meta.modified() {
                                if now.duration_since(modified).unwrap_or_default().as_secs() >= TTL_SECONDS {
                                    Self::secure_remove_file(&entry.path()).await;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// One-click manual flush of all uploaded, temporary, and generated media
    pub async fn flush_all_now(&self) {
        for dir in [&self.uploads_dir, &self.outputs_dir] {
            if let Ok(mut entries) = fs::read_dir(dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    Self::secure_remove_file(&entry.path()).await;
                }
            }
        }
    }
}

```

## 5.3. (Updated) Rust Backend API & Payload Schema (`backend-rust/src/api.rs`)

### 1. Model Discovery & Defaults Endpoint (`GET /api/v1/models`)

Scans the 5 model directories without logging filenames to stdout/stderr, prevents path traversal (`..`), and returns both discovered models and the canonical default configuration:

```rust
// backend-rust/src/api.rs
use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;

#[derive(Serialize)]
pub struct ModelCatalogResponse {
    pub available: AvailableModels,
    pub defaults: GenerationConfigDefaults,
    pub gpu_topology: Vec<GpuInfo>,
}

#[derive(Serialize)]
pub struct AvailableModels {
    pub diffussion: Vec<String>, // Matches exact folder name model/diffussion/
    pub vae: Vec<String>,
    pub clip: Vec<String>,
    pub lora: Vec<String>,
    pub controlnet: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GenerationConfigDefaults {
    pub diffusion_model: String,
    pub video_vae_model: String,
    pub audio_vae_model: String,
    pub clip_model: String,
    pub lora_model: String,
    pub generation_mode: String, // "normal" | "turbo"
    pub steps: u32,
    pub normal_default_steps: u32,
    pub turbo_default_steps: u32,
    pub lora_strength: f32,
    pub aspect_ratio: String,
    pub megapixels: f64,
    pub multiple: u32,
    pub duration_sec: f64,
    pub fps: u32,
    pub ref_image_size: String,
    pub sampler_name: String,
    pub scheduler: String,
    pub denoise: f32,
}

impl Default for GenerationConfigDefaults {
    fn default() -> Self {
        Self {
            diffusion_model: "minimax_h3_ref2va_pruned_int8_convrot.safetensors".into(),
            video_vae_model: "minimax_h3_video_vae_fp16.safetensors".into(),
            audio_vae_model: "minimax_h3_audio_vae_fp32.safetensors".into(),
            clip_model: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors".into(),
            lora_model: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors".into(),
            generation_mode: "normal".into(),
            steps: 20,
            normal_default_steps: 20,
            turbo_default_steps: 4,
            lora_strength: 1.0,
            aspect_ratio: "16:9 (Widescreen)".into(),
            megapixels: 0.4,
            multiple: 32,
            duration_sec: 5.0,
            fps: 24,
            ref_image_size: "match".into(),
            sampler_name: "res_multistep".into(),
            scheduler: "simple".into(),
            denoise: 1.0,
        }
    }
}

async fn scan_safetensors(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    scan_dir_recursive(dir, dir, &mut files).await;
    files.sort();
    files
}

async fn scan_dir_recursive(base: &Path, current: &Path, out: &mut Vec<String>) {
    let Ok(mut entries) = fs::read_dir(current).await else { return };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Ok(name) = entry.file_name().into_string() else { continue };
        if name.starts_with('.') || name.contains("..") { continue; }
        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path).await else { continue };
        if meta.file_type().is_symlink() { continue; }
        if meta.is_dir() {
            Box::pin(scan_dir_recursive(base, &path, out)).await;
            continue;
        }
        if meta.is_file() && is_selectable_model_file(&name) {
            if let Ok(rel) = path.strip_prefix(base) {
                let rel_str = rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/");
                if !rel_str.is_empty() { out.push(rel_str); }
            }
        }
    }
}

fn is_selectable_model_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        || lower.ends_with(".sft")
        || lower.ends_with(".ckpt")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".bin")
        || lower.ends_with(".gguf")
        || lower.ends_with(".onnx")
}

```

### 2. Updated `POST /api/v1/generate` Request Schema

```json
{
  "models": {
    "diffusion": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "controlnet": null
  },
  "mode": "turbo",
  "steps": 4,
  "lora_strength": 1.0,
  "prompt": "Bold comic-book ink style... Use <Picture 1> and <Picture 2>...",
  "duration_sec": 5.0,
  "aspect_ratio": "16:9 (Widescreen)",
  "megapixels": 0.4,
  "fps": 24,
  "ref_image_size": "match",
  "sampler_name": "res_multistep",
  "scheduler": "simple",
  "denoise": 1.0,
  "seed": null,
  "controlnet_config": {
    "enabled": false,
    "control_media_id": null,
    "strength": 0.8,
    "start_percent": 0.0,
    "end_percent": 0.85
  },
  "references": {
    "images": ["uuid-img-1", "uuid-img-2"],
    "videos": [],
    "audios": []
  }
}

```

---

## 6. (Updated) Pinned-RAM Model Registry & Hot-Swapping (`inference-worker/memory_manager.py`)

When the user selects a different diffusion, VAE, CLIP, LoRA, or ControlNet file in the UI Model Selector, the GPU worker must safely evict the previously cached weights from Pinned Host RAM before loading the new file so system RAM never leaks or OOMs. The implementation (`ModelRegistry`) follows this sketch and also applies the per-model RAM gate (§4.3) and loads each model under its planned GPU (§4.2):

```python
# inference-worker/memory_manager.py
import os, gc, ctypes, torch

class PinnedModelRegistry:
    def __init__(self, model_root="/app/model"):
        self.model_root = model_root
        self.loaded_slots = {
            "diffusion": {"filename": None, "module": None},
            "video_vae": {"filename": None, "module": None},
            "audio_vae": {"filename": None, "module": None},
            "clip":      {"filename": None, "module": None},
            "lora":      {"filename": None, "weights": None},
            "controlnet":{"filename": None, "module": None},
        }

    def _safe_path(self, subdir: str, filename: str) -> str:
        # Prevent path traversal and enforce exact folder structure
        clean_name = os.path.basename(filename)
        full_path = os.path.join(self.model_root, subdir, clean_name)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Missing model file in {subdir}: {clean_name}")
        return full_path

    def get_or_swap(self, slot: str, subdir: str, filename: str, loader_fn):
        current = self.loaded_slots[slot]
        if current["filename"] != filename or current.get("module") is None:
            # Evict previous model from GPU & Pinned Host RAM
            if current.get("module") is not None:
                del current["module"]
                current["module"] = None
                self.trim_all_memory()
            path = self._safe_path(subdir, filename)
            current["module"] = loader_fn(path)
            current["filename"] = filename
        return current["module"]

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
            pass

```

### Mode & Steps Execution Logic in `pipeline_h3.py`:

```python
# Inside run_job(job_payload):
steps = int(job_payload["steps"]) # Always respects the user's Steps Slider value
base_model = registry.get_or_swap(
    "diffusion", "diffussion", job_payload["models"]["diffusion"], load_unet
)

if job_payload["mode"] == "turbo":
    # Node 141 & 145: Patch with selected Turbo LoRA when in Turbo Mode
    lora_name = job_payload["models"]["lora"]
    lora_strength = float(job_payload.get("lora_strength", 1.0))
    active_model = apply_lora_model_only(base_model, lora_name, lora_strength)
else:
    # Normal Mode: Unpatched base diffusion model
    active_model = base_model

```

---

## 7. (Updated) Frontend UI Components (`frontend/src/components/`)

Provide the coding agent with the following React/TypeScript blueprint implementing the **Model Selector**, **Normal/Turbo Mode Switcher + Steps Slider**, and **Reset to Defaults Button**:

```tsx
// frontend/src/components/GenerationControls.tsx
import React from "react";

export interface GenerationState {
  diffusionModel: string;
  videoVaeModel: string;
  audioVaeModel: string;
  clipModel: string;
  loraModel: string;
  controlnetModel: string;
  mode: "normal" | "turbo";
  steps: number;
  loraStrength: number;
  aspectRatio: string;
  megapixels: number;
  durationSec: number;
  fps: number;
  refImageSize: "match" | "max";
  scheduler: "simple" | "beta" | "normal";
  controlnetEnabled: boolean;
  controlnetStrength: number;
}

export const FACTORY_DEFAULTS: GenerationState = {
  diffusionModel: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
  videoVaeModel: "minimax_h3_video_vae_fp16.safetensors",
  audioVaeModel: "minimax_h3_audio_vae_fp32.safetensors",
  clipModel: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
  loraModel: "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
  controlnetModel: "",
  mode: "normal",
  steps: 20, // Node 143 Int (Full)
  loraStrength: 1.0, // Node 145 strength_model
  aspectRatio: "16:9 (Widescreen)", // Node 115
  megapixels: 0.4, // Node 115 (864x480 @ multiple=32)
  durationSec: 5.0, // Node 132 (124 frames @ 24fps)
  fps: 24, // Node 130
  refImageSize: "match", // Node 136
  scheduler: "simple", // Node 124
  controlnetEnabled: false,
  controlnetStrength: 0.8,
};

export const GenerationControls: React.FC<{
  state: GenerationState;
  setState: React.Dispatch<React.SetStateAction<GenerationState>>;
  availableModels: {
    diffussion: string[];
    vae: string[];
    clip: string[];
    lora: string[];
    controlnet: string[];
  };
  onResetDefaults: () => void;
  onRefreshModels: () => void;
}> = ({ state, setState, availableModels, onResetDefaults, onRefreshModels }) => {
  
  // Switching between Normal and Turbo mode updates the mode and snaps Steps to its mode default (20 or 4),
  // while still letting the user adjust the Steps slider afterward.
  const handleModeChange = (newMode: "normal" | "turbo") => {
    setState((prev) => ({
      ...prev,
      mode: newMode,
      steps: newMode === "turbo" ? 4 : 20,
    }));
  };

  // Live resolution preview (Multiple of 32)
  const computeDimensions = (aspect: string, mp: number) => {
    const [wR, hR] = aspect.startsWith("16:9") ? [16, 9]
      : aspect.startsWith("9:16") ? [9, 16]
      : aspect.startsWith("1:1") ? [1, 1]
      : aspect.startsWith("4:3") ? [4, 3]
      : [21, 9];
    const pixels = mp * 1024 * 1024;
    const w = Math.max(32, Math.round(Math.sqrt(pixels * (wR / hR)) / 32) * 32);
    const h = Math.max(32, Math.round(Math.sqrt(pixels * (hR / wR)) / 32) * 32);
    return `${w} × ${h}`;
  };

  // Live aligned frame count preview: max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
  const computeAlignedFrames = (sec: number) => {
    const raw = Math.max(5, Math.round(sec * 24));
    const rem = ((5 - (raw % 17)) % 17 + 17) % 17;
    return raw + rem;
  };

  const renderModelOptions = (options: string[], current: string, prefix: string) => {
    const merged = current && !options.includes(current) ? [...options, current] : options;
    return merged.map((m) => (
      <option key={`${prefix}-${m}`} value={m}>
        {options.includes(m) ? m : `${m} (not found in folder)`}
      </option>
    ));
  };

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-6 text-zinc-100">
      {/* Header + Live Subfolder Rescan + 1-Click Reset to Defaults Button */}
      <div className="flex flex-wrap items-center justify-between border-b border-zinc-800 pb-3 gap-2">
        <h2 className="text-lg font-semibold tracking-tight">Model & Sampling Controls</h2>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onRefreshModels}
            className="px-2.5 py-1.5 text-xs font-medium rounded-lg bg-zinc-950 hover:bg-zinc-800 text-emerald-400 border border-zinc-800 transition"
            title="Rescan model/diffussion, model/clip, model/vae, model/lora, and model/controlnet subfolders"
          >
            ↻ Rescan Subfolders
          </button>
          <button
            type="button"
            onClick={onResetDefaults}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 transition"
            title="Reset all models, mode, steps, resolution, and duration to factory defaults"
          >
            ↺ Reset to Defaults
          </button>
        </div>
      </div>

      {/* 1. 6-Slot Dynamic Subfolder Model Selector Rack */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`Diffusion Model (model/diffussion/ · ${availableModels.diffussion.length} found)`}
          </label>
          <select
            value={state.diffusionModel}
            onChange={(e) => setState({ ...state, diffusionModel: e.target.value })}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.diffussion, state.diffusionModel, "diff")}
          </select>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`CLIP / Text Encoder (model/clip/ · ${availableModels.clip.length} found)`}
          </label>
          <select
            value={state.clipModel}
            onChange={(e) => setState({ ...state, clipModel: e.target.value })}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.clip, state.clipModel, "clip")}
          </select>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`Video VAE [FP16] (model/vae/ · ${availableModels.vae.length} found)`}
          </label>
          <select
            value={state.videoVaeModel}
            onChange={(e) => setState({ ...state, videoVaeModel: e.target.value })}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.vae, state.videoVaeModel, "vvae")}
          </select>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`Audio VAE [FP32] (model/vae/ · ${availableModels.vae.length} found)`}
          </label>
          <select
            value={state.audioVaeModel}
            onChange={(e) => setState({ ...state, audioVaeModel: e.target.value })}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.vae, state.audioVaeModel, "avae")}
          </select>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`LoRA Model (model/lora/ · ${availableModels.lora.length} found)`}
          </label>
          <select
            value={state.loraModel}
            onChange={(e) => setState({ ...state, loraModel: e.target.value })}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            {renderModelOptions(availableModels.lora, state.loraModel, "lora")}
          </select>
        </div>

        <div>
          <label className="block text-xs text-zinc-400 mb-1">
            {`ControlNet Model (model/controlnet/ · ${availableModels.controlnet.length} found)`}
          </label>
          <select
            value={state.controlnetModel}
            onChange={(e) =>
              setState({
                ...state,
                controlnetModel: e.target.value,
                controlnetEnabled: e.target.value !== "" ? state.controlnetEnabled : false,
              })
            }
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">None (Disabled)</option>
            {availableModels.controlnet.map((m) => (
              <option key={`cn-${m}`} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {/* 2. Normal vs. Turbo Mode Selector + Interactive Steps Slider */}
      <div className="bg-zinc-950/70 border border-zinc-800 rounded-lg p-4 space-y-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">Generation Mode</span>
          <div className="inline-flex rounded-lg bg-zinc-900 p-1 border border-zinc-800">
            <button
              type="button"
              onClick={() => handleModeChange("normal")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                state.mode === "normal"
                  ? "bg-blue-600 text-white"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Normal Mode (Default 20 Steps)
            </button>
            <button
              type="button"
              onClick={() => handleModeChange("turbo")}
              className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                state.mode === "turbo"
                  ? "bg-amber-500 text-zinc-950 font-semibold"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              ⚡ Turbo LoRA Mode (Default 4 Steps)
            </button>
          </div>
        </div>

        {/* Interactive Steps Slider */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">
              Sampling Steps ({state.mode === "turbo" ? "Turbo LoRA Active" : "Full Quality"})
            </span>
            <span className="font-mono font-semibold text-zinc-100">{state.steps} steps</span>
          </div>
          <input
            type="range"
            min={1}
            max={state.mode === "turbo" ? 20 : 50}
            step={1}
            value={state.steps}
            onChange={(e) => setState({ ...state, steps: Number(e.target.value) })}
            className="w-full accent-blue-500 cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-zinc-500 mt-0.5">
            <span>1</span>
            <span>{state.mode === "turbo" ? "Recommended: 4" : "Recommended: 20"}</span>
            <span>{state.mode === "turbo" ? "20" : "50"}</span>
          </div>
        </div>

        {/* Conditional Turbo LoRA Selector & Strength when Turbo Mode is active */}
        {state.mode === "turbo" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-zinc-800/80">
            <div>
              <label className="block text-xs text-amber-400 mb-1">Turbo LoRA File (model/lora/)</label>
              <select
                value={state.loraModel}
                onChange={(e) => setState({ ...state, loraModel: e.target.value })}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs"
              >
                {availableModels.lora.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-amber-400">LoRA Strength</span>
                <span className="font-mono">{state.loraStrength.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0.1}
                max={1.5}
                step={0.05}
                value={state.loraStrength}
                onChange={(e) => setState({ ...state, loraStrength: Number(e.target.value) })}
                className="w-full accent-amber-500"
              />
            </div>
          </div>
        )}
      </div>

      {/* 3. Resolution, Aspect Ratio, Duration, and FPS Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">Megapixels & Output Size</span>
            <span className="font-mono text-emerald-400">
              {state.megapixels.toFixed(1)} MP → {computeDimensions(state.aspectRatio, state.megapixels)}
            </span>
          </div>
          <input
            type="range"
            min={0.2}
            max={1.0}
            step={0.1}
            value={state.megapixels}
            onChange={(e) => setState({ ...state, megapixels: Number(e.target.value) })}
            className="w-full accent-emerald-500"
          />
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-zinc-400">Duration (24fps Causal Alignment)</span>
            <span className="font-mono text-emerald-400">
              {state.durationSec.toFixed(1)}s → {computeAlignedFrames(state.durationSec)} frames
            </span>
          </div>
          <input
            type="range"
            min={1.0}
            max={15.0}
            step={0.5}
            value={state.durationSec}
            onChange={(e) => setState({ ...state, durationSec: Number(e.target.value) })}
            className="w-full accent-emerald-500"
          />
        </div>
      </div>
    </div>
  );
};

```

### Top-Level Reset Handler (`frontend/src/App.tsx`)

When the user clicks **`↺ Reset to Defaults`**, the UI resets both the `GenerationState` (`FACTORY_DEFAULTS`) and restores the default sample prompt from Node `138` (`Input Text (Prompt)`), without wiping uploaded reference files (which remains isolated to the red **`🔥 Flush All Media Now`** button):

```tsx
const handleResetToDefaults = () => {
  setGenerationState({ ...FACTORY_DEFAULTS });
  setScheduler("simple");
  setSeed(null);
};

```



---

## 8. Isolated, Zero-Log Docker Deployment (`Dockerfile` & `docker-compose.yml`)

### 8.1. `docker-compose.yml`

```yaml
services:
  minimax-h3-app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: minimax_h3_ref2v
    # 1. STRICT NON-LOGGING MODE AT CONTAINER LEVEL
    logging:
      driver: "none"
    # 2. ALL 1..N GPUS VISIBLE; WORKER PLACES WHOLE MODELS PER GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Hard RAM ceiling so overruns are contained in the container, never the host
    mem_limit: ${MMH3_MEM_LIMIT:-100g}
    memswap_limit: ${MMH3_MEM_LIMIT:-100g}
    ulimits:
      memlock: -1
      stack: 67108864
    # 3. READ-ONLY ROOT FILESYSTEM + RAM-BACKED EPHEMERAL STORAGE
    read_only: true
    tmpfs:
      - /tmp:size=4G,mode=1777
      - /dev/shm:size=16G,mode=1777
      - /app/ephemeral:size=32G,mode=0700
    volumes:
      # Exact user folder structure mounted Read-Only
      - ./model/diffussion:/app/model/diffussion:ro
      - ./model/lora:/app/model/lora:ro
      - ./model/controlnet:/app/model/controlnet:ro
      - ./model/clip:/app/model/clip:ro
      - ./model/vae:/app/model/vae:ro
    environment:
      - RUST_LOG=off
      - PYTHONIOENCODING=utf-8
      - PYTHONDONTWRITEBYTECODE=1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
      - EPHEMERAL_DIR=/app/ephemeral
      - TTL_SECONDS=86400
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - MMH3_MIN_FREE_VRAM_GB=6
    ports:
      - "8443:8443"
    restart: unless-stopped

```

### 8.2. Multi-Stage `Dockerfile`

```dockerfile
# Stage 1: Build Frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Build Rust Backend
FROM rust:1.99-slim-bookworm AS rust-builder
WORKDIR /backend
COPY backend-rust/Cargo.* ./
COPY backend-rust/src ./src
RUN cargo build --release --locked

# Stage 3: CUDA 13.0 Runtime + PyTorch cu130 + Rust Binary
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /dev/null /var/log/dpkg.log \
    && ln -sf /dev/null /var/log/alternatives.log

# Install PyTorch with cu130 (Required for minimax_h3 int8_convrot & nvfp4_awq)
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 \
    safetensors einops transformers accelerate msgpack pyyaml scipy

WORKDIR /app
COPY --from=rust-builder /backend/target/release/minimax-h3-backend /app/minimax-h3-backend
COPY --from=frontend-builder /ui/dist /app/static
COPY inference-worker/ /app/inference-worker/

# Entrypoint launches one self-restarting GPU worker (placing models across 1..N GPUs) + Rust Axum server, stdout/stderr routed to /dev/null
ENTRYPOINT ["/bin/sh", "-c", "(while true; do python3 /app/inference-worker/worker_daemon.py >/dev/null 2>&1; sleep 3; done) & exec /app/minimax-h3-backend >/dev/null 2>&1"]

```

---

## 9. Step-by-Step Execution Prompt for Your Coding Agent

Copy and paste the following directive when starting your coding agent session:

> **"Read `AGENT_BLUEPRINT.md` and implement the complete project in the current workspace. Execute in 4 sequential phases, running `cargo check` and `npm run build` to verify each phase before proceeding:**
> 1. **Phase 1 (Rust Backend)**: Scaffold `backend-rust/` with `axum`, implementing `math.rs` (exact `multiple=32` resolution and `17k+5` frame alignment from `46a303cbccf9.json`), `storage.rs` (zero-overwrite file deletion, 24-hour background reaper task, and the `flush_all` envelope action), `ipc.rs` (MessagePack over `/dev/shm/minimax_h3.sock`), and zero-logging HTTP/WS routes.
> 
> 
> 2. **Phase 2 (Multi-GPU Worker)**: Implement `inference-worker/` with `worker_daemon.py`, `memory_manager.py`, and `pipeline_h3.py`. Map the exact folder names (`model/diffussion/`, `model/lora/`, `model/controlnet/`, `model/clip/`, `model/vae/`), implement the 3-stage offloader on top of ComfyUI model management, $1 \dots N$ GPU whole-model placement with resource guardrails (§4.2–4.3), Turbo LoRA 4-step switch (`minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`), ControlNet union patching, and bitexact FFmpeg muxing of `VAEDecode` (FP16) + `VAEDecodeAudio` (FP32).
> 
> 
> 3. **Phase 3 (Frontend UI)**: Build the React + Tailwind studio in `frontend/` with the 9-Image / 3-Video / 3-Audio Reference Rack, `<Picture N>` / `<Video N>` / `<Audio N>` tag-inserting prompt editor, Turbo LoRA & ControlNet controls, and the 1-Click Flush Media button.
> 
> 
> 4. **Phase 4 (Docker Isolation)**: Generate the `Dockerfile` and `docker-compose.yml` enforcing `logging: driver: "none"`, `read_only: true`, and `tmpfs` mounts."
> 
>