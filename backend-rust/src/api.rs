use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use axum::body::Bytes;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, Semaphore};
use uuid::Uuid;
use zeroize::Zeroize;

use crate::crypto::{self, SessionKey};
use crate::ipc::{IpcClient, ProgressEvent};
use crate::math;
use crate::storage::{EphemeralStorage, TTL_SECONDS};

#[derive(Clone, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mb: u64,
}

#[derive(Serialize)]
pub struct ModelCatalogResponse {
    pub available: AvailableModels,
    pub defaults: GenerationConfigDefaults,
    pub gpu_topology: Vec<GpuInfo>,
}

#[derive(Serialize)]
pub struct AvailableModels {
    pub diffussion: Vec<String>,
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
    pub generation_mode: String,
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

#[derive(Clone)]
pub struct AppState {
    pub storage: Arc<EphemeralStorage>,
    pub sessions: Arc<DashMap<Uuid, SessionRecord>>,
    pub media: Arc<DashMap<Uuid, MediaRecord>>,
    pub jobs: Arc<DashMap<Uuid, JobRecord>>,
    pub progress: broadcast::Sender<ProgressEvent>,
    pub model_root: PathBuf,
    pub ipc: Arc<IpcClient>,
    pub job_slots: Arc<Semaphore>,
}

pub struct SessionRecord {
    pub key: SessionKey,
    pub created: Instant,
}

pub struct MediaRecord {
    pub path: PathBuf,
    pub kind: String,
    pub created: SystemTime,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Error,
    Aborted,
}

pub struct JobRecord {
    pub status: JobStatus,
    pub stage: String,
    pub progress: f32,
    pub detail: String,
    pub output: PathBuf,
    pub error: Option<String>,
    pub created: Instant,
}

#[derive(Deserialize)]
struct HandshakeRequest {
    client_public_key: String,
}

#[derive(Serialize)]
struct HandshakeResponse {
    session_id: Uuid,
    server_public_key: String,
    alg: &'static str,
}

#[derive(Deserialize)]
struct WsQuery {
    sid: String,
}

#[derive(Serialize, Deserialize)]
struct EnvelopeFile {
    name: String,
    mime: String,
    kind: String,
    #[serde(with = "serde_bytes")]
    data: Vec<u8>,
}

#[derive(Deserialize)]
struct EnvelopeIn {
    action: String,
    #[serde(default)]
    headers: Value,
    #[serde(default)]
    metadata: Value,
    #[serde(default)]
    files: Vec<EnvelopeFile>,
}

#[derive(Serialize)]
struct EnvelopeOut {
    ok: bool,
    action: String,
    metadata: Value,
    files: Vec<EnvelopeFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub fn router(state: AppState, static_dir: PathBuf) -> Router {
    let api = Router::new()
        .route(
            "/api/v1/crypto/handshake",
            post(handshake).layer(DefaultBodyLimit::max(4096)),
        )
        .route("/api/v1/envelope", post(envelope))
        .route("/api/v1/ws", get(ws_upgrade))
        .with_state(state);
    let spa = tower_http::services::ServeDir::new(static_dir).append_index_html_on_directories(true);
    api.fallback_service(spa)
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024))
        .layer(axum::middleware::from_fn(security_headers))
}

async fn security_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    insert_static(headers, header::CACHE_CONTROL, "no-store");
    insert_static(headers, header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    insert_static(headers, header::REFERRER_POLICY, "no-referrer");
    response
}

fn insert_static(headers: &mut HeaderMap, name: header::HeaderName, value: &'static str) {
    headers.insert(name, header::HeaderValue::from_static(value));
}

async fn handshake(
    State(state): State<AppState>,
    Json(req): Json<HandshakeRequest>,
) -> Response {
    let client_public = match crypto::decode_public_key(&req.client_public_key) {
        Ok(key) => key,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let (server_public, session_key) = match crypto::handshake(&client_public) {
        Ok(pair) => pair,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let session_id = Uuid::new_v4();
    state.sessions.insert(
        session_id,
        SessionRecord {
            key: session_key,
            created: Instant::now(),
        },
    );
    Json(HandshakeResponse {
        session_id,
        server_public_key: crypto::encode_public_key(&server_public),
        alg: crypto::SESSION_ALG,
    })
    .into_response()
}

async fn envelope(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let Some(session_id) = session_from_headers(&headers) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let Some(session) = state.sessions.get(&session_id) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let mut opened = match crypto::open(session.key.as_bytes(), &body) {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };
    let mut key_bytes = *session.key.as_bytes();
    drop(session);
    let key = SessionKey::new(key_bytes);
    key_bytes.zeroize();

    let mut padded = match crypto::unpad_block(&opened) {
        Ok(bytes) => bytes,
        Err(_) => {
            opened.zeroize();
            return sealed_error(&key, "envelope", "bad_envelope");
        }
    };
    opened.zeroize();
    let envelope: EnvelopeIn = match rmp_serde::from_slice(&padded) {
        Ok(value) => value,
        Err(_) => {
            padded.zeroize();
            return sealed_error(&key, "envelope", "bad_envelope");
        }
    };
    padded.zeroize();

    let action = envelope.action.clone();
    let result = dispatch(&state, &session_id, envelope).await;
    let response = match result {
        Ok(out) => encode_envelope(&key, out),
        Err(code) => sealed_error(&key, &action, &code),
    };
    if action == "flush_all" {
        state.sessions.clear();
    }
    let _ = key;
    response
}

async fn dispatch(
    state: &AppState,
    session_id: &Uuid,
    mut envelope: EnvelopeIn,
) -> Result<EnvelopeOut, String> {
    let _ = (session_id, &envelope.headers);
    match envelope.action.as_str() {
        "list_models" => Ok(list_models(state).await),
        "upload_media" => upload_media(state, &envelope.metadata, &mut envelope.files).await,
        "generate_video" => generate_video(state, &envelope.metadata).await,
        "job_status" => job_status(state, &envelope.metadata).await,
        "download_output" => download_output(state, &envelope.metadata).await,
        "flush_all" => flush_all(state).await,
        _ => Err("unknown_action".into()),
    }
}

async fn list_models(state: &AppState) -> EnvelopeOut {
    let available = AvailableModels {
        diffussion: scan_safetensors(&state.model_root.join("diffussion")).await,
        vae: scan_safetensors(&state.model_root.join("vae")).await,
        clip: scan_safetensors(&state.model_root.join("clip")).await,
        lora: scan_safetensors(&state.model_root.join("lora")).await,
        controlnet: scan_safetensors(&state.model_root.join("controlnet")).await,
    };
    let body = ModelCatalogResponse {
        available,
        defaults: GenerationConfigDefaults::default(),
        gpu_topology: detect_gpus(),
    };
    EnvelopeOut {
        ok: true,
        action: "list_models".into(),
        metadata: serde_json::to_value(body).unwrap_or_else(|_| json!({})),
        files: Vec::new(),
        error: None,
    }
}

async fn scan_safetensors(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    scan_dir_recursive(dir, dir, &mut files).await;
    files.sort();
    files
}

async fn scan_dir_recursive(base: &Path, current: &Path, out: &mut Vec<String>) {
    let Ok(mut entries) = fs::read_dir(current).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if name.starts_with('.') || name.contains("..") {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path).await else {
            continue;
        };
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            Box::pin(scan_dir_recursive(base, &path, out)).await;
            continue;
        }
        if !meta.is_file() || !is_selectable_model_file(&name) {
            continue;
        }
        if let Ok(rel) = path.strip_prefix(base) {
            let rel_str = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            if !rel_str.is_empty() {
                out.push(rel_str);
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

pub fn detect_gpus() -> Vec<GpuInfo> {
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            let mut gpus = Vec::new();
            for index in 0..count {
                let Ok(device) = nvml.device_by_index(index) else {
                    continue;
                };
                let name = device.name().unwrap_or_else(|_| "gpu".to_string());
                let memory_total_mb = device
                    .memory_info()
                    .map(|info| info.total / (1024 * 1024))
                    .unwrap_or(0);
                gpus.push(GpuInfo {
                    index,
                    name,
                    memory_total_mb,
                });
            }
            if !gpus.is_empty() {
                return gpus;
            }
        }
    }
    fallback_gpus()
}

fn fallback_gpus() -> Vec<GpuInfo> {
    let Ok(visible) = std::env::var("NVIDIA_VISIBLE_DEVICES") else {
        return Vec::new();
    };
    if visible.is_empty() || visible == "void" || visible == "none" || visible == "NoDevFiles" {
        return Vec::new();
    }
    if visible == "all" {
        return vec![GpuInfo {
            index: 0,
            name: "all".into(),
            memory_total_mb: 0,
        }];
    }
    visible
        .split(',')
        .enumerate()
        .map(|(index, id)| GpuInfo {
            index: index as u32,
            name: format!("gpu:{id}"),
            memory_total_mb: 0,
        })
        .collect()
}

async fn upload_media(
    state: &AppState,
    metadata: &Value,
    files: &mut [EnvelopeFile],
) -> Result<EnvelopeOut, String> {
    let kind = metadata
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("image");
    if !matches!(kind, "image" | "video" | "audio" | "control") {
        return Err("bad_media".into());
    }
    let file = files.first_mut().ok_or_else(|| "bad_media".to_string())?;
    if file.data.is_empty() {
        return Err("bad_media".into());
    }
    let upload_len = file.data.len() as u64;
    check_system_gates(
        &state.storage.root,
        upload_len + 64 * 1024 * 1024,
        upload_len + 2 * 1024 * 1024 * 1024,
    )?;
    let filename = metadata
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or(file.name.as_str());
    let ext = safe_extension(filename);
    let id = Uuid::new_v4();
    let path = state.storage.uploads_dir.join(format!("{id}.{ext}"));
    {
        let mut handle = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .await
            .map_err(|_| "store_failed".to_string())?;
        handle
            .write_all(&file.data)
            .await
            .map_err(|_| "store_failed".to_string())?;
        handle
            .sync_all()
            .await
            .map_err(|_| "store_failed".to_string())?;
    }
    let nbytes = file.data.len();
    file.data.zeroize();
    state.media.insert(
        id,
        MediaRecord {
            path,
            kind: kind.to_string(),
            created: SystemTime::now(),
        },
    );
    Ok(EnvelopeOut {
        ok: true,
        action: "upload_media".into(),
        metadata: json!({
            "media_id": id.to_string(),
            "kind": kind,
            "bytes": nbytes,
        }),
        files: Vec::new(),
        error: None,
    })
}

fn safe_extension(filename: &str) -> &'static str {
    let name = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
    let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "png" => "png",
        "jpg" | "jpeg" => "jpg",
        "webp" => "webp",
        "gif" => "gif",
        "mp4" => "mp4",
        "webm" => "webm",
        "mov" => "mov",
        "mkv" => "mkv",
        "wav" => "wav",
        "mp3" => "mp3",
        "flac" => "flac",
        "ogg" => "ogg",
        "m4a" => "m4a",
        _ => "bin",
    }
}

#[derive(Deserialize)]
struct GenerateRequest {
    models: ModelSelection,
    mode: String,
    steps: u32,
    #[serde(default = "default_lora_strength")]
    lora_strength: f32,
    prompt: String,
    duration_sec: f64,
    aspect_ratio: String,
    megapixels: f64,
    #[serde(default = "default_fps")]
    fps: u32,
    #[serde(default = "default_ref_size")]
    ref_image_size: String,
    #[serde(default = "default_sampler")]
    sampler_name: String,
    #[serde(default = "default_scheduler")]
    scheduler: String,
    #[serde(default = "default_denoise")]
    denoise: f32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    controlnet_config: ControlnetConfig,
    references: ReferenceSet,
}

fn default_lora_strength() -> f32 {
    1.0
}
fn default_fps() -> u32 {
    24
}
fn default_ref_size() -> String {
    "match".into()
}
fn default_sampler() -> String {
    "res_multistep".into()
}
fn default_scheduler() -> String {
    "simple".into()
}
fn default_denoise() -> f32 {
    1.0
}

#[derive(Deserialize)]
struct ModelSelection {
    diffusion: String,
    video_vae: String,
    audio_vae: String,
    clip: String,
    lora: String,
    #[serde(default)]
    controlnet: Option<String>,
}

#[derive(Deserialize)]
struct ControlnetConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    control_media_id: Option<String>,
    #[serde(default = "default_cn_strength")]
    strength: f32,
    #[serde(default)]
    start_percent: f32,
    #[serde(default = "default_cn_end")]
    end_percent: f32,
}

impl Default for ControlnetConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            control_media_id: None,
            strength: 0.8,
            start_percent: 0.0,
            end_percent: 0.85,
        }
    }
}

fn default_cn_strength() -> f32 {
    0.8
}
fn default_cn_end() -> f32 {
    0.85
}

#[derive(Deserialize)]
struct ReferenceSet {
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    videos: Vec<String>,
    #[serde(default)]
    video_audios: Vec<Option<String>>,
    #[serde(default)]
    audios: Vec<String>,
}

async fn generate_video(state: &AppState, metadata: &Value) -> Result<EnvelopeOut, String> {
    let req: GenerateRequest =
        serde_json::from_value(metadata.clone()).map_err(|_| "bad_request".to_string())?;
    if req.prompt.trim().is_empty() || req.prompt.len() > 16_000 {
        return Err("bad_request".into());
    }
    if req.mode != "normal" && req.mode != "turbo" {
        return Err("bad_request".into());
    }
    let max_steps = if req.mode == "turbo" { 20 } else { 50 };
    if req.steps < 1 || req.steps > max_steps {
        return Err("bad_request".into());
    }
    if !(1.0..=15.0).contains(&req.duration_sec) || !(0.2..=2.0).contains(&req.megapixels) {
        return Err("bad_request".into());
    }
    if !(1..=60).contains(&req.fps) || !(0.0..=1.0).contains(&req.denoise) {
        return Err("bad_request".into());
    }
    if !(0.0..=2.0).contains(&req.lora_strength) {
        return Err("bad_request".into());
    }
    if req.ref_image_size != "match" && req.ref_image_size != "max" {
        return Err("bad_request".into());
    }
    if req.sampler_name != "res_multistep" {
        return Err("bad_request".into());
    }
    if !matches!(req.scheduler.as_str(), "simple" | "beta" | "normal") {
        return Err("bad_request".into());
    }
    if !math::known_aspect(&req.aspect_ratio) {
        return Err("bad_request".into());
    }
    if req.references.images.len() > 9
        || req.references.videos.len() > 3
        || req.references.audios.len() > 3
        || req.references.video_audios.len() > 3
    {
        return Err("bad_request".into());
    }
    if req.controlnet_config.enabled {
        if req.controlnet_config.control_media_id.is_none() {
            return Err("bad_request".into());
        }
        if !(0.0..=2.0).contains(&req.controlnet_config.strength) {
            return Err("bad_request".into());
        }
        if !(0.0..=1.0).contains(&req.controlnet_config.start_percent)
            || !(0.0..=1.0).contains(&req.controlnet_config.end_percent)
            || req.controlnet_config.start_percent >= req.controlnet_config.end_percent
        {
            return Err("bad_request".into());
        }
    }

    ensure_model_file(&state.model_root, "diffussion", &req.models.diffusion)?;
    ensure_model_file(&state.model_root, "vae", &req.models.video_vae)?;
    ensure_model_file(&state.model_root, "vae", &req.models.audio_vae)?;
    ensure_model_file(&state.model_root, "clip", &req.models.clip)?;
    if req.mode == "turbo" {
        ensure_model_file(&state.model_root, "lora", &req.models.lora)?;
    }
    if req.controlnet_config.enabled {
        let name = req
            .models
            .controlnet
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "model_missing".to_string())?;
        ensure_model_file(&state.model_root, "controlnet", name)?;
    }

    let images = resolve_media_list(state, &req.references.images, "image")?;
    let videos = resolve_media_list(state, &req.references.videos, "video")?;
    let audios = resolve_media_list(state, &req.references.audios, "audio")?;
    let mut video_audios = Vec::new();
    for (idx, _) in videos.iter().enumerate() {
        let id = req.references.video_audios.get(idx).and_then(|v| v.clone());
        if let Some(id) = id {
            if id.is_empty() {
                video_audios.push(Value::Null);
            } else {
                let path = resolve_media(state, &id, "audio")?;
                video_audios.push(Value::String(path));
            }
        } else {
            video_audios.push(Value::Null);
        }
    }
    let control_path = if req.controlnet_config.enabled {
        let id = req
            .controlnet_config
            .control_media_id
            .clone()
            .ok_or_else(|| "bad_request".to_string())?;
        Some(resolve_media_any(state, &id, &["image", "video", "control"])?)
    } else {
        None
    };

    let (width, height) = math::compute_dimensions(&req.aspect_ratio, req.megapixels, 32);
    let frames = math::compute_aligned_frame_length(req.duration_sec);
    let est_ram = (width as u64)
        .saturating_mul(height as u64)
        .saturating_mul(frames as u64)
        .saturating_mul(16)
        .max(2 * 1024 * 1024 * 1024);
    let est_output = (width as u64)
        .saturating_mul(height as u64)
        .saturating_mul(frames as u64)
        / 16;
    check_system_gates(&state.storage.root, est_output.max(512 * 1024 * 1024), est_ram)?;
    if state.job_slots.available_permits() == 0 {
        let active_running = state
            .jobs
            .iter()
            .filter(|entry| matches!(entry.status, JobStatus::Queued | JobStatus::Running))
            .count();
        if active_running >= 2 {
            return Err("server_busy".into());
        }
    }
    let seed = req.seed.unwrap_or_else(random_seed);
    let job_id = Uuid::new_v4();
    let output = state.storage.outputs_dir.join(format!("{job_id}.mp4"));
    state.jobs.insert(
        job_id,
        JobRecord {
            status: JobStatus::Queued,
            stage: "queued".into(),
            progress: 0.01,
            detail: format!("Queued ({width}x{height} · {frames} frames)"),
            output: output.clone(),
            error: None,
            created: Instant::now(),
        },
    );

    let payload = json!({
        "job_id": job_id.to_string(),
        "models": {
            "diffusion": basename(&req.models.diffusion),
            "video_vae": basename(&req.models.video_vae),
            "audio_vae": basename(&req.models.audio_vae),
            "clip": basename(&req.models.clip),
            "lora": basename(&req.models.lora),
            "controlnet": req.models.controlnet.as_ref().map(|s| basename(s)),
        },
        "mode": req.mode,
        "steps": req.steps,
        "lora_strength": req.lora_strength,
        "prompt": req.prompt,
        "duration_sec": req.duration_sec,
        "aspect_ratio": req.aspect_ratio,
        "megapixels": req.megapixels,
        "fps": req.fps,
        "ref_image_size": req.ref_image_size,
        "sampler_name": req.sampler_name,
        "scheduler": req.scheduler,
        "denoise": req.denoise,
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "output_path": output.to_string_lossy(),
        "model_root": state.model_root.to_string_lossy(),
        "controlnet_config": {
            "enabled": req.controlnet_config.enabled,
            "strength": req.controlnet_config.strength,
            "start_percent": req.controlnet_config.start_percent,
            "end_percent": req.controlnet_config.end_percent,
            "path": control_path,
        },
        "references": {
            "images": images,
            "videos": videos,
            "video_audios": video_audios,
            "audios": audios,
        }
    });

    let progress = state.progress.clone();
    let jobs = state.jobs.clone();
    let ipc = state.ipc.clone();
    let slots = state.job_slots.clone();
    let job_key = job_id;
    tokio::spawn(async move {
        let Ok(_permit) = slots.acquire().await else {
            return;
        };
        {
            let Some(mut job) = jobs.get_mut(&job_key) else {
                return;
            };
            if job.status == JobStatus::Aborted {
                return;
            }
            job.status = JobStatus::Running;
            job.stage = "starting".into();
            job.progress = 0.02;
            job.detail = "Connecting to multi-GPU inference worker".into();
        }
        let _ = progress.send(ProgressEvent {
            job_id: job_key.to_string(),
            stage: "starting".into(),
            progress: 0.02,
            detail: "Connecting to multi-GPU inference worker".into(),
            done: false,
            ok: true,
        });
        let tx = progress.clone();
        let jobs_for_cb = jobs.clone();
        let outcome = ipc
            .generate(&job_key.to_string(), payload, move |event| {
                if let Some(mut job) = jobs_for_cb.get_mut(&job_key) {
                    job.stage = event.stage.clone();
                    job.progress = event.progress;
                    job.detail = event.detail.clone();
                }
                let _ = tx.send(event);
            })
            .await;
        match outcome {
            Ok(_) => {
                if let Some(mut job) = jobs.get_mut(&job_key) {
                    if job.status != JobStatus::Aborted {
                        job.status = JobStatus::Done;
                        job.stage = "done".into();
                        job.progress = 1.0;
                        job.detail = "Video & stereo audio muxed".into();
                        job.error = None;
                    }
                }
                let _ = progress.send(ProgressEvent {
                    job_id: job_key.to_string(),
                    stage: "done".into(),
                    progress: 1.0,
                    detail: "ready".into(),
                    done: true,
                    ok: true,
                });
            }
            Err(code) => {
                let aborted = code == "aborted";
                if let Some(mut job) = jobs.get_mut(&job_key) {
                    job.status = if aborted {
                        JobStatus::Aborted
                    } else {
                        JobStatus::Error
                    };
                    job.stage = "done".into();
                    job.progress = 1.0;
                    job.detail = code.clone();
                    job.error = Some(code.clone());
                }
                let _ = progress.send(ProgressEvent {
                    job_id: job_key.to_string(),
                    stage: "done".into(),
                    progress: 1.0,
                    detail: code,
                    done: true,
                    ok: false,
                });
            }
        }
    });

    Ok(EnvelopeOut {
        ok: true,
        action: "generate_video".into(),
        metadata: json!({
            "job_id": job_id.to_string(),
            "width": width,
            "height": height,
            "frames": frames,
            "seed": seed,
            "status": "queued",
        }),
        files: Vec::new(),
        error: None,
    })
}

fn random_seed() -> u64 {
    use rand::rngs::OsRng;
    use rand::RngCore;
    let mut buf = [0u8; 8];
    OsRng.fill_bytes(&mut buf);
    u64::from_le_bytes(buf)
}

fn basename(name: &str) -> String {
    let normalized = name.trim().replace('\\', "/");
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|part| !part.is_empty() && *part != "." && *part != "..")
        .collect();
    parts.join("/")
}

fn ensure_model_file(root: &Path, subdir: &str, name: &str) -> Result<(), String> {
    let clean = basename(name);
    if clean.is_empty()
        || clean.contains("..")
        || clean.split('/').any(|seg| seg.starts_with('.'))
        || !is_selectable_model_file(&clean)
    {
        return Err("model_missing".into());
    }
    let dir = root.join(subdir);
    let path = dir.join(&clean);
    let canon_dir = std::fs::canonicalize(&dir).map_err(|_| "model_missing".to_string())?;
    let canon = std::fs::canonicalize(&path).map_err(|_| "model_missing".to_string())?;
    if canon.starts_with(&canon_dir) && canon.is_file() {
        Ok(())
    } else {
        Err("model_missing".into())
    }
}

fn resolve_media_list(state: &AppState, ids: &[String], kind: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        out.push(resolve_media(state, id, kind)?);
    }
    Ok(out)
}

fn resolve_media(state: &AppState, id: &str, kind: &str) -> Result<String, String> {
    resolve_media_any(state, id, &[kind])
}

fn resolve_media_any(state: &AppState, id: &str, kinds: &[&str]) -> Result<String, String> {
    let uuid = Uuid::parse_str(id).map_err(|_| "bad_media".to_string())?;
    let record = state.media.get(&uuid).ok_or_else(|| "bad_media".to_string())?;
    if !kinds.iter().any(|kind| record.kind == *kind) {
        return Err("bad_media".into());
    }
    let root = std::fs::canonicalize(&state.storage.root).map_err(|_| "bad_media".to_string())?;
    let uploads = std::fs::canonicalize(&state.storage.uploads_dir)
        .map_err(|_| "bad_media".to_string())?;
    let path = std::fs::canonicalize(&record.path).map_err(|_| "bad_media".to_string())?;
    if !path.starts_with(&root) || !path.starts_with(&uploads) || !path.is_file() {
        return Err("bad_media".into());
    }
    Ok(path.to_string_lossy().into_owned())
}

async fn job_status(state: &AppState, metadata: &Value) -> Result<EnvelopeOut, String> {
    let job_id = metadata
        .get("job_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "bad_request".to_string())?;
    let uuid = Uuid::parse_str(job_id).map_err(|_| "bad_request".to_string())?;
    let record = state
        .jobs
        .get(&uuid)
        .ok_or_else(|| "not_found".to_string())?;
    let (status_str, done, ok) = match record.status {
        JobStatus::Queued => ("queued", false, true),
        JobStatus::Running => ("running", false, true),
        JobStatus::Done => ("done", true, true),
        JobStatus::Error => ("error", true, false),
        JobStatus::Aborted => ("aborted", true, false),
    };
    Ok(EnvelopeOut {
        ok: true,
        action: "job_status".into(),
        metadata: json!({
            "job_id": job_id,
            "status": status_str,
            "stage": record.stage,
            "progress": record.progress,
            "detail": record.detail,
            "done": done,
            "ok": ok,
            "error": record.error,
        }),
        files: Vec::new(),
        error: None,
    })
}

fn check_system_gates(storage_root: &Path, min_storage_bytes: u64, min_ram_bytes: u64) -> Result<(), String> {
    if let Some(avail_ram) = host_available_ram_bytes() {
        if avail_ram < min_ram_bytes {
            return Err("insufficient_ram".into());
        }
    }
    match storage_free_bytes(storage_root) {
        Some(free) if free >= min_storage_bytes => Ok(()),
        _ => Err("insufficient_storage".into()),
    }
}

fn storage_free_bytes(root: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(root.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `path` is a valid NUL-terminated string and `stat` is a writable statvfs.
    if unsafe { libc::statvfs(path.as_ptr(), &mut stat) } != 0 {
        return None;
    }
    Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
}

/// MemAvailable, capped by the container's cgroup-v2 limit (tmpfs uploads are charged there too).
fn host_available_ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let host = text.lines().find_map(|line| {
        let rest = line.strip_prefix("MemAvailable:")?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    let host = host.saturating_mul(1024);
    Some(cgroup_available_ram_bytes().map_or(host, |cg| cg.min(host)))
}

fn cgroup_available_ram_bytes() -> Option<u64> {
    let read = |name: &str| std::fs::read_to_string(format!("/sys/fs/cgroup/{name}")).ok();
    let limit: u64 = read("memory.max")?.trim().parse().ok()?;
    let current: u64 = read("memory.current")?.trim().parse().ok()?;
    let reclaimable: u64 = read("memory.stat")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.split_once(' '))
        .filter(|(key, _)| matches!(*key, "active_file" | "inactive_file"))
        .filter_map(|(_, value)| value.trim().parse::<u64>().ok())
        .sum();
    Some(limit.saturating_sub(current.saturating_sub(reclaimable)))
}

async fn download_output(state: &AppState, metadata: &Value) -> Result<EnvelopeOut, String> {
    let job_id = metadata
        .get("job_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "bad_request".to_string())?;
    let uuid = Uuid::parse_str(job_id).map_err(|_| "bad_request".to_string())?;
    let record = state
        .jobs
        .get(&uuid)
        .ok_or_else(|| "not_found".to_string())?;
    match record.status {
        JobStatus::Done => {}
        JobStatus::Error => {
            let code = record.error.clone().unwrap_or_else(|| "inference_failed".into());
            return Err(code);
        }
        JobStatus::Aborted => return Err("aborted".into()),
        JobStatus::Queued | JobStatus::Running => return Err("not_ready".into()),
    }
    let output = record.output.clone();
    drop(record);
    let outputs = std::fs::canonicalize(&state.storage.outputs_dir)
        .map_err(|_| "not_found".to_string())?;
    let path = std::fs::canonicalize(&output).map_err(|_| "not_found".to_string())?;
    if !path.starts_with(&outputs) {
        return Err("not_found".into());
    }
    let mut bytes = fs::read(&path).await.map_err(|_| "not_found".to_string())?;
    let nbytes = bytes.len();
    let files = vec![EnvelopeFile {
        name: format!("{job_id}.mp4"),
        mime: "video/mp4".into(),
        kind: "output".into(),
        data: std::mem::take(&mut bytes),
    }];
    bytes.zeroize();
    Ok(EnvelopeOut {
        ok: true,
        action: "download_output".into(),
        metadata: json!({
            "job_id": job_id,
            "mime": "video/mp4",
            "bytes": nbytes,
        }),
        files,
        error: None,
    })
}

async fn flush_all(state: &AppState) -> Result<EnvelopeOut, String> {
    let ids: Vec<Uuid> = state.jobs.iter().map(|entry| *entry.key()).collect();
    for id in &ids {
        if let Some(mut job) = state.jobs.get_mut(id) {
            job.status = JobStatus::Aborted;
            job.stage = "aborted".into();
            job.detail = "Flushed by user".into();
        }
    }
    let worker_ok = state.ipc.control("flush", "").await.is_ok();
    state.storage.flush_all_now().await;
    state.media.clear();
    state.jobs.clear();
    if !worker_ok {
        return Err("flush_failed".into());
    }
    Ok(EnvelopeOut {
        ok: true,
        action: "flush_all".into(),
        metadata: json!({
            "flushed": true,
            "revoked": true,
        }),
        files: Vec::new(),
        error: None,
    })
}

fn session_from_headers(headers: &HeaderMap) -> Option<Uuid> {
    let value = headers.get("x-session-id")?.to_str().ok()?;
    Uuid::parse_str(value.trim()).ok()
}

fn sealed_error(key: &SessionKey, action: &str, code: &str) -> Response {
    let out = EnvelopeOut {
        ok: false,
        action: action.to_string(),
        metadata: json!({}),
        files: Vec::new(),
        error: Some(code.to_string()),
    };
    encode_envelope(key, out)
}

fn encode_envelope(key: &SessionKey, mut envelope: EnvelopeOut) -> Response {
    let packed = match rmp_serde::to_vec_named(&envelope) {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    for file in &mut envelope.files {
        file.data.zeroize();
    }
    let mut padded = match crypto::pad_block(&packed) {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    let sealed = match crypto::seal(key.as_bytes(), &padded) {
        Ok(bytes) => bytes,
        Err(_) => {
            padded.zeroize();
            return StatusCode::INTERNAL_SERVER_ERROR.into_response();
        }
    };
    padded.zeroize();
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/octet-stream"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        sealed,
    )
        .into_response()
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(state): State<AppState>,
) -> Response {
    let Ok(sid) = Uuid::parse_str(query.sid.trim()) else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    if !state.sessions.contains_key(&sid) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    ws.on_upgrade(move |socket| ws_loop(socket, sid, state))
}

async fn ws_loop(mut socket: WebSocket, sid: Uuid, state: AppState) {
    let mut rx = state.progress.subscribe();
    loop {
        tokio::select! {
            biased;
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(payload))) => {
                        if socket.send(Message::Pong(payload)).await.is_err() {
                            break;
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
            event = rx.recv() => {
                let event = match event {
                    Ok(event) => event,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(_) => break,
                };
                let Some(session) = state.sessions.get(&sid) else {
                    break;
                };
                let mut plain = match rmp_serde::to_vec_named(&event) {
                    Ok(bytes) => bytes,
                    Err(_) => break,
                };
                let sealed = match crypto::seal(session.key.as_bytes(), &plain) {
                    Ok(bytes) => bytes,
                    Err(_) => {
                        plain.zeroize();
                        break;
                    }
                };
                plain.zeroize();
                drop(session);
                if socket.send(Message::Binary(sealed.into())).await.is_err() {
                    break;
                }
            }
        }
    }
}

pub async fn sweep_sessions(state: AppState) {
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
    loop {
        interval.tick().await;
        state.sessions.retain(|_, session| {
            session.created.elapsed().as_secs() < TTL_SECONDS
        });
        state.jobs.retain(|_, job| job.created.elapsed().as_secs() < TTL_SECONDS);
        state.media.retain(|_, media| {
            SystemTime::now()
                .duration_since(media.created)
                .unwrap_or_default()
                .as_secs()
                < TTL_SECONDS
                && media.path.exists()
        });
    }
}
