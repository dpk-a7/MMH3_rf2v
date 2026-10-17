use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::time::timeout;

const MAX_FRAME: usize = 64 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
pub struct ProgressEvent {
    pub job_id: String,
    pub stage: String,
    pub progress: f32,
    pub detail: String,
    pub done: bool,
    pub ok: bool,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct IpcFrame {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub job_id: String,
    #[serde(default)]
    pub payload: Value,
}

pub struct IpcClient {
    pub sock_path: PathBuf,
}

impl IpcClient {
    pub fn new(sock_path: PathBuf) -> Self {
        Self { sock_path }
    }

    pub async fn generate<F>(
        &self,
        job_id: &str,
        payload: Value,
        mut on_progress: F,
    ) -> Result<Value, String>
    where
        F: FnMut(ProgressEvent),
    {
        let mut stream = connect_retry(&self.sock_path, 180, Duration::from_millis(500))
            .await
            .map_err(|_| "worker_unavailable".to_string())?;
        let frame = IpcFrame {
            kind: "generate".into(),
            job_id: job_id.to_string(),
            payload,
        };
        write_frame(&mut stream, &frame)
            .await
            .map_err(|_| "worker_unavailable".to_string())?;
        loop {
            let incoming = read_frame(&mut stream)
                .await
                .map_err(|_| "inference_failed".to_string())?;
            match incoming.kind.as_str() {
                "progress" => {
                    let stage = incoming
                        .payload
                        .get("stage")
                        .and_then(|v| v.as_str())
                        .unwrap_or("running")
                        .to_string();
                    let progress = incoming
                        .payload
                        .get("progress")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0) as f32;
                    let detail = incoming
                        .payload
                        .get("detail")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    on_progress(ProgressEvent {
                        job_id: job_id.to_string(),
                        stage,
                        progress,
                        detail,
                        done: false,
                        ok: true,
                    });
                }
                "result" => return Ok(incoming.payload),
                "error" => {
                    let code = incoming
                        .payload
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("inference_failed");
                    return Err(sanitize_code(code));
                }
                _ => return Err("inference_failed".into()),
            }
        }
    }

    pub async fn control(&self, kind: &str, job_id: &str) -> Result<(), ()> {
        let stream = match timeout(
            Duration::from_secs(3),
            UnixStream::connect(&self.sock_path),
        )
        .await
        {
            Ok(Ok(stream)) => stream,
            _ => return Err(()),
        };
        let mut stream = stream;
        let frame = IpcFrame {
            kind: kind.to_string(),
            job_id: job_id.to_string(),
            payload: Value::Null,
        };
        write_frame(&mut stream, &frame).await.map_err(|_| ())?;
        let _ = timeout(Duration::from_secs(180), read_frame(&mut stream)).await;
        Ok(())
    }
}

fn sanitize_code(code: &str) -> String {
    match code {
        "model_missing"
        | "bad_reference"
        | "aborted"
        | "mux_failed"
        | "inference_failed"
        | "worker_unavailable"
        | "insufficient_vram"
        | "insufficient_ram"
        | "insufficient_storage"
        | "server_busy" => code.to_string(),
        _ => "inference_failed".to_string(),
    }
}

async fn connect_retry(path: &Path, attempts: usize, pause: Duration) -> Result<UnixStream, ()> {
    for _ in 0..attempts {
        if let Ok(stream) = UnixStream::connect(path).await {
            return Ok(stream);
        }
        tokio::time::sleep(pause).await;
    }
    Err(())
}

async fn write_frame(stream: &mut UnixStream, frame: &IpcFrame) -> Result<(), ()> {
    let bytes = rmp_serde::to_vec_named(frame).map_err(|_| ())?;
    if bytes.len() > MAX_FRAME {
        return Err(());
    }
    let len = (bytes.len() as u32).to_le_bytes();
    stream.write_all(&len).await.map_err(|_| ())?;
    stream.write_all(&bytes).await.map_err(|_| ())?;
    stream.flush().await.map_err(|_| ())?;
    Ok(())
}

async fn read_frame(stream: &mut UnixStream) -> Result<IpcFrame, ()> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await.map_err(|_| ())?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > MAX_FRAME {
        return Err(());
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await.map_err(|_| ())?;
    rmp_serde::from_slice(&buf).map_err(|_| ())
}
