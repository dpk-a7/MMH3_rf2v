mod api;
mod crypto;
mod ipc;
mod math;
mod storage;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum_server::tls_rustls::RustlsConfig;
use rcgen::{CertificateParams, DnType, ExtendedKeyUsagePurpose, KeyPair, KeyUsagePurpose};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::ServerConfig;
use tokio::sync::broadcast;
use zeroize::Zeroize;

use api::AppState;
use ipc::IpcClient;
use storage::EphemeralStorage;

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => std::process::exit(1),
    };
    if runtime.block_on(run()).is_err() {
        std::process::exit(1);
    }
}

async fn run() -> Result<(), ()> {
    let model_root = model_root();
    for sub in ["diffussion", "lora", "controlnet", "clip", "vae"] {
        let _ = std::fs::create_dir_all(model_root.join(sub));
    }
    let storage = Arc::new(EphemeralStorage::create(ephemeral_root()).await);
    let (progress, _) = broadcast::channel(256);
    let state = AppState {
        storage: storage.clone(),
        sessions: Arc::new(dashmap::DashMap::new()),
        media: Arc::new(dashmap::DashMap::new()),
        jobs: Arc::new(dashmap::DashMap::new()),
        progress,
        model_root,
        ipc: Arc::new(IpcClient::new(sock_path())),
        job_slots: Arc::new(tokio::sync::Semaphore::new(1)),
    };
    let reaper = storage.clone();
    tokio::spawn(async move {
        reaper.run_ttl_reaper().await;
    });
    let sweeper = state.clone();
    tokio::spawn(async move {
        api::sweep_sessions(sweeper).await;
    });

    let tls = build_tls()?;
    let app = api::router(state, static_dir());
    axum_server::bind_rustls(bind_addr(), tls)
        .serve(app.into_make_service())
        .await
        .map_err(|_| ())
}

fn build_tls() -> Result<RustlsConfig, ()> {
    let key_pair = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).map_err(|_| ())?;
    let mut params = CertificateParams::new(vec!["localhost".to_string(), "minimax-h3".to_string()])
        .map_err(|_| ())?;
    params
        .distinguished_name
        .push(DnType::CommonName, "minimax-h3-ref2va");
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let cert = params.self_signed(&key_pair).map_err(|_| ())?;
    let cert_der = CertificateDer::from(cert.der().to_vec());
    let mut pkcs8 = key_pair.serialize_der();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8.clone()));
    pkcs8.zeroize();

    let mut provider = rustls::crypto::ring::default_provider();
    provider
        .cipher_suites
        .retain(|suite| suite.suite() == rustls::CipherSuite::TLS13_AES_256_GCM_SHA384);
    if provider.cipher_suites.is_empty() {
        return Err(());
    }
    let mut server = ServerConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|_| ())?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|_| ())?;
    server.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(RustlsConfig::from_config(Arc::new(server)))
}

fn bind_addr() -> SocketAddr {
    std::env::var("BIND_ADDR")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| SocketAddr::from(([0, 0, 0, 0], 8443)))
}

fn model_root() -> PathBuf {
    if let Ok(path) = std::env::var("MODEL_ROOT") {
        return PathBuf::from(path);
    }
    if Path::new("/app/model").exists() {
        PathBuf::from("/app/model")
    } else {
        PathBuf::from("model")
    }
}

fn static_dir() -> PathBuf {
    if let Ok(path) = std::env::var("STATIC_DIR") {
        return PathBuf::from(path);
    }
    if Path::new("/app/static").exists() {
        PathBuf::from("/app/static")
    } else {
        PathBuf::from("frontend/dist")
    }
}

fn ephemeral_root() -> PathBuf {
    if let Ok(path) = std::env::var("EPHEMERAL_DIR") {
        return PathBuf::from(path);
    }
    let app = PathBuf::from("/app/ephemeral");
    if app.exists() || std::fs::create_dir_all(&app).is_ok() {
        return app;
    }
    let local = PathBuf::from("ephemeral");
    let _ = std::fs::create_dir_all(&local);
    local
}

fn sock_path() -> PathBuf {
    std::env::var("IPC_SOCK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/dev/shm/minimax_h3.sock"))
}
