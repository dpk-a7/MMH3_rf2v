use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub const TTL_SECONDS: u64 = 86_400;

pub struct EphemeralStorage {
    pub root: PathBuf,
    pub uploads_dir: PathBuf,
    pub outputs_dir: PathBuf,
}

impl EphemeralStorage {
    pub async fn create(root: PathBuf) -> Self {
        let uploads_dir = root.join("uploads");
        let outputs_dir = root.join("outputs");
        let _ = fs::create_dir_all(&uploads_dir).await;
        let _ = fs::create_dir_all(&outputs_dir).await;
        restrict_dir(&root).await;
        restrict_dir(&uploads_dir).await;
        restrict_dir(&outputs_dir).await;
        Self {
            root,
            uploads_dir,
            outputs_dir,
        }
    }

    pub async fn secure_remove_file(path: &Path) {
        let Ok(meta) = fs::symlink_metadata(path).await else {
            return;
        };
        if meta.file_type().is_symlink() {
            let _ = fs::remove_file(path).await;
            return;
        }
        if meta.is_dir() {
            if let Ok(mut entries) = fs::read_dir(path).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    Box::pin(Self::secure_remove_file(&entry.path())).await;
                }
            }
            let _ = fs::remove_dir(path).await;
            return;
        }
        if meta.is_file() {
            if let Ok(mut file) = fs::OpenOptions::new().write(true).open(path).await {
                let zeros = vec![0u8; 65_536];
                let mut remaining = meta.len();
                while remaining > 0 {
                    let to_write = remaining.min(zeros.len() as u64) as usize;
                    if file.write_all(&zeros[..to_write]).await.is_err() {
                        break;
                    }
                    remaining -= to_write as u64;
                }
                let _ = file.sync_all().await;
            }
            let _ = fs::remove_file(path).await;
        }
    }

    pub async fn run_ttl_reaper(self: std::sync::Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            self.purge_expired().await;
        }
    }

    pub async fn purge_expired(&self) {
        let now = SystemTime::now();
        for dir in [&self.uploads_dir, &self.outputs_dir] {
            self.purge_dir(dir, now).await;
        }
    }

    async fn purge_dir(&self, dir: &Path, now: SystemTime) {
        let Ok(mut entries) = fs::read_dir(dir).await else {
            return;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let Ok(meta) = entry.metadata().await else {
                continue;
            };
            if meta.is_dir() {
                Box::pin(self.purge_dir(&entry.path(), now)).await;
                continue;
            }
            let Ok(modified) = meta.modified() else {
                continue;
            };
            if now.duration_since(modified).unwrap_or_default().as_secs() >= TTL_SECONDS {
                Self::secure_remove_file(&entry.path()).await;
            }
        }
    }

    pub async fn flush_all_now(&self) {
        for dir in [&self.uploads_dir, &self.outputs_dir] {
            let Ok(mut entries) = fs::read_dir(dir).await else {
                continue;
            };
            while let Ok(Some(entry)) = entries.next_entry().await {
                Self::secure_remove_file(&entry.path()).await;
            }
        }
    }
}

async fn restrict_dir(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).await;
    }
    let _ = path;
}
