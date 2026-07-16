use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
};

use axum::http::StatusCode;
use bytes::Bytes;
use chrono::Utc;
use futures::{Stream, StreamExt};
use hmac::{Hmac, KeyInit, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{Mutex, OnceCell, OwnedMutexGuard},
};

use super::{AuthPrincipal, ProtocolError};

pub const DEFAULT_MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MAX_STAGED_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const FILE_TTL_SECONDS: i64 = 25 * 24 * 60 * 60;
const METADATA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize)]
pub struct FileResponse {
    pub id: String,
    pub r#type: &'static str,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ResolvedFile {
    pub id: String,
    pub path: PathBuf,
    pub filename: String,
    pub mime_type: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredFile {
    id: String,
    principal: String,
    filename: String,
    mime_type: String,
    size_bytes: u64,
    content_sha256: String,
    created_at: i64,
    last_used: i64,
    #[serde(default)]
    references: BTreeSet<String>,
    #[serde(default)]
    expired: bool,
}

impl StoredFile {
    fn response(&self) -> FileResponse {
        FileResponse {
            id: self.id.clone(),
            r#type: "file",
            filename: self.filename.clone(),
            mime_type: self.mime_type.clone(),
            size_bytes: self.size_bytes,
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedFiles {
    version: u32,
    files: Vec<StoredFile>,
}

struct FileIndex {
    files: HashMap<String, StoredFile>,
    total_bytes: u64,
}

pub struct StagedFileStore {
    root: PathBuf,
    objects: PathBuf,
    metadata_path: PathBuf,
    key_path: PathBuf,
    max_file_bytes: u64,
    max_staged_bytes: u64,
    hmac_key: OnceCell<[u8; 32]>,
    index: Mutex<FileIndex>,
    upload_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

struct UploadOperation {
    id: String,
    _guard: OwnedMutexGuard<()>,
    registry: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

pub(crate) struct StagedUpload {
    response: FileResponse,
    created: bool,
    _operation: UploadOperation,
}

impl StagedUpload {
    pub(crate) fn into_response(self) -> FileResponse {
        self.response
    }

    pub(crate) async fn rollback(self, store: &StagedFileStore) -> Result<(), ProtocolError> {
        if self.created {
            store
                .discard_created_unreferenced(&self.response.id)
                .await?;
        }
        Ok(())
    }
}

struct TempUploadGuard {
    path: PathBuf,
    active: bool,
}

impl TempUploadGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, active: true }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for TempUploadGuard {
    fn drop(&mut self) {
        if self.active {
            let path = self.path.clone();
            if tokio::runtime::Handle::try_current().is_ok() {
                tokio::spawn(async move {
                    tokio::task::yield_now().await;
                    let _ = fs::remove_file(path).await;
                });
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

impl Drop for UploadOperation {
    fn drop(&mut self) {
        let id = self.id.clone();
        let registry = self.registry.clone();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            let mut locks = registry.lock().await;
            if locks
                .get(&id)
                .is_some_and(|lock| Arc::strong_count(lock) == 1)
            {
                locks.remove(&id);
            }
        });
    }
}

impl StagedFileStore {
    pub async fn persistent(root: impl Into<PathBuf>) -> Result<Arc<Self>, ProtocolError> {
        Self::persistent_with_limits(root, DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_STAGED_BYTES).await
    }

    pub async fn persistent_with_limits(
        root: impl Into<PathBuf>,
        max_file_bytes: u64,
        max_staged_bytes: u64,
    ) -> Result<Arc<Self>, ProtocolError> {
        let root = root.into();
        let objects = root.join("objects");
        fs::create_dir_all(&objects)
            .await
            .map_err(storage_io_error)?;
        set_owner_only_dir(&root).await?;
        set_owner_only_dir(&objects).await?;
        let metadata_path = root.join("files.json");
        let key_path = root.join("hmac.key");
        let files = load_metadata(&metadata_path).await?;
        let total_bytes = files
            .values()
            .filter(|file| !file.expired)
            .map(|file| file.size_bytes)
            .sum();
        let store = Arc::new(Self {
            root,
            objects,
            metadata_path,
            key_path,
            max_file_bytes,
            max_staged_bytes,
            hmac_key: OnceCell::new(),
            index: Mutex::new(FileIndex { files, total_bytes }),
            upload_locks: Arc::new(Mutex::new(HashMap::new())),
        });
        let _ = store.hmac_key().await?;
        Ok(store)
    }

    pub async fn stage_stream<S, E>(
        &self,
        principal: &AuthPrincipal,
        filename: &str,
        mime_type: &str,
        stream: S,
    ) -> Result<FileResponse, ProtocolError>
    where
        S: Stream<Item = Result<Bytes, E>>,
        E: std::fmt::Display,
    {
        self.stage_stream_with_status(principal, filename, mime_type, stream)
            .await
            .map(StagedUpload::into_response)
    }

    pub(crate) async fn stage_stream_with_status<S, E>(
        &self,
        principal: &AuthPrincipal,
        filename: &str,
        mime_type: &str,
        stream: S,
    ) -> Result<StagedUpload, ProtocolError>
    where
        S: Stream<Item = Result<Bytes, E>>,
        E: std::fmt::Display,
    {
        let filename = normalize_filename(filename)?;
        let mime_type = normalize_mime(mime_type)?;
        let mut identity = Hmac::<Sha256>::new_from_slice(self.hmac_key().await?)
            .expect("HMAC accepts a 256-bit key");
        identity.update(b"clewdr-staged-file-v1\0");
        update_identity_field(&mut identity, principal.0.as_bytes());
        update_identity_field(&mut identity, filename.as_bytes());
        update_identity_field(&mut identity, mime_type.as_bytes());
        let temp_path = self
            .root
            .join(format!("upload-{}.tmp", uuid::Uuid::new_v4()));
        let mut output = fs::File::create(&temp_path)
            .await
            .map_err(storage_io_error)?;
        let mut temp_guard = TempUploadGuard::new(temp_path.clone());
        set_owner_only_file(&temp_path).await?;
        let mut digest = Sha256::new();
        let mut size_bytes = 0u64;
        futures::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                ProtocolError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_multipart",
                    format!("Failed to read multipart file data: {error}"),
                )
            })?;
            size_bytes = size_bytes.saturating_add(chunk.len() as u64);
            if size_bytes > self.max_file_bytes {
                return Err(ProtocolError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "file_too_large",
                    format!("File exceeds the {} byte limit", self.max_file_bytes),
                ));
            }
            digest.update(&chunk);
            identity.update(&chunk);
            output.write_all(&chunk).await.map_err(storage_io_error)?;
        }
        output.flush().await.map_err(storage_io_error)?;
        output.sync_all().await.map_err(storage_io_error)?;
        drop(output);

        let content_sha256 = hex::encode(digest.finalize());
        let id = format!(
            "file_clewdr_v1_{}",
            hex::encode(identity.finalize().into_bytes())
        );
        let lock = {
            let mut locks = self.upload_locks.lock().await;
            locks
                .entry(id.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let operation = UploadOperation {
            id: id.clone(),
            _guard: lock.lock_owned().await,
            registry: self.upload_locks.clone(),
        };

        let mut index = self.index.lock().await;
        if let Some(existing) = index.files.get(&id)
            && !existing.expired
        {
            let response = existing.response();
            drop(index);
            return Ok(StagedUpload {
                response,
                created: false,
                _operation: operation,
            });
        }
        self.evict_unreferenced(&mut index, size_bytes).await?;
        if index.total_bytes.saturating_add(size_bytes) > self.max_staged_bytes {
            return Err(ProtocolError::new(
                StatusCode::INSUFFICIENT_STORAGE,
                "staged_storage_full",
                "Staged file storage has no capacity for this upload",
            ));
        }
        let object_path = self.object_path(&id);
        fs::rename(&temp_path, &object_path)
            .await
            .map_err(storage_io_error)?;
        temp_guard.disarm();
        set_owner_only_file(&object_path).await?;
        let now = Utc::now().timestamp();
        let stored = StoredFile {
            id: id.clone(),
            principal: principal.0.clone(),
            filename,
            mime_type,
            size_bytes,
            content_sha256,
            created_at: now,
            last_used: now,
            references: BTreeSet::new(),
            expired: false,
        };
        let response = stored.response();
        index.total_bytes += size_bytes;
        index.files.insert(id, stored);
        self.persist_locked(&index).await?;
        Ok(StagedUpload {
            response,
            created: true,
            _operation: operation,
        })
    }

    async fn discard_created_unreferenced(&self, id: &str) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        if index
            .files
            .get(id)
            .is_some_and(|file| file.references.is_empty())
            && let Some(file) = index.files.remove(id)
        {
            index.total_bytes = index.total_bytes.saturating_sub(file.size_bytes);
            let _ = fs::remove_file(self.object_path(id)).await;
            self.persist_locked(&index).await?;
        }
        Ok(())
    }

    pub async fn resolve(
        &self,
        principal: &AuthPrincipal,
        id: &str,
    ) -> Result<ResolvedFile, ProtocolError> {
        let mut index = self.index.lock().await;
        let Some(file) = index.files.get_mut(id) else {
            return Err(ProtocolError::new(
                StatusCode::NOT_FOUND,
                "file_not_found",
                "Staged file does not exist",
            ));
        };
        if file.principal != principal.0 {
            return Err(ProtocolError::new(
                StatusCode::FORBIDDEN,
                "file_forbidden",
                "Staged file belongs to another authenticated principal",
            ));
        }
        if file.expired {
            return Err(ProtocolError::new(
                StatusCode::GONE,
                "file_expired",
                "Staged file has expired",
            ));
        }
        if file.references.is_empty()
            && Utc::now().timestamp().saturating_sub(file.last_used) > FILE_TTL_SECONDS
        {
            return Err(ProtocolError::new(
                StatusCode::GONE,
                "file_expired",
                "Staged file has expired",
            ));
        }
        if !self.object_path(id).exists() {
            return Err(ProtocolError::new(
                StatusCode::NOT_FOUND,
                "file_not_found",
                "Staged file data is missing",
            ));
        }
        file.last_used = Utc::now().timestamp();
        let resolved = ResolvedFile {
            id: file.id.clone(),
            path: self.object_path(id),
            filename: file.filename.clone(),
            mime_type: file.mime_type.clone(),
            size_bytes: file.size_bytes,
        };
        self.persist_locked(&index).await?;
        Ok(resolved)
    }

    pub async fn add_reference(&self, id: &str, session_ref: &str) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        if let Some(file) = index.files.get_mut(id) {
            file.references.insert(session_ref.to_owned());
            file.last_used = Utc::now().timestamp();
            self.persist_locked(&index).await?;
        }
        Ok(())
    }

    pub async fn remove_session_references(&self, session_ref: &str) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        for file in index.files.values_mut() {
            file.references.remove(session_ref);
        }
        self.persist_locked(&index).await
    }

    pub async fn reconcile_references(
        &self,
        references: &HashMap<String, BTreeSet<String>>,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        for (id, file) in &mut index.files {
            file.references = references.get(id).cloned().unwrap_or_default();
        }
        self.persist_locked(&index).await
    }

    pub async fn cleanup(&self) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let cutoff = Utc::now().timestamp() - FILE_TTL_SECONDS;
        let expired = index
            .files
            .iter()
            .filter(|(_, file)| file.references.is_empty() && file.last_used < cutoff)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in expired {
            let size_bytes = index
                .files
                .get(&id)
                .map(|file| file.size_bytes)
                .unwrap_or_default();
            index.total_bytes = index.total_bytes.saturating_sub(size_bytes);
            if let Some(file) = index.files.get_mut(&id) {
                let _ = fs::remove_file(self.object_path(&id)).await;
                file.expired = true;
            }
        }
        self.persist_locked(&index).await
    }

    async fn evict_unreferenced(
        &self,
        index: &mut FileIndex,
        incoming: u64,
    ) -> Result<(), ProtocolError> {
        if index.total_bytes.saturating_add(incoming) <= self.max_staged_bytes {
            return Ok(());
        }
        let mut candidates = index
            .files
            .values()
            .filter(|file| file.references.is_empty())
            .map(|file| (file.last_used, file.id.clone()))
            .collect::<Vec<_>>();
        candidates.sort();
        for (_, id) in candidates {
            if index.total_bytes.saturating_add(incoming) <= self.max_staged_bytes {
                break;
            }
            if let Some(file) = index.files.remove(&id) {
                index.total_bytes = index.total_bytes.saturating_sub(file.size_bytes);
                fs::remove_file(self.object_path(&id))
                    .await
                    .map_err(storage_io_error)?;
            }
        }
        Ok(())
    }

    async fn hmac_key(&self) -> Result<&[u8; 32], ProtocolError> {
        self.hmac_key
            .get_or_try_init(|| async {
                match fs::read(&self.key_path).await {
                    Ok(bytes) if bytes.len() == 32 => {
                        let mut key = [0u8; 32];
                        key.copy_from_slice(&bytes);
                        Ok(key)
                    }
                    Ok(_) => Err(ProtocolError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "staged_files_unavailable",
                        "Staged file HMAC key has an invalid length",
                    )),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        let mut key = [0u8; 32];
                        rand::rng().fill_bytes(&mut key);
                        let temp = self.key_path.with_extension("key.tmp");
                        fs::write(&temp, key).await.map_err(storage_io_error)?;
                        set_owner_only_file(&temp).await?;
                        fs::rename(temp, &self.key_path)
                            .await
                            .map_err(storage_io_error)?;
                        Ok(key)
                    }
                    Err(error) => Err(storage_io_error(error)),
                }
            })
            .await
    }

    fn object_path(&self, id: &str) -> PathBuf {
        self.objects.join(id)
    }

    async fn persist_locked(&self, index: &FileIndex) -> Result<(), ProtocolError> {
        let snapshot = PersistedFiles {
            version: METADATA_VERSION,
            files: index.files.values().cloned().collect(),
        };
        let data = serde_json::to_vec_pretty(&snapshot).map_err(|error| {
            ProtocolError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "staged_files_unavailable",
                format!("Failed to serialize staged file metadata: {error}"),
            )
        })?;
        let temp = self.metadata_path.with_extension("json.tmp");
        fs::write(&temp, data).await.map_err(storage_io_error)?;
        set_owner_only_file(&temp).await?;
        fs::rename(temp, &self.metadata_path)
            .await
            .map_err(storage_io_error)
    }
}

fn update_identity_field(mac: &mut Hmac<Sha256>, field: &[u8]) {
    mac.update(&(field.len() as u64).to_be_bytes());
    mac.update(field);
}

async fn load_metadata(path: &Path) -> Result<HashMap<String, StoredFile>, ProtocolError> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(error) => return Err(storage_io_error(error)),
    };
    let persisted: PersistedFiles = serde_json::from_slice(&bytes).map_err(|error| {
        ProtocolError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "staged_files_unavailable",
            format!("Failed to parse staged file metadata: {error}"),
        )
    })?;
    if persisted.version != METADATA_VERSION {
        return Err(ProtocolError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "staged_files_unavailable",
            "Unsupported staged file metadata version",
        ));
    }
    Ok(persisted
        .files
        .into_iter()
        .map(|file| (file.id.clone(), file))
        .collect())
}

fn normalize_filename(input: &str) -> Result<String, ProtocolError> {
    let portable = input.replace('\\', "/");
    let filename = portable
        .rsplit('/')
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(|| {
            ProtocolError::new(
                StatusCode::BAD_REQUEST,
                "invalid_multipart",
                "Multipart file must have a valid base filename",
            )
        })?;
    let normalized = filename
        .chars()
        .filter(|ch| !ch.is_control())
        .collect::<String>();
    if normalized.is_empty() {
        return Err(ProtocolError::new(
            StatusCode::BAD_REQUEST,
            "invalid_multipart",
            "Multipart file must have a valid base filename",
        ));
    }
    Ok(normalized)
}

fn normalize_mime(input: &str) -> Result<String, ProtocolError> {
    let mime = input
        .split(';')
        .next()
        .map(str::trim)
        .filter(|mime| mime.contains('/') && !mime.chars().any(char::is_whitespace))
        .ok_or_else(|| {
            ProtocolError::new(
                StatusCode::BAD_REQUEST,
                "invalid_multipart",
                "Multipart file must declare a valid MIME type",
            )
        })?;
    Ok(mime.to_ascii_lowercase())
}

fn storage_io_error(error: std::io::Error) -> ProtocolError {
    ProtocolError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "staged_files_unavailable",
        format!("Staged file storage error: {error}"),
    )
}

#[cfg(unix)]
async fn set_owner_only_file(path: &Path) -> Result<(), ProtocolError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .await
        .map_err(storage_io_error)
}

#[cfg(not(unix))]
async fn set_owner_only_file(_path: &Path) -> Result<(), ProtocolError> {
    Ok(())
}

#[cfg(unix)]
async fn set_owner_only_dir(path: &Path) -> Result<(), ProtocolError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .await
        .map_err(storage_io_error)
}

#[cfg(not(unix))]
async fn set_owner_only_dir(_path: &Path) -> Result<(), ProtocolError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use futures::stream;

    use super::*;

    async fn stage(
        store: &StagedFileStore,
        principal: &AuthPrincipal,
        name: &str,
        mime: &str,
        bytes: &'static [u8],
    ) -> Result<FileResponse, ProtocolError> {
        store
            .stage_stream(
                principal,
                name,
                mime,
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(bytes))]),
            )
            .await
    }

    #[tokio::test]
    async fn deterministic_ids_survive_restart_and_include_identity_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let first = stage(
            &store,
            &principal,
            "../report.pdf",
            "application/pdf",
            b"bytes",
        )
        .await
        .unwrap();
        assert_eq!(first.filename, "report.pdf");
        drop(store);

        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let duplicate = stage(
            &store,
            &principal,
            "report.pdf",
            "application/pdf",
            b"bytes",
        )
        .await
        .unwrap();
        assert_eq!(first.id, duplicate.id);
        assert_ne!(
            first.id,
            stage(&store, &principal, "other.pdf", "application/pdf", b"bytes")
                .await
                .unwrap()
                .id
        );
        assert_ne!(
            first.id,
            stage(
                &store,
                &AuthPrincipal::test_principal("other"),
                "report.pdf",
                "application/pdf",
                b"bytes"
            )
            .await
            .unwrap()
            .id
        );
    }

    #[tokio::test]
    async fn enforces_per_file_and_referenced_aggregate_limits() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent_with_limits(temp.path(), 4, 5)
            .await
            .unwrap();
        assert_eq!(
            stage(
                &store,
                &principal,
                "large.bin",
                "application/octet-stream",
                b"12345"
            )
            .await
            .unwrap_err()
            .code,
            "file_too_large"
        );
        let first = stage(
            &store,
            &principal,
            "first.bin",
            "application/octet-stream",
            b"1234",
        )
        .await
        .unwrap();
        store.add_reference(&first.id, "session").await.unwrap();
        assert_eq!(
            stage(
                &store,
                &principal,
                "second.bin",
                "application/octet-stream",
                b"12"
            )
            .await
            .unwrap_err()
            .code,
            "staged_storage_full"
        );
    }

    #[tokio::test]
    async fn rejects_cross_principal_resolution_distinctly() {
        let temp = tempfile::tempdir().unwrap();
        let owner = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let file = stage(&store, &owner, "a.txt", "text/plain", b"a")
            .await
            .unwrap();
        assert_eq!(
            store
                .resolve(&AuthPrincipal::test_principal("other"), &file.id)
                .await
                .unwrap_err()
                .code,
            "file_forbidden"
        );
        assert_eq!(
            store
                .resolve(&owner, "file_clewdr_v1_missing")
                .await
                .unwrap_err()
                .code,
            "file_not_found"
        );
    }

    #[tokio::test]
    async fn concurrent_duplicate_uploads_create_one_object() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let (left, right) = tokio::join!(
            stage(&store, &principal, "a.txt", "text/plain", b"same"),
            stage(&store, &principal, "a.txt", "text/plain", b"same")
        );
        assert_eq!(left.unwrap().id, right.unwrap().id);
        let count = std::fs::read_dir(temp.path().join("objects"))
            .unwrap()
            .count();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn rolled_back_upload_retains_ownership_until_duplicate_can_create() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let first = store
            .stage_stream_with_status(
                &principal,
                "a.txt",
                "text/plain",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"same"))]),
            )
            .await
            .unwrap();
        assert!(first.created);

        let duplicate_store = store.clone();
        let duplicate_principal = principal.clone();
        let mut duplicate = tokio::spawn(async move {
            duplicate_store
                .stage_stream_with_status(
                    &duplicate_principal,
                    "a.txt",
                    "text/plain",
                    stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"same"))]),
                )
                .await
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut duplicate)
                .await
                .is_err()
        );

        first.rollback(&store).await.unwrap();
        let duplicate = duplicate.await.unwrap().unwrap();
        assert!(duplicate.created);
        let response = duplicate.into_response();
        store.resolve(&principal, &response.id).await.unwrap();
    }

    #[tokio::test]
    async fn ttl_cleanup_preserves_expired_identity_across_restart() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let file = stage(&store, &principal, "a.txt", "text/plain", b"a")
            .await
            .unwrap();
        {
            let mut index = store.index.lock().await;
            index.files.get_mut(&file.id).unwrap().last_used =
                Utc::now().timestamp() - FILE_TTL_SECONDS - 1;
        }
        store.cleanup().await.unwrap();
        assert_eq!(
            store.resolve(&principal, &file.id).await.unwrap_err().code,
            "file_expired"
        );
        drop(store);
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        assert_eq!(
            store.resolve(&principal, &file.id).await.unwrap_err().code,
            "file_expired"
        );
    }
}
