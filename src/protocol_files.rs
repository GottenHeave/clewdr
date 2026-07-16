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
use sha2::Sha256;
use tokio::{
    fs,
    io::AsyncWriteExt,
    sync::{Mutex, OnceCell, OwnedMutexGuard},
};

use crate::{
    protocol::{AuthPrincipal, ProtocolError},
    utils::write_json_atomically,
};

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

#[derive(Debug)]
pub struct ResolvedFile {
    pub path: PathBuf,
    pub filename: String,
    pub mime_type: String,
    _lease: FileLease,
}

#[derive(Debug)]
struct FileLease {
    id: String,
    index: Arc<Mutex<FileIndex>>,
}

impl Drop for FileLease {
    fn drop(&mut self) {
        let id = self.id.clone();
        let index = self.index.clone();
        tokio::spawn(async move {
            if let Some(file) = index.lock().await.files.get_mut(&id) {
                file.active_leases = file.active_leases.saturating_sub(1);
            }
        });
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StoredFile {
    id: String,
    principal: String,
    filename: String,
    mime_type: String,
    size_bytes: u64,
    last_used: i64,
    #[serde(default)]
    references: BTreeSet<String>,
    #[serde(default)]
    expired: bool,
    #[serde(skip)]
    active_leases: usize,
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

#[derive(Debug)]
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
    index: Arc<Mutex<FileIndex>>,
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
        set_owner_only(&root, 0o700).await?;
        set_owner_only(&objects, 0o700).await?;
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
            index: Arc::new(Mutex::new(FileIndex { files, total_bytes })),
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
        update_identity_field(&mut identity, principal.as_str().as_bytes());
        update_identity_field(&mut identity, filename.as_bytes());
        update_identity_field(&mut identity, mime_type.as_bytes());
        let temp_path = self
            .root
            .join(format!("upload-{}.tmp", uuid::Uuid::new_v4()));
        let mut output = fs::File::create(&temp_path)
            .await
            .map_err(storage_io_error)?;
        let mut temp_guard = TempUploadGuard::new(temp_path.clone());
        set_owner_only(&temp_path, 0o600).await?;
        let mut size_bytes = 0u64;
        futures::pin_mut!(stream);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|error| {
                invalid_multipart(format!("Failed to read multipart file data: {error}"))
            })?;
            size_bytes = size_bytes.saturating_add(chunk.len() as u64);
            if size_bytes > self.max_file_bytes {
                return Err(file_error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "file_too_large",
                    format!("File exceeds the {} byte limit", self.max_file_bytes),
                ));
            }
            identity.update(&chunk);
            output.write_all(&chunk).await.map_err(storage_io_error)?;
        }
        output.flush().await.map_err(storage_io_error)?;
        output.sync_all().await.map_err(storage_io_error)?;
        drop(output);

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
            return Err(file_error(
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
        set_owner_only(&object_path, 0o600).await?;
        let now = Utc::now().timestamp();
        let stored = StoredFile {
            id: id.clone(),
            principal: principal.as_str().to_owned(),
            filename,
            mime_type,
            size_bytes,
            last_used: now,
            references: BTreeSet::new(),
            expired: false,
            active_leases: 0,
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
            .is_some_and(|file| file.references.is_empty() && file.active_leases == 0)
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
            return Err(file_not_found("Staged file does not exist"));
        };
        if file.principal != principal.as_str() {
            return Err(file_error(
                StatusCode::FORBIDDEN,
                "file_forbidden",
                "Staged file belongs to another authenticated principal",
            ));
        }
        if file.expired {
            return Err(file_expired());
        }
        if file.references.is_empty()
            && Utc::now().timestamp().saturating_sub(file.last_used) > FILE_TTL_SECONDS
        {
            return Err(file_expired());
        }
        if !self.object_path(id).exists() {
            return Err(file_not_found("Staged file data is missing"));
        }
        file.last_used = Utc::now().timestamp();
        file.active_leases += 1;
        let resolved = ResolvedFile {
            path: self.object_path(id),
            filename: file.filename.clone(),
            mime_type: file.mime_type.clone(),
            _lease: FileLease {
                id: id.to_owned(),
                index: self.index.clone(),
            },
        };
        self.persist_locked(&index).await?;
        Ok(resolved)
    }

    pub async fn add_reference(&self, id: &str, session_ref: &str) -> Result<(), ProtocolError> {
        self.update_index(|index| {
            if let Some(file) = index.files.get_mut(id) {
                file.references.insert(session_ref.to_owned());
                file.last_used = Utc::now().timestamp();
            }
        })
        .await
    }

    pub async fn remove_session_references(&self, session_ref: &str) -> Result<(), ProtocolError> {
        self.update_index(|index| {
            index.files.values_mut().for_each(|file| {
                file.references.remove(session_ref);
            });
        })
        .await
    }

    pub async fn remove_orphaned_references(
        &self,
        existing_session_refs: &BTreeSet<String>,
    ) -> Result<(), ProtocolError> {
        self.update_index(|index| {
            index.files.values_mut().for_each(|file| {
                file.references
                    .retain(|reference| existing_session_refs.contains(reference));
            });
        })
        .await
    }

    pub async fn cleanup(&self) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let cutoff = Utc::now().timestamp() - FILE_TTL_SECONDS;
        let expired = index
            .files
            .iter()
            .filter(|(_, file)| {
                file.references.is_empty() && file.active_leases == 0 && file.last_used < cutoff
            })
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
            .filter(|file| file.references.is_empty() && file.active_leases == 0 && !file.expired)
            .map(|file| (file.last_used, file.id.clone()))
            .collect::<Vec<_>>();
        candidates.sort();
        for (_, id) in candidates {
            if index.total_bytes.saturating_add(incoming) <= self.max_staged_bytes {
                break;
            }
            if let Some(file) = index.files.remove(&id) {
                index.total_bytes = index.total_bytes.saturating_sub(file.size_bytes);
                match fs::remove_file(self.object_path(&id)).await {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(storage_io_error(error)),
                }
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
                    Ok(_) => Err(files_unavailable(
                        "Staged file HMAC key has an invalid length",
                    )),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        let mut key = [0u8; 32];
                        rand::rng().fill_bytes(&mut key);
                        let temp = self.key_path.with_extension("key.tmp");
                        fs::write(&temp, key).await.map_err(storage_io_error)?;
                        set_owner_only(&temp, 0o600).await?;
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

    async fn update_index(&self, update: impl FnOnce(&mut FileIndex)) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        update(&mut index);
        self.persist_locked(&index).await
    }

    async fn persist_locked(&self, index: &FileIndex) -> Result<(), ProtocolError> {
        let snapshot = PersistedFiles {
            version: METADATA_VERSION,
            files: index.files.values().cloned().collect(),
        };
        write_json_atomically(&self.metadata_path, &snapshot)
            .await
            .map_err(|error| {
                files_unavailable(format!("Failed to persist staged file metadata: {error}"))
            })
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
        files_unavailable(format!("Failed to parse staged file metadata: {error}"))
    })?;
    if persisted.version != METADATA_VERSION {
        return Err(files_unavailable(
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
        .ok_or_else(|| invalid_multipart("Multipart file must have a valid base filename"))?;
    let normalized = filename
        .chars()
        .filter(|ch| !ch.is_control())
        .collect::<String>();
    if normalized.is_empty() {
        return Err(invalid_multipart(
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
        .ok_or_else(|| invalid_multipart("Multipart file must declare a valid MIME type"))?;
    Ok(mime.to_ascii_lowercase())
}

fn file_error(status: StatusCode, code: &'static str, message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(status, code, message)
}

fn invalid_multipart(message: impl Into<String>) -> ProtocolError {
    file_error(StatusCode::BAD_REQUEST, "invalid_multipart", message)
}

fn file_not_found(message: &'static str) -> ProtocolError {
    file_error(StatusCode::NOT_FOUND, "file_not_found", message)
}

fn file_expired() -> ProtocolError {
    file_error(StatusCode::GONE, "file_expired", "Staged file has expired")
}

fn files_unavailable(message: impl Into<String>) -> ProtocolError {
    file_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "staged_files_unavailable",
        message,
    )
}

fn storage_io_error(error: std::io::Error) -> ProtocolError {
    files_unavailable(format!("Staged file storage error: {error}"))
}

#[cfg(unix)]
async fn set_owner_only(path: &Path, mode: u32) -> Result<(), ProtocolError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .await
        .map_err(storage_io_error)
}

#[cfg(not(unix))]
async fn set_owner_only(_path: &Path, _mode: u32) -> Result<(), ProtocolError> {
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
        bytes: &'static [u8],
    ) -> Result<FileResponse, ProtocolError> {
        store
            .stage_stream(
                principal,
                name,
                "application/octet-stream",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(bytes))]),
            )
            .await
    }

    async fn staged_upload(
        store: &StagedFileStore,
        principal: &AuthPrincipal,
        name: &str,
        bytes: &'static [u8],
    ) -> StagedUpload {
        store
            .stage_stream_with_status(
                principal,
                name,
                "application/octet-stream",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(bytes))]),
            )
            .await
            .unwrap()
    }

    fn code<T>(result: Result<T, ProtocolError>) -> &'static str {
        match result {
            Ok(_) => panic!("expected protocol error"),
            Err(error) => error.code,
        }
    }

    #[tokio::test]
    async fn identity_restart_and_principal_ownership_are_enforced() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let first = stage(&store, &principal, "../report.pdf", b"bytes")
            .await
            .unwrap();
        assert_eq!(first.filename, "report.pdf");
        drop(store);

        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let duplicate = stage(&store, &principal, "report.pdf", b"bytes")
            .await
            .unwrap();
        assert_eq!(first.id, duplicate.id);
        assert_ne!(
            first.id,
            stage(&store, &principal, "other.pdf", b"bytes")
                .await
                .unwrap()
                .id
        );
        let other = AuthPrincipal::test_principal("other");
        let other_file = stage(&store, &other, "report.pdf", b"bytes").await.unwrap();
        assert_ne!(first.id, other_file.id);
        assert_eq!(
            code(store.resolve(&other, &first.id).await),
            "file_forbidden"
        );
        assert_eq!(
            code(store.resolve(&principal, "file_clewdr_v1_missing").await),
            "file_not_found"
        );
    }

    #[tokio::test]
    async fn per_file_and_referenced_aggregate_limits_are_enforced() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent_with_limits(temp.path(), 4, 5)
            .await
            .unwrap();
        assert_eq!(
            code(stage(&store, &principal, "large.bin", b"12345").await),
            "file_too_large"
        );
        let first = stage(&store, &principal, "first.bin", b"1234")
            .await
            .unwrap();
        store.add_reference(&first.id, "session").await.unwrap();
        assert_eq!(
            code(stage(&store, &principal, "second.bin", b"12").await),
            "staged_storage_full"
        );
    }

    #[tokio::test]
    async fn concurrent_duplicates_and_rollback_serialize_by_identity() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent(temp.path()).await.unwrap();
        let (left, right) = tokio::join!(
            stage(&store, &principal, "a.txt", b"same"),
            stage(&store, &principal, "a.txt", b"same")
        );
        assert_eq!(left.unwrap().id, right.unwrap().id);
        assert_eq!(std::fs::read_dir(&store.objects).unwrap().count(), 1);

        let first = staged_upload(&store, &principal, "rollback.txt", b"rollback").await;
        assert!(first.created);
        let duplicate_store = store.clone();
        let duplicate_principal = principal.clone();
        let mut duplicate = tokio::spawn(async move {
            staged_upload(
                &duplicate_store,
                &duplicate_principal,
                "rollback.txt",
                b"rollback",
            )
            .await
        });
        let waiting = tokio::time::timeout(std::time::Duration::from_millis(50), &mut duplicate);
        assert!(waiting.await.is_err());
        first.rollback(&store).await.unwrap();
        let duplicate = duplicate.await.unwrap();
        assert!(duplicate.created);
        let response = duplicate.into_response();
        store.resolve(&principal, &response.id).await.unwrap();
    }

    #[tokio::test]
    async fn expired_identity_survives_restart_without_blocking_capacity() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent_with_limits(temp.path(), 2, 2)
            .await
            .unwrap();
        let file = stage(&store, &principal, "old.txt", b"aa").await.unwrap();
        {
            let mut index = store.index.lock().await;
            index.files.get_mut(&file.id).unwrap().last_used =
                Utc::now().timestamp() - FILE_TTL_SECONDS - 1;
        }
        store.cleanup().await.unwrap();
        assert_eq!(
            code(store.resolve(&principal, &file.id).await),
            "file_expired"
        );
        drop(store);
        let store = StagedFileStore::persistent_with_limits(temp.path(), 2, 2)
            .await
            .unwrap();
        assert_eq!(
            code(store.resolve(&principal, &file.id).await),
            "file_expired"
        );
        stage(&store, &principal, "new.txt", b"bb").await.unwrap();
    }

    #[tokio::test]
    async fn resolved_file_lease_prevents_eviction_during_upload() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent_with_limits(temp.path(), 1, 1)
            .await
            .unwrap();
        let first = stage(&store, &principal, "first.txt", b"a").await.unwrap();
        let lease = store.resolve(&principal, &first.id).await.unwrap();
        assert_eq!(
            code(stage(&store, &principal, "second.txt", b"b").await),
            "staged_storage_full"
        );
        drop(lease);
        tokio::task::yield_now().await;
        stage(&store, &principal, "second.txt", b"b").await.unwrap();
    }

    #[tokio::test]
    async fn orphan_cleanup_releases_session_references() {
        let temp = tempfile::tempdir().unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let store = StagedFileStore::persistent_with_limits(temp.path(), 1, 1)
            .await
            .unwrap();
        let first = stage(&store, &principal, "first.txt", b"a").await.unwrap();
        store
            .add_reference(&first.id, "expired-session")
            .await
            .unwrap();
        store
            .remove_orphaned_references(&BTreeSet::new())
            .await
            .unwrap();
        stage(&store, &principal, "second.txt", b"b").await.unwrap();
    }
}
