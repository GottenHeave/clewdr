use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

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
use tracing::warn;

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

    fn unreferenced(&self) -> bool {
        self.references.is_empty() && self.active_leases == 0
    }
}

#[derive(Default, Serialize, Deserialize)]
struct PersistedFiles {
    version: u32,
    files: Vec<StoredFile>,
}

#[derive(Clone, Debug)]
struct FileIndex {
    files: HashMap<String, StoredFile>,
}

impl FileIndex {
    fn total_bytes(&self) -> u64 {
        self.files
            .values()
            .filter(|file| !file.expired)
            .map(|file| file.size_bytes)
            .sum()
    }
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
    upload_locks: Mutex<HashMap<String, Weak<Mutex<()>>>>,
    #[cfg(test)]
    faults: TestFaults,
}

#[cfg(test)]
#[derive(Default)]
struct TestFaults {
    persist: AtomicUsize,
    rename: AtomicUsize,
    delete: AtomicUsize,
}

pub(crate) struct StagedUpload {
    response: FileResponse,
    created: bool,
    _guard: OwnedMutexGuard<()>,
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

struct TempUploadGuard(Option<PathBuf>);

impl TempUploadGuard {
    fn new(path: PathBuf) -> Self {
        Self(Some(path))
    }

    fn disarm(&mut self) {
        self.0.take();
    }
}

impl Drop for TempUploadGuard {
    fn drop(&mut self) {
        let Some(path) = self.0.take() else { return };
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
        let store = Arc::new(Self {
            root,
            objects,
            metadata_path,
            key_path,
            max_file_bytes,
            max_staged_bytes,
            hmac_key: OnceCell::new(),
            index: Arc::new(Mutex::new(FileIndex { files })),
            upload_locks: Mutex::new(HashMap::new()),
            #[cfg(test)]
            faults: TestFaults::default(),
        });
        let _ = store.hmac_key().await?;
        store.reconcile_objects().await?;
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
            locks.retain(|_, lock| lock.strong_count() > 0);
            locks.get(&id).and_then(Weak::upgrade).unwrap_or_else(|| {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(id.clone(), Arc::downgrade(&lock));
                lock
            })
        };
        let guard = lock.lock_owned().await;

        let mut index = self.index.lock().await;
        if let Some(existing) = index.files.get(&id)
            && !existing.expired
        {
            let response = existing.response();
            drop(index);
            return Ok(StagedUpload {
                response,
                created: false,
                _guard: guard,
            });
        }
        let mut after = index.clone();
        let evicted = self.evict_unreferenced(&mut after, size_bytes);
        if after.total_bytes().saturating_add(size_bytes) > self.max_staged_bytes {
            return Err(file_error(
                StatusCode::INSUFFICIENT_STORAGE,
                "staged_storage_full",
                "Staged file storage has no capacity for this upload",
            ));
        }
        let object_path = self.object_path(&id);
        let created_object = if fs::try_exists(&object_path)
            .await
            .map_err(storage_io_error)?
        {
            false
        } else {
            self.rename_file(&temp_path, &object_path).await?;
            temp_guard.disarm();
            true
        };
        if let Err(error) = set_owner_only(&object_path, 0o600).await {
            if created_object {
                let _ = fs::remove_file(&object_path).await;
            }
            return Err(error);
        }
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
        after.files.insert(id, stored);
        if let Err(error) = self.persist_locked(&after).await {
            if created_object {
                let _ = fs::remove_file(&object_path).await;
            }
            return Err(error);
        }
        *index = after;
        drop(index);
        for id in evicted {
            if let Err(error) = self.remove_object(&self.object_path(&id)).await {
                warn!("Failed to remove evicted staged file {id}: {error:?}");
            }
        }
        Ok(StagedUpload {
            response,
            created: true,
            _guard: guard,
        })
    }

    async fn discard_created_unreferenced(&self, id: &str) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        if !index.files.get(id).is_some_and(StoredFile::unreferenced) {
            return Ok(());
        }
        let mut after = index.clone();
        after.files.remove(id);
        self.persist_locked(&after).await?;
        *index = after;
        drop(index);
        if let Err(error) = self.remove_object(&self.object_path(id)).await {
            warn!("Failed to remove rolled back staged file {id}: {error:?}");
        }
        Ok(())
    }

    pub async fn resolve(
        &self,
        principal: &AuthPrincipal,
        id: &str,
    ) -> Result<ResolvedFile, ProtocolError> {
        let mut index = self.index.lock().await;
        let Some(file) = index.files.get(id) else {
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
        let mut after = index.clone();
        let file = after
            .files
            .get_mut(id)
            .expect("staged file remains present while index lock is held");
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
        self.persist_locked(&after).await?;
        *index = after;
        Ok(resolved)
    }

    pub async fn add_reference(&self, id: &str, session_ref: &str) -> Result<bool, ProtocolError> {
        self.update_index(|index| {
            if let Some(file) = index.files.get_mut(id) {
                let inserted = file.references.insert(session_ref.to_owned());
                file.last_used = Utc::now().timestamp();
                return inserted;
            }
            false
        })
        .await
    }

    pub async fn remove_reference(
        &self,
        id: &str,
        session_ref: &str,
    ) -> Result<bool, ProtocolError> {
        self.update_index(|index| {
            index
                .files
                .get_mut(id)
                .is_some_and(|file| file.references.remove(session_ref))
        })
        .await
    }

    pub async fn remove_session_references(&self, session_ref: &str) -> Result<(), ProtocolError> {
        self.set_session_references(session_ref, &[]).await
    }

    pub async fn set_session_references(
        &self,
        session_ref: &str,
        staged_file_ids: &[String],
    ) -> Result<(), ProtocolError> {
        self.update_index(|index| {
            index.files.values_mut().for_each(|file| {
                file.references.remove(session_ref);
            });
            for id in staged_file_ids {
                if let Some(file) = index.files.get_mut(id)
                    && !file.expired
                {
                    file.references.insert(session_ref.to_owned());
                }
            }
        })
        .await
    }

    pub async fn reconcile_references(
        &self,
        expected: &BTreeSet<(String, String)>,
    ) -> Result<(), ProtocolError> {
        self.update_index(|index| {
            for (id, file) in &mut index.files {
                file.references
                    .retain(|reference| expected.contains(&(reference.clone(), id.clone())));
            }
            for (reference, id) in expected {
                if let Some(file) = index.files.get_mut(id)
                    && !file.expired
                {
                    file.references.insert(reference.clone());
                }
            }
        })
        .await?;
        self.reconcile_objects().await
    }

    pub async fn cleanup(&self) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let cutoff = Utc::now().timestamp() - FILE_TTL_SECONDS;
        let mut after = index.clone();
        let mut moved = Vec::new();
        for file in after
            .files
            .values_mut()
            .filter(|file| file.unreferenced() && file.last_used < cutoff)
        {
            let id = &file.id;
            let object = self.object_path(id);
            let trash = self
                .objects
                .join(format!(".cleanup-{}-{id}", uuid::Uuid::new_v4()));
            if let Err(error) = self.prepare_object_removal(&object, &trash).await {
                self.restore_moved_objects(&moved).await;
                return Err(error);
            }
            moved.push((trash, object));
            file.expired = true;
        }
        if let Err(error) = self.persist_locked(&after).await {
            self.restore_moved_objects(&moved).await;
            return Err(error);
        }
        *index = after;
        drop(index);
        for (trash, _) in moved {
            if let Err(error) = fs::remove_file(&trash).await
                && error.kind() != std::io::ErrorKind::NotFound
            {
                warn!("Failed to remove staged file cleanup artifact {trash:?}: {error}");
            }
        }
        Ok(())
    }

    fn evict_unreferenced(&self, index: &mut FileIndex, incoming: u64) -> Vec<String> {
        if index.total_bytes().saturating_add(incoming) <= self.max_staged_bytes {
            return Vec::new();
        }
        let mut candidates = index
            .files
            .values()
            .filter(|file| file.unreferenced() && !file.expired)
            .map(|file| (file.last_used, file.id.clone()))
            .collect::<Vec<_>>();
        candidates.sort();
        let mut evicted = Vec::new();
        for (_, id) in candidates {
            if index.total_bytes().saturating_add(incoming) <= self.max_staged_bytes {
                break;
            }
            if index.files.remove(&id).is_some() {
                evicted.push(id);
            }
        }
        evicted
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

    async fn update_index<R>(
        &self,
        update: impl FnOnce(&mut FileIndex) -> R,
    ) -> Result<R, ProtocolError> {
        let mut index = self.index.lock().await;
        let mut after = index.clone();
        let result = update(&mut after);
        self.persist_locked(&after).await?;
        *index = after;
        Ok(result)
    }

    async fn persist_locked(&self, index: &FileIndex) -> Result<(), ProtocolError> {
        #[cfg(test)]
        self.fail_if_requested(&self.faults.persist, "injected metadata write failure")?;
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

    async fn rename_file(&self, from: &Path, to: &Path) -> Result<(), ProtocolError> {
        #[cfg(test)]
        self.fail_if_requested(&self.faults.rename, "injected object rename failure")?;
        fs::rename(from, to).await.map_err(storage_io_error)
    }

    async fn remove_object(&self, path: &Path) -> Result<(), ProtocolError> {
        #[cfg(test)]
        self.fail_if_requested(&self.faults.delete, "injected object delete failure")?;
        match fs::remove_file(path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(storage_io_error(error)),
        }
    }

    async fn prepare_object_removal(
        &self,
        object: &Path,
        trash: &Path,
    ) -> Result<(), ProtocolError> {
        #[cfg(test)]
        self.fail_if_requested(&self.faults.delete, "injected object delete failure")?;
        match self.rename_file(object, trash).await {
            Ok(()) => Ok(()),
            Err(_error) if !object.exists() => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn restore_moved_objects(&self, moved: &[(PathBuf, PathBuf)]) {
        for (trash, object) in moved.iter().rev() {
            if let Err(error) = fs::rename(trash, object).await
                && error.kind() != std::io::ErrorKind::NotFound
            {
                warn!("Failed to restore staged file after cleanup rollback: {error}");
            }
        }
    }

    async fn reconcile_objects(&self) -> Result<(), ProtocolError> {
        let live = self
            .index
            .lock()
            .await
            .files
            .iter()
            .filter(|(_, file)| !file.expired)
            .map(|(id, _)| id.clone())
            .collect::<BTreeSet<_>>();
        let mut entries = fs::read_dir(&self.objects)
            .await
            .map_err(storage_io_error)?;
        while let Some(entry) = entries.next_entry().await.map_err(storage_io_error)? {
            let name = entry.file_name().to_string_lossy().into_owned();
            if live.contains(&name) {
                continue;
            }
            if let Some(id) = cleanup_artifact_id(&name)
                && live.contains(id)
                && !self.object_path(id).exists()
            {
                if let Err(error) = self.rename_file(&entry.path(), &self.object_path(id)).await {
                    warn!("Failed to restore staged file cleanup artifact {name}: {error:?}");
                }
                continue;
            }
            if let Err(error) = self.remove_object(&entry.path()).await {
                warn!("Failed to remove orphaned staged file object {name}: {error:?}");
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn fail_if_requested(
        &self,
        counter: &AtomicUsize,
        message: &'static str,
    ) -> Result<(), ProtocolError> {
        if counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |count| {
                count.checked_sub(1)
            })
            .is_ok()
        {
            return Err(files_unavailable(message));
        }
        Ok(())
    }
}

fn cleanup_artifact_id(name: &str) -> Option<&str> {
    let artifact = name.strip_prefix(".cleanup-")?;
    Some(&artifact[artifact.find("file_clewdr_v1_")?..])
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
    let normalized = portable
        .rsplit('/')
        .next()
        .map(str::trim)
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(invalid_filename)?
        .chars()
        .filter(|ch| !ch.is_control())
        .collect::<String>();
    if normalized.is_empty() {
        return Err(invalid_filename());
    }
    Ok(normalized)
}

fn invalid_filename() -> ProtocolError {
    invalid_multipart("Multipart file must have a valid base filename")
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

    struct Fixture {
        temp: tempfile::TempDir,
        store: Arc<StagedFileStore>,
    }
    impl Fixture {
        async fn new(limits: (u64, u64)) -> Self {
            let temp = tempfile::tempdir().unwrap();
            let store = StagedFileStore::persistent_with_limits(temp.path(), limits.0, limits.1)
                .await
                .unwrap();
            Self { temp, store }
        }
        async fn stage(&self, name: &str, bytes: &[u8]) -> Result<FileResponse, ProtocolError> {
            self.store
                .stage_stream(
                    &AuthPrincipal::for_authenticated_user(),
                    name,
                    "application/octet-stream",
                    stream::iter([Ok::<_, std::io::Error>(Bytes::copy_from_slice(bytes))]),
                )
                .await
        }
        async fn ok(&self, name: &str, bytes: &[u8]) -> FileResponse {
            self.stage(name, bytes).await.unwrap()
        }
        async fn restart(&mut self) {
            self.store = StagedFileStore::persistent_with_limits(
                self.temp.path(),
                self.store.max_file_bytes,
                self.store.max_staged_bytes,
            )
            .await
            .unwrap();
        }
        async fn expire(&self, id: &str) {
            self.store
                .index
                .lock()
                .await
                .files
                .get_mut(id)
                .unwrap()
                .last_used = Utc::now().timestamp() - FILE_TTL_SECONDS - 1;
        }
        async fn state(&self, id: Option<&str>) -> (usize, usize, usize, bool) {
            let index = self.store.index.lock().await;
            (
                index.files.len(),
                std::fs::read_dir(&self.store.objects).unwrap().count(),
                index.files.values().filter(|file| file.expired).count(),
                id.is_some_and(|id| self.store.object_path(id).exists()),
            )
        }
    }
    #[derive(Clone, Copy)]
    enum Fault {
        Persist,
        Rename,
        Delete,
    }
    fn inject(store: &StagedFileStore, fault: Fault) {
        match fault {
            Fault::Persist => &store.faults.persist,
            Fault::Rename => &store.faults.rename,
            Fault::Delete => &store.faults.delete,
        }
        .store(1, Ordering::SeqCst);
    }
    fn code<T: std::fmt::Debug>(result: Result<T, ProtocolError>, expected: &str) {
        assert_eq!(result.unwrap_err().code, expected);
    }
    fn principal() -> AuthPrincipal {
        AuthPrincipal::for_authenticated_user()
    }

    #[tokio::test]
    async fn quotas_and_expired_identity_status_survive_restart() {
        let mut limited = Fixture::new((4, 5)).await;
        code(limited.stage("large", b"12345").await, "file_too_large");
        let first = limited.ok("first", b"1234").await;
        limited
            .store
            .add_reference(&first.id, "session")
            .await
            .unwrap();
        code(limited.stage("second", b"12").await, "staged_storage_full");
        limited
            .store
            .remove_session_references("session")
            .await
            .unwrap();
        limited.expire(&first.id).await;
        limited.store.cleanup().await.unwrap();
        limited.restart().await;
        limited.ok("live", b"1234").await;
        limited.ok("incoming", b"1234").await;
        code(
            limited.store.resolve(&principal(), &first.id).await,
            "file_expired",
        );
        code(
            limited
                .store
                .resolve(&principal(), "file_clewdr_v1_missing")
                .await,
            "file_not_found",
        );
    }

    #[tokio::test]
    async fn leases_and_startup_cleanup_artifacts_preserve_live_files() {
        let mut fixture = Fixture::new((10, 10)).await;
        let file = fixture.ok("file", b"file").await;
        let lease = fixture.store.resolve(&principal(), &file.id).await.unwrap();
        fixture.expire(&file.id).await;
        fixture.store.cleanup().await.unwrap();
        assert!(fixture.store.object_path(&file.id).exists());
        drop(lease);
        tokio::task::yield_now().await;
        let artifact =
            fixture
                .store
                .objects
                .join(format!(".cleanup-{}-{}", uuid::Uuid::new_v4(), file.id));
        fs::rename(fixture.store.object_path(&file.id), &artifact)
            .await
            .unwrap();
        fixture.restart().await;
        assert!(!artifact.exists());
        assert!(!fixture.store.index.lock().await.files[&file.id].expired);
    }

    #[derive(Clone, Copy)]
    enum Operation {
        Stage,
        Evict,
        Cleanup,
    }
    #[tokio::test]
    async fn fault_operation_matrix_rolls_back_or_reconciles_after_restart() {
        let cases = [
            (Operation::Stage, Fault::Persist),
            (Operation::Stage, Fault::Rename),
            (Operation::Evict, Fault::Persist),
            (Operation::Evict, Fault::Delete),
            (Operation::Cleanup, Fault::Persist),
            (Operation::Cleanup, Fault::Delete),
        ];
        for (operation, fault) in cases {
            let mut fixture = Fixture::new((1, 1)).await;
            let previous = match operation {
                Operation::Stage => None,
                _ => Some(fixture.ok("old", b"a").await),
            };
            if matches!(operation, Operation::Cleanup) {
                fixture.expire(&previous.as_ref().unwrap().id).await;
            }
            inject(&fixture.store, fault);
            let result = match operation {
                Operation::Stage => fixture.stage("new", b"b").await.map(|_| ()),
                Operation::Evict => fixture.stage("new", b"b").await.map(|_| ()),
                Operation::Cleanup => fixture.store.cleanup().await,
            };
            if !matches!((operation, fault), (Operation::Evict, Fault::Delete)) {
                code(result, "staged_files_unavailable");
            }
            let old_id = previous.as_ref().map(|file| file.id.as_str());
            let before = match (operation, fault) {
                (Operation::Stage, _) => (0, 0, 0, false),
                (Operation::Evict, Fault::Delete) => (1, 2, 0, true),
                _ => (1, 1, 0, true),
            };
            assert_eq!(fixture.state(old_id).await, before);
            fixture.restart().await;
            let old_exists = !matches!((operation, fault), (Operation::Evict, Fault::Delete));
            let after = if matches!(operation, Operation::Stage) {
                (0, 0, 0, false)
            } else {
                (1, 1, 0, old_exists)
            };
            assert_eq!(fixture.state(old_id).await, after);
            if matches!(operation, Operation::Stage) {
                fixture.ok("new", b"b").await;
            }
        }
    }

    #[tokio::test]
    async fn existing_target_and_exact_reference_reconciliation() {
        let fixture = Fixture::new((10, 10)).await;
        let first = fixture.ok("same", b"same").await;
        fixture
            .store
            .update_index(|index| {
                index.files.remove(&first.id).unwrap();
            })
            .await
            .unwrap();
        let second = fixture.ok("same", b"same").await;
        assert_eq!(first.id, second.id);
        assert_eq!(fixture.state(Some(&second.id)).await, (1, 1, 0, true));
        fixture
            .store
            .set_session_references("stale", std::slice::from_ref(&second.id))
            .await
            .unwrap();
        fixture
            .store
            .reconcile_references(&BTreeSet::from([("expected".into(), second.id.clone())]))
            .await
            .unwrap();
        inject(&fixture.store, Fault::Persist);
        code(
            fixture.store.remove_session_references("expected").await,
            "staged_files_unavailable",
        );
        assert_eq!(
            fixture.store.index.lock().await.files[&second.id].references,
            BTreeSet::from(["expected".into()])
        );
    }
}
