use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
};

use axum::http::StatusCode;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::types::claude::{ContentBlock, Message, MessageContent, Role};

use super::{AuthPrincipal, ProtocolError};

const SESSION_FILE_VERSION: u32 = 1;
const MAX_SESSION_RECORDS_PER_PRINCIPAL: usize = 4096;
const MAX_LIVE_SESSIONS_PER_PRINCIPAL: usize = 1024;
const TOMBSTONE_TTL_SECONDS: i64 = 25 * 24 * 60 * 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Provisioning,
    InFlight,
    Committed,
    Uncertain,
    Tombstoned,
}

impl SessionState {
    fn is_live(self) -> bool {
        !matches!(self, Self::Tombstoned)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTurn {
    pub parent_uuid_before: Option<String>,
    pub user_digests: Vec<String>,
    pub assistant_uuid_after: String,
    #[serde(default)]
    pub assistant_digests_before: Vec<String>,
    #[serde(default)]
    pub assistant_digest_after: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingTurn {
    pub parent_uuid_before: Option<String>,
    pub user_digests: Vec<String>,
    pub assistant_uuid_after: String,
    pub replace_from_turn: usize,
    #[serde(default)]
    pub assistant_digests_before: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolSession {
    pub principal: String,
    pub session_digest: String,
    pub state: SessionState,
    pub cookie_id: String,
    pub organization_uuid: String,
    pub conversation_uuid: String,
    pub model_digest: String,
    pub system_digest: String,
    pub turns: Vec<SessionTurn>,
    pub pending: Option<PendingTurn>,
    #[serde(default)]
    pub file_mappings: HashMap<String, String>,
    pub created_at: i64,
    pub last_used: i64,
}

impl ProtocolSession {
    pub fn session_ref(&self) -> String {
        session_ref(&self.principal, &self.session_digest)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReusePlan {
    Create,
    Append {
        parent_uuid: String,
        suffix_start: usize,
    },
    Fork {
        parent_uuid: Option<String>,
        suffix_start: usize,
        replace_from_turn: usize,
    },
    Regenerate {
        parent_uuid: Option<String>,
        suffix_start: usize,
        replace_from_turn: usize,
    },
}

#[derive(Serialize, Deserialize)]
struct PersistedSessions {
    version: u32,
    sessions: Vec<ProtocolSession>,
}

struct SessionIndex {
    sessions: HashMap<String, ProtocolSession>,
}

#[derive(Debug)]
pub struct SessionOperation {
    pub key: String,
    session_digest: String,
    _guard: OwnedMutexGuard<()>,
    lock_registry: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

impl Drop for SessionOperation {
    fn drop(&mut self) {
        let key = self.key.clone();
        let registry = self.lock_registry.clone();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            let mut locks = registry.lock().await;
            if locks
                .get(&key)
                .is_some_and(|lock| Arc::strong_count(lock) == 1)
            {
                locks.remove(&key);
            }
        });
    }
}

#[derive(Clone)]
pub struct SessionLifecycle {
    inner: Arc<SessionLifecycleInner>,
}

struct SessionLifecycleInner {
    store: Arc<ProtocolSessionStore>,
    operation: Mutex<Option<SessionOperation>>,
    finalized: AtomicBool,
}

impl SessionLifecycle {
    pub fn new(store: Arc<ProtocolSessionStore>, operation: SessionOperation) -> Self {
        Self {
            inner: Arc::new(SessionLifecycleInner {
                store,
                operation: Mutex::new(Some(operation)),
                finalized: AtomicBool::new(false),
            }),
        }
    }

    pub async fn commit(
        &self,
        assistant_digest_after: Option<String>,
    ) -> Result<(), ProtocolError> {
        let mut operation = self.inner.operation.lock().await;
        let Some(active) = operation.as_ref() else {
            return Ok(());
        };
        self.inner
            .store
            .commit(active, assistant_digest_after)
            .await?;
        operation.take();
        self.inner.finalized.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn uncertain(&self) {
        let mut operation = self.inner.operation.lock().await;
        if let Some(active) = operation.as_ref() {
            let _ = self.inner.store.mark_uncertain(active).await;
        }
        operation.take();
        self.inner.finalized.store(true, Ordering::Release);
    }
}

impl Drop for SessionLifecycle {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 || self.inner.finalized.load(Ordering::Acquire) {
            return;
        }
        let inner = self.inner.clone();
        tokio::spawn(async move {
            let mut operation = inner.operation.lock().await;
            if let Some(active) = operation.as_ref() {
                let _ = inner.store.mark_uncertain(active).await;
            }
            operation.take();
            inner.finalized.store(true, Ordering::Release);
        });
    }
}

pub struct StreamSessionGuard {
    lifecycle: SessionLifecycle,
    finalized: bool,
}

impl StreamSessionGuard {
    pub fn new(lifecycle: SessionLifecycle) -> Self {
        Self {
            lifecycle,
            finalized: false,
        }
    }

    pub async fn commit(
        &mut self,
        assistant_digest_after: Option<String>,
    ) -> Result<(), ProtocolError> {
        self.lifecycle.commit(assistant_digest_after).await?;
        self.finalized = true;
        Ok(())
    }

    pub async fn uncertain(&mut self) {
        self.lifecycle.uncertain().await;
        self.finalized = true;
    }
}

impl Drop for StreamSessionGuard {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let lifecycle = self.lifecycle.clone();
        tokio::spawn(async move {
            lifecycle.uncertain().await;
        });
    }
}

pub struct ProtocolSessionStore {
    path: Option<PathBuf>,
    index: Mutex<SessionIndex>,
    locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
}

impl ProtocolSessionStore {
    pub fn memory() -> Arc<Self> {
        Arc::new(Self {
            path: None,
            index: Mutex::new(SessionIndex {
                sessions: HashMap::new(),
            }),
            locks: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub async fn persistent(path: impl Into<PathBuf>) -> Result<Arc<Self>, ProtocolError> {
        let path = path.into();
        let sessions = load_sessions(&path).await?;
        Ok(Arc::new(Self {
            path: Some(path),
            index: Mutex::new(SessionIndex { sessions }),
            locks: Arc::new(Mutex::new(HashMap::new())),
        }))
    }

    pub async fn try_begin(
        &self,
        principal: &AuthPrincipal,
        session_digest: &str,
    ) -> Result<SessionOperation, ProtocolError> {
        let key = session_ref(&principal.0, session_digest);
        let lock = {
            let mut locks = self.locks.lock().await;
            locks
                .entry(key.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let guard = lock.try_lock_owned().map_err(|_| {
            ProtocolError::new(
                StatusCode::CONFLICT,
                "session_busy",
                "Another operation is already using this session",
            )
        })?;
        Ok(SessionOperation {
            key,
            session_digest: session_digest.to_owned(),
            _guard: guard,
            lock_registry: self.locks.clone(),
        })
    }

    pub async fn get(&self, operation: &SessionOperation) -> Option<ProtocolSession> {
        self.index
            .lock()
            .await
            .sessions
            .get(&operation.key)
            .cloned()
    }

    pub async fn plan(
        &self,
        operation: &SessionOperation,
        user_digests: &[String],
        assistant_digests: &[String],
        model_digest: &str,
        system_digest: &str,
    ) -> Result<ReusePlan, ProtocolError> {
        let index = self.index.lock().await;
        let Some(session) = index.sessions.get(&operation.key) else {
            return Ok(ReusePlan::Create);
        };
        match session.state {
            SessionState::Uncertain | SessionState::Provisioning | SessionState::InFlight => {
                return Err(ProtocolError::new(
                    StatusCode::CONFLICT,
                    "conversation_state_uncertain",
                    "The previous session operation did not reach a verified terminal state",
                ));
            }
            SessionState::Tombstoned => {
                return Err(ProtocolError::new(
                    StatusCode::GONE,
                    "conversation_expired",
                    "The upstream conversation has expired and must be reset",
                ));
            }
            SessionState::Committed => {}
        }
        if session.model_digest != model_digest || session.system_digest != system_digest {
            return Err(reuse_failed("Model or system prompt changed"));
        }
        let plan = plan_committed_turns(&session.turns, user_digests)?;
        validate_assistant_context(&session.turns, assistant_digests, &plan)?;
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn create_provisional(
        &self,
        operation: &SessionOperation,
        principal: &AuthPrincipal,
        session_digest: &str,
        cookie_id: String,
        organization_uuid: String,
        conversation_uuid: String,
        model_digest: String,
        system_digest: String,
        pending: PendingTurn,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        enforce_capacity(&index.sessions, &principal.0)?;
        let now = Utc::now().timestamp();
        index.sessions.insert(
            operation.key.clone(),
            ProtocolSession {
                principal: principal.0.clone(),
                session_digest: session_digest.to_owned(),
                state: SessionState::Provisioning,
                cookie_id,
                organization_uuid,
                conversation_uuid,
                model_digest,
                system_digest,
                turns: Vec::new(),
                pending: Some(pending),
                file_mappings: HashMap::new(),
                created_at: now,
                last_used: now,
            },
        );
        self.persist_locked(&index).await
    }

    pub async fn start_existing(
        &self,
        operation: &SessionOperation,
        pending: PendingTurn,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let session = index
            .sessions
            .get_mut(&operation.key)
            .ok_or_else(|| reuse_failed("Session disappeared before reuse"))?;
        if session.state != SessionState::Committed {
            return Err(reuse_failed("Session is not committed"));
        }
        session.state = SessionState::InFlight;
        session.pending = Some(pending);
        session.last_used = Utc::now().timestamp();
        self.persist_locked(&index).await
    }

    pub async fn put_file_mapping(
        &self,
        operation: &SessionOperation,
        staged_file_id: &str,
        claude_file_uuid: &str,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let session = index
            .sessions
            .get_mut(&operation.key)
            .ok_or_else(|| reuse_failed("Session disappeared while uploading a file"))?;
        session
            .file_mappings
            .insert(staged_file_id.to_owned(), claude_file_uuid.to_owned());
        self.persist_locked(&index).await
    }

    pub async fn file_mapping(
        &self,
        operation: &SessionOperation,
        staged_file_id: &str,
    ) -> Option<String> {
        self.index
            .lock()
            .await
            .sessions
            .get(&operation.key)
            .and_then(|session| session.file_mappings.get(staged_file_id))
            .cloned()
    }

    pub async fn commit(
        &self,
        operation: &SessionOperation,
        assistant_digest_after: Option<String>,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let session = index
            .sessions
            .get_mut(&operation.key)
            .ok_or_else(|| reuse_failed("Session disappeared before commit"))?;
        let pending = session
            .pending
            .take()
            .ok_or_else(|| reuse_failed("Session has no pending turn"))?;
        session.turns.truncate(pending.replace_from_turn);
        session.turns.push(SessionTurn {
            parent_uuid_before: pending.parent_uuid_before,
            user_digests: pending.user_digests,
            assistant_uuid_after: pending.assistant_uuid_after,
            assistant_digests_before: pending.assistant_digests_before,
            assistant_digest_after,
        });
        session.state = SessionState::Committed;
        session.last_used = Utc::now().timestamp();
        self.persist_locked(&index).await
    }

    pub async fn mark_uncertain(&self, operation: &SessionOperation) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        if let Some(session) = index.sessions.get_mut(&operation.key) {
            session.state = SessionState::Uncertain;
            session.last_used = Utc::now().timestamp();
            self.persist_locked(&index).await?;
        }
        Ok(())
    }

    pub async fn restore_committed_before_completion(
        &self,
        operation: &SessionOperation,
    ) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        let session = index
            .sessions
            .get_mut(&operation.key)
            .ok_or_else(|| reuse_failed("Session disappeared before completion"))?;
        session.state = SessionState::Committed;
        session.pending = None;
        session.last_used = Utc::now().timestamp();
        self.persist_locked(&index).await
    }

    pub async fn tombstone(&self, operation: &SessionOperation) -> Result<(), ProtocolError> {
        let mut index = self.index.lock().await;
        if let Some(session) = index.sessions.get_mut(&operation.key) {
            session.state = SessionState::Tombstoned;
            session.pending = None;
            session.last_used = Utc::now().timestamp();
            self.persist_locked(&index).await?;
        }
        Ok(())
    }

    pub async fn reset(
        &self,
        operation: &SessionOperation,
        principal: &AuthPrincipal,
    ) -> Result<ProtocolSession, ProtocolError> {
        let mut index = self.index.lock().await;
        let session = match index.sessions.get(&operation.key) {
            Some(session) => session,
            None if index.sessions.values().any(|session| {
                session.session_digest == operation.session_digest
                    && session.principal != principal.0
            }) =>
            {
                return Err(ProtocolError::new(
                    StatusCode::FORBIDDEN,
                    "session_forbidden",
                    "Session belongs to another authenticated principal",
                ));
            }
            None => {
                return Err(ProtocolError::new(
                    StatusCode::NOT_FOUND,
                    "session_not_found",
                    "Session does not exist",
                ));
            }
        };
        if session.principal != principal.0 {
            return Err(ProtocolError::new(
                StatusCode::FORBIDDEN,
                "session_forbidden",
                "Session belongs to another authenticated principal",
            ));
        }
        let removed = index
            .sessions
            .remove(&operation.key)
            .expect("checked above");
        self.persist_locked(&index).await?;
        Ok(removed)
    }

    pub async fn cleanup_tombstones(&self) -> Result<(), ProtocolError> {
        let cutoff = Utc::now().timestamp() - TOMBSTONE_TTL_SECONDS;
        let candidates = {
            let index = self.index.lock().await;
            index
                .sessions
                .iter()
                .filter(|(_, session)| {
                    session.state == SessionState::Tombstoned && session.last_used < cutoff
                })
                .map(|(key, session)| (key.clone(), session.session_digest.clone()))
                .collect::<Vec<_>>()
        };
        for (key, session_digest) in candidates {
            let lock = {
                let mut locks = self.locks.lock().await;
                locks
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(())))
                    .clone()
            };
            let operation = SessionOperation {
                key: key.clone(),
                session_digest,
                _guard: lock.lock_owned().await,
                lock_registry: self.locks.clone(),
            };
            let mut index = self.index.lock().await;
            if index.sessions.get(&operation.key).is_some_and(|session| {
                session.state == SessionState::Tombstoned && session.last_used < cutoff
            }) {
                index.sessions.remove(&operation.key);
            }
        }
        let index = self.index.lock().await;
        self.persist_locked(&index).await
    }

    pub async fn staged_file_references(&self) -> HashMap<String, BTreeSet<String>> {
        let index = self.index.lock().await;
        let mut references = HashMap::<String, BTreeSet<String>>::new();
        for session in index.sessions.values() {
            let session_ref = session.session_ref();
            for file_id in session.file_mappings.keys() {
                references
                    .entry(file_id.clone())
                    .or_default()
                    .insert(session_ref.clone());
            }
        }
        references
    }

    async fn persist_locked(&self, index: &SessionIndex) -> Result<(), ProtocolError> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(session_io_error)?;
        }
        let snapshot = PersistedSessions {
            version: SESSION_FILE_VERSION,
            sessions: index.sessions.values().cloned().collect(),
        };
        let data = serde_json::to_vec_pretty(&snapshot).map_err(|error| {
            ProtocolError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "session_storage_unavailable",
                format!("Failed to serialize session state: {error}"),
            )
        })?;
        let temp = path.with_extension("json.tmp");
        tokio::fs::write(&temp, data)
            .await
            .map_err(session_io_error)?;
        set_owner_only_session_file(&temp).await?;
        tokio::fs::rename(temp, path)
            .await
            .map_err(session_io_error)
    }
}

pub fn digest_json(value: &serde_json::Value) -> String {
    let canonical = serde_json::to_vec(value).expect("JSON values serialize");
    hex::encode(Sha256::digest(canonical))
}

pub fn digest_model(model: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"clewdr-model-v1\0");
    digest.update(model.as_bytes());
    hex::encode(digest.finalize())
}

pub fn digest_system(system: &Option<serde_json::Value>) -> String {
    digest_json(system.as_ref().unwrap_or(&serde_json::Value::Null))
}

pub fn digest_user_messages(messages: &[Message]) -> Vec<(usize, String)> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.role == Role::User)
        .filter_map(|(index, message)| {
            let content = match &message.content {
                MessageContent::Text { content } => serde_json::json!([{
                    "type": "text",
                    "text": content,
                }]),
                MessageContent::Blocks { content } => {
                    let relevant = content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text, .. } => {
                                Some(serde_json::json!({ "type": "text", "text": text }))
                            }
                            ContentBlock::Image { source, .. } => Some(serde_json::json!({
                                "type": "image",
                                "source": source,
                            })),
                            ContentBlock::Document {
                                source,
                                context,
                                title,
                                ..
                            } => Some(serde_json::json!({
                                "type": "document",
                                "source": source,
                                "context": context,
                                "title": title,
                            })),
                            ContentBlock::ContainerUpload { file_id, .. } => {
                                Some(serde_json::json!({
                                    "type": "container_upload",
                                    "file_id": file_id,
                                }))
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if relevant.is_empty() {
                        return None;
                    }
                    serde_json::Value::Array(relevant)
                }
            };
            Some((index, digest_json(&content)))
        })
        .collect()
}

pub fn digest_assistant_messages(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
        .filter_map(|message| {
            let text = match &message.content {
                MessageContent::Text { content } => content.clone(),
                MessageContent::Blocks { content } => content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            (!text.is_empty()).then(|| digest_assistant_output(&text))
        })
        .collect()
}

pub fn digest_assistant_output(text: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"clewdr-assistant-text-v1\0");
    digest.update(text.as_bytes());
    hex::encode(digest.finalize())
}

pub fn session_ref(principal: &str, session_digest: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(b"clewdr-session-key-v1\0");
    digest.update(principal.as_bytes());
    digest.update([0]);
    digest.update(session_digest.as_bytes());
    hex::encode(digest.finalize())
}

fn plan_committed_turns(
    turns: &[SessionTurn],
    requested: &[String],
) -> Result<ReusePlan, ProtocolError> {
    if turns.is_empty() {
        return Err(reuse_failed("Committed session has no turns"));
    }
    let committed = turns
        .iter()
        .flat_map(|turn| turn.user_digests.iter().cloned())
        .collect::<Vec<_>>();
    if requested.len() < committed.len() {
        return Err(reuse_failed("Request is shorter than committed history"));
    }
    if requested.starts_with(&committed) {
        if requested.len() > committed.len() {
            return Ok(ReusePlan::Append {
                parent_uuid: turns.last().unwrap().assistant_uuid_after.clone(),
                suffix_start: committed.len(),
            });
        }
        let last = turns.last().unwrap();
        let suffix_start = committed.len() - last.user_digests.len();
        return Ok(ReusePlan::Regenerate {
            parent_uuid: last.parent_uuid_before.clone(),
            suffix_start,
            replace_from_turn: turns.len() - 1,
        });
    }

    let mut offset = 0usize;
    for (turn_index, turn) in turns.iter().enumerate() {
        let end = offset + turn.user_digests.len();
        let matches_turn = requested
            .get(offset..end)
            .is_some_and(|candidate| candidate == turn.user_digests);
        if !matches_turn {
            if turn_index == 0 {
                return Err(reuse_failed("Cannot fork inside bootstrap history"));
            }
            return Ok(ReusePlan::Fork {
                parent_uuid: turn.parent_uuid_before.clone(),
                suffix_start: offset,
                replace_from_turn: turn_index,
            });
        }
        offset = end;
    }
    Err(reuse_failed(
        "Request history is not aligned to committed turns",
    ))
}

fn validate_assistant_context(
    turns: &[SessionTurn],
    requested: &[String],
    plan: &ReusePlan,
) -> Result<(), ProtocolError> {
    let expected = match plan {
        ReusePlan::Create => return Ok(()),
        ReusePlan::Append { .. } => {
            let latest = turns.last().expect("append requires a committed turn");
            let mut expected = latest.assistant_digests_before.clone();
            expected.extend(latest.assistant_digest_after.iter().cloned());
            expected
        }
        ReusePlan::Fork {
            replace_from_turn, ..
        }
        | ReusePlan::Regenerate {
            replace_from_turn, ..
        } => turns
            .get(*replace_from_turn)
            .expect("reuse plan references a committed turn")
            .assistant_digests_before
            .clone(),
    };
    if requested != expected {
        return Err(reuse_failed(
            "Assistant history or prefill does not match the selected conversation parent",
        ));
    }
    Ok(())
}

fn enforce_capacity(
    sessions: &HashMap<String, ProtocolSession>,
    principal: &str,
) -> Result<(), ProtocolError> {
    let total = sessions
        .values()
        .filter(|session| session.principal == principal)
        .count();
    let live = sessions
        .values()
        .filter(|session| session.principal == principal && session.state.is_live())
        .count();
    if total >= MAX_SESSION_RECORDS_PER_PRINCIPAL || live >= MAX_LIVE_SESSIONS_PER_PRINCIPAL {
        return Err(ProtocolError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "session_capacity_exceeded",
            "Authenticated principal has reached the session capacity limit",
        ));
    }
    Ok(())
}

async fn load_sessions(path: &Path) -> Result<HashMap<String, ProtocolSession>, ProtocolError> {
    let data = match tokio::fs::read(path).await {
        Ok(data) => data,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(error) => return Err(session_io_error(error)),
    };
    let persisted: PersistedSessions = serde_json::from_slice(&data).map_err(|error| {
        ProtocolError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_storage_unavailable",
            format!("Failed to parse session state: {error}"),
        )
    })?;
    if persisted.version != SESSION_FILE_VERSION {
        return Err(ProtocolError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "session_storage_unavailable",
            "Unsupported session state version",
        ));
    }
    Ok(persisted
        .sessions
        .into_iter()
        .map(|session| (session.session_ref(), session))
        .collect())
}

fn reuse_failed(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(StatusCode::CONFLICT, "conversation_reuse_failed", message)
}

fn session_io_error(error: std::io::Error) -> ProtocolError {
    ProtocolError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "session_storage_unavailable",
        format!("Session storage error: {error}"),
    )
}

#[cfg(unix)]
async fn set_owner_only_session_file(path: &Path) -> Result<(), ProtocolError> {
    use std::os::unix::fs::PermissionsExt;

    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .await
        .map_err(session_io_error)
}

#[cfg(not(unix))]
async fn set_owner_only_session_file(_path: &Path) -> Result<(), ProtocolError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures::stream;

    use super::*;
    use crate::protocol::files::StagedFileStore;

    fn turn(parent: Option<&str>, users: &[&str], assistant: &str) -> SessionTurn {
        SessionTurn {
            parent_uuid_before: parent.map(str::to_owned),
            user_digests: users.iter().map(|value| (*value).to_owned()).collect(),
            assistant_uuid_after: assistant.to_owned(),
            assistant_digests_before: vec![],
            assistant_digest_after: None,
        }
    }

    #[test]
    fn plans_append_fork_and_regeneration_at_committed_boundaries() {
        let turns = vec![
            turn(None, &["u1", "u2"], "a1"),
            turn(Some("a1"), &["u3"], "a2"),
        ];
        assert_eq!(
            plan_committed_turns(
                &turns,
                &["u1".into(), "u2".into(), "u3".into(), "u4".into()]
            )
            .unwrap(),
            ReusePlan::Append {
                parent_uuid: "a2".into(),
                suffix_start: 3
            }
        );
        assert_eq!(
            plan_committed_turns(&turns, &["u1".into(), "u2".into(), "edited".into()]).unwrap(),
            ReusePlan::Fork {
                parent_uuid: Some("a1".into()),
                suffix_start: 2,
                replace_from_turn: 1
            }
        );
        assert_eq!(
            plan_committed_turns(&turns, &["u1".into(), "u2".into(), "u3".into()]).unwrap(),
            ReusePlan::Regenerate {
                parent_uuid: Some("a1".into()),
                suffix_start: 2,
                replace_from_turn: 1
            }
        );
    }

    #[test]
    fn pending_turn_without_assistant_digests_loads_from_v1_metadata() {
        let pending: PendingTurn = serde_json::from_value(serde_json::json!({
            "parent_uuid_before": null,
            "user_digests": ["u1"],
            "assistant_uuid_after": "a1",
            "replace_from_turn": 0
        }))
        .unwrap();

        assert!(pending.assistant_digests_before.is_empty());
    }

    #[test]
    fn bootstrap_edits_and_short_history_fail_closed() {
        let turns = vec![turn(None, &["u1", "u2"], "a1")];
        assert_eq!(
            plan_committed_turns(&turns, &["u1".into(), "edited".into()])
                .unwrap_err()
                .code,
            "conversation_reuse_failed"
        );
        assert_eq!(
            plan_committed_turns(&turns, &["u1".into()])
                .unwrap_err()
                .code,
            "conversation_reuse_failed"
        );
    }

    #[tokio::test]
    async fn uncertain_state_persists_until_explicit_reset() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("sessions.json");
        let principal = AuthPrincipal::for_authenticated_user();
        let store = ProtocolSessionStore::persistent(&path).await.unwrap();
        let operation = store.try_begin(&principal, &"ab".repeat(32)).await.unwrap();
        store
            .create_provisional(
                &operation,
                &principal,
                &"ab".repeat(32),
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u1".into()],
                    assistant_uuid_after: "a1".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        store.mark_uncertain(&operation).await.unwrap();
        drop(operation);
        drop(store);

        let store = ProtocolSessionStore::persistent(&path).await.unwrap();
        let operation = store.try_begin(&principal, &"ab".repeat(32)).await.unwrap();
        assert_eq!(
            store
                .plan(&operation, &["u1".into()], &[], "model", "system")
                .await
                .unwrap_err()
                .code,
            "conversation_state_uncertain"
        );
        let removed = store.reset(&operation, &principal).await.unwrap();
        assert_eq!(removed.conversation_uuid, "conv");
    }

    #[tokio::test]
    async fn one_session_allows_only_one_operation() {
        let store = ProtocolSessionStore::memory();
        let principal = AuthPrincipal::for_authenticated_user();
        let first = store.try_begin(&principal, &"ab".repeat(32)).await.unwrap();
        assert_eq!(
            store
                .try_begin(&principal, &"ab".repeat(32))
                .await
                .unwrap_err()
                .code,
            "session_busy"
        );
        drop(first);
        store.try_begin(&principal, &"ab".repeat(32)).await.unwrap();
    }

    #[test]
    fn capacity_counts_live_records_and_unexpired_tombstones() {
        let principal = AuthPrincipal::test_principal("owner");
        let mut sessions = HashMap::new();
        for index in 0..MAX_LIVE_SESSIONS_PER_PRINCIPAL {
            let digest = format!("{index:064x}");
            let session = ProtocolSession {
                principal: principal.0.clone(),
                session_digest: digest.clone(),
                state: SessionState::Committed,
                cookie_id: "cookie".into(),
                organization_uuid: "org".into(),
                conversation_uuid: format!("conversation-{index}"),
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: vec![],
                pending: None,
                file_mappings: HashMap::new(),
                created_at: Utc::now().timestamp(),
                last_used: Utc::now().timestamp(),
            };
            sessions.insert(session.session_ref(), session);
        }
        assert_eq!(
            enforce_capacity(&sessions, &principal.0).unwrap_err().code,
            "session_capacity_exceeded"
        );
        for session in sessions.values_mut() {
            session.state = SessionState::Tombstoned;
        }
        assert!(enforce_capacity(&sessions, &principal.0).is_ok());
        for index in MAX_LIVE_SESSIONS_PER_PRINCIPAL..MAX_SESSION_RECORDS_PER_PRINCIPAL {
            let digest = format!("{index:064x}");
            let session = ProtocolSession {
                principal: principal.0.clone(),
                session_digest: digest,
                state: SessionState::Tombstoned,
                cookie_id: "cookie".into(),
                organization_uuid: "org".into(),
                conversation_uuid: format!("conversation-{index}"),
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: vec![],
                pending: None,
                file_mappings: HashMap::new(),
                created_at: Utc::now().timestamp(),
                last_used: Utc::now().timestamp(),
            };
            sessions.insert(session.session_ref(), session);
        }
        assert_eq!(
            enforce_capacity(&sessions, &principal.0).unwrap_err().code,
            "session_capacity_exceeded"
        );
    }

    #[tokio::test]
    async fn tombstone_is_gone_until_reset_and_other_principal_is_forbidden() {
        let store = ProtocolSessionStore::memory();
        let owner = AuthPrincipal::test_principal("owner");
        let digest = "ef".repeat(32);
        let operation = store.try_begin(&owner, &digest).await.unwrap();
        store
            .create_provisional(
                &operation,
                &owner,
                &digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u".into()],
                    assistant_uuid_after: "a".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        store.tombstone(&operation).await.unwrap();
        assert_eq!(
            store
                .plan(&operation, &["u".into()], &[], "model", "system")
                .await
                .unwrap_err()
                .code,
            "conversation_expired"
        );
        drop(operation);

        let other = AuthPrincipal::test_principal("other");
        let other_operation = store.try_begin(&other, &digest).await.unwrap();
        assert_eq!(
            store
                .reset(&other_operation, &other)
                .await
                .unwrap_err()
                .code,
            "session_forbidden"
        );
        drop(other_operation);

        let operation = store.try_begin(&owner, &digest).await.unwrap();
        store.reset(&operation, &owner).await.unwrap();
        drop(operation);
        let operation = store.try_begin(&owner, &digest).await.unwrap();
        assert_eq!(
            store
                .plan(&operation, &["u".into()], &[], "model", "system")
                .await
                .unwrap(),
            ReusePlan::Create
        );
    }

    #[tokio::test]
    async fn pre_completion_failure_restores_the_last_committed_parent() {
        let store = ProtocolSessionStore::memory();
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "fa".repeat(32);
        let operation = store.try_begin(&principal, &digest).await.unwrap();
        store
            .create_provisional(
                &operation,
                &principal,
                &digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u1".into()],
                    assistant_uuid_after: "a1".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        store.commit(&operation, None).await.unwrap();
        store
            .start_existing(
                &operation,
                PendingTurn {
                    parent_uuid_before: Some("a1".into()),
                    user_digests: vec!["u2".into()],
                    assistant_uuid_after: "a2".into(),
                    replace_from_turn: 1,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        store
            .restore_committed_before_completion(&operation)
            .await
            .unwrap();
        assert_eq!(
            store
                .plan(
                    &operation,
                    &["u1".into(), "u2".into()],
                    &[],
                    "model",
                    "system",
                )
                .await
                .unwrap(),
            ReusePlan::Append {
                parent_uuid: "a1".into(),
                suffix_start: 1,
            }
        );
    }

    #[tokio::test]
    async fn assistant_history_and_prefill_must_match_the_selected_parent() {
        let store = ProtocolSessionStore::memory();
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "fb".repeat(32);
        let prefill = digest_assistant_output("existing prefill");
        let generated = digest_assistant_output("generated answer");
        let operation = store.try_begin(&principal, &digest).await.unwrap();
        store
            .create_provisional(
                &operation,
                &principal,
                &digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u1".into()],
                    assistant_uuid_after: "a1".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![prefill.clone()],
                },
            )
            .await
            .unwrap();
        store
            .commit(&operation, Some(generated.clone()))
            .await
            .unwrap();

        assert!(matches!(
            store
                .plan(
                    &operation,
                    &["u1".into(), "u2".into()],
                    &[prefill.clone(), generated.clone()],
                    "model",
                    "system",
                )
                .await
                .unwrap(),
            ReusePlan::Append { .. }
        ));
        assert_eq!(
            store
                .plan(
                    &operation,
                    &["u1".into(), "u2".into()],
                    &[prefill.clone(), digest_assistant_output("changed answer")],
                    "model",
                    "system",
                )
                .await
                .unwrap_err()
                .code,
            "conversation_reuse_failed"
        );
        assert!(matches!(
            store
                .plan(
                    &operation,
                    &["u1".into()],
                    std::slice::from_ref(&prefill),
                    "model",
                    "system",
                )
                .await
                .unwrap(),
            ReusePlan::Regenerate { .. }
        ));
        assert_eq!(
            store
                .plan(
                    &operation,
                    &["u1".into()],
                    &[digest_assistant_output("changed prefill")],
                    "model",
                    "system",
                )
                .await
                .unwrap_err()
                .code,
            "conversation_reuse_failed"
        );
    }

    #[test]
    fn assistant_digest_preserves_text_block_boundaries() {
        let messages = [Message::new_blocks(
            Role::Assistant,
            vec![ContentBlock::text("ab"), ContentBlock::text("c")],
        )];
        let regrouped = [Message::new_blocks(
            Role::Assistant,
            vec![ContentBlock::text("a"), ContentBlock::text("bc")],
        )];

        assert_ne!(
            digest_assistant_messages(&messages),
            digest_assistant_messages(&regrouped)
        );
    }

    #[tokio::test]
    async fn tombstone_cleanup_waits_for_the_active_session_operation() {
        let store = ProtocolSessionStore::memory();
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "fd".repeat(32);
        let operation = store.try_begin(&principal, &digest).await.unwrap();
        store
            .create_provisional(
                &operation,
                &principal,
                &digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u".into()],
                    assistant_uuid_after: "a".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        store.tombstone(&operation).await.unwrap();
        {
            let mut index = store.index.lock().await;
            index.sessions.get_mut(&operation.key).unwrap().last_used =
                Utc::now().timestamp() - TOMBSTONE_TTL_SECONDS - 1;
        }

        let cleanup_store = store.clone();
        let mut cleanup = tokio::spawn(async move { cleanup_store.cleanup_tombstones().await });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut cleanup)
                .await
                .is_err()
        );
        assert!(store.get(&operation).await.is_some());

        drop(operation);
        cleanup.await.unwrap().unwrap();
        let operation = store.try_begin(&principal, &digest).await.unwrap();
        assert!(store.get(&operation).await.is_none());
    }

    #[tokio::test]
    async fn tombstone_ttl_cleanup_reconciles_file_refs_for_quota_recovery() {
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent_with_limits(temp.path().join("files"), 4, 4)
            .await
            .unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let first = files
            .stage_stream(
                &principal,
                "first.bin",
                "application/octet-stream",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"1234"))]),
            )
            .await
            .unwrap();
        let sessions = ProtocolSessionStore::memory();
        let digest = "fc".repeat(32);
        let operation = sessions.try_begin(&principal, &digest).await.unwrap();
        sessions
            .create_provisional(
                &operation,
                &principal,
                &digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["u".into()],
                    assistant_uuid_after: "a".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        sessions
            .put_file_mapping(&operation, &first.id, "claude-file")
            .await
            .unwrap();
        let session_ref = sessions.get(&operation).await.unwrap().session_ref();
        files.add_reference(&first.id, &session_ref).await.unwrap();
        sessions.tombstone(&operation).await.unwrap();
        {
            let mut index = sessions.index.lock().await;
            index.sessions.get_mut(&operation.key).unwrap().last_used =
                Utc::now().timestamp() - TOMBSTONE_TTL_SECONDS - 1;
        }
        drop(operation);

        sessions.cleanup_tombstones().await.unwrap();
        files
            .reconcile_references(&sessions.staged_file_references().await)
            .await
            .unwrap();
        let second = files
            .stage_stream(
                &principal,
                "second.bin",
                "application/octet-stream",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"5678"))]),
            )
            .await
            .unwrap();
        assert_ne!(first.id, second.id);
        assert_eq!(
            files.resolve(&principal, &first.id).await.unwrap_err().code,
            "file_not_found"
        );
    }
}
