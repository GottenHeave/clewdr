use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_with::{TimestampSecondsWithFrac, serde_as};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::warn;

use super::explicit_session::{
    ExplicitConversation, ExplicitSessionState, ExplicitTurn, PendingExplicitTurn,
};
use crate::protocol::ProtocolError;
use crate::utils::write_json_atomically;

const CACHE_FILE_VERSION: u32 = 1;
const MAX_SESSION_RECORDS_PER_PRINCIPAL: usize = 4096;
const MAX_LIVE_SESSIONS_PER_PRINCIPAL: usize = 1024;

fn explicit_missing(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(
        http::StatusCode::CONFLICT,
        "conversation_reuse_failed",
        message,
    )
}

fn storage_error(error: impl std::fmt::Display) -> ProtocolError {
    ProtocolError::new(
        http::StatusCode::INTERNAL_SERVER_ERROR,
        "session_storage_unavailable",
        format!("Session storage error: {error}"),
    )
}

fn restore_record(
    map: &mut HashMap<StoredCacheKey, CachedConversation>,
    key: StoredCacheKey,
    previous: Option<CachedConversation>,
) {
    match previous {
        Some(previous) => {
            map.insert(key, previous);
        }
        None => {
            map.remove(&key);
        }
    }
}

/// Represents one round-trip (ClewdR request → Claude response) in a cached conversation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedTurn {
    /// Hash of each Role::User message's text content sent in this turn.
    /// Turn 0 (full paste) may contain multiple user hashes.
    /// Subsequent turns typically contain 1+ user hashes (bundled).
    pub user_hashes: Vec<u64>,
    /// The assistant_message_uuid from turn_message_uuids.
    /// Used as `parent_message_uuid` for the next turn.
    pub assistant_uuid: String,
}

/// A cached conversation that can be reused across requests
#[derive(Clone, Debug)]
pub struct CachedConversation {
    /// Claude.ai conversation UUID
    pub conv_uuid: String,
    /// Organization UUID (must match)
    pub org_uuid: String,
    /// Cookie identifier string (must match — different cookie = different account)
    pub cookie_id: String,
    /// Model used (must match)
    pub model: String,
    /// Whether the account was pro when conversation was created
    pub is_pro: bool,
    /// Hash of the system prompt (system change → full rebuild)
    pub system_hash: u64,
    /// Ordered list of completed turns
    pub turns: Vec<CachedTurn>,
    /// When this conversation was first created
    pub created_at: DateTime<Utc>,
    /// Last time this conversation was successfully used
    pub last_used: DateTime<Utc>,
    /// Whether cache is currently valid (set to false on stream errors)
    pub valid: bool,
    /// Shared flag set to true when the SSE stream completes with a stop signal.
    /// Checked on next reuse; if still false, the previous stream was incomplete.
    pub last_stream_healthy: Arc<AtomicBool>,
    /// Strict client-managed session state. Legacy cache entries leave this unset.
    pub explicit: Option<ExplicitConversation>,
}

impl CachedConversation {
    /// Check if this cached conversation has expired (conservative 25-day TTL)
    pub fn is_expired(&self) -> bool {
        Utc::now() - self.created_at > Duration::days(25)
    }

    fn should_retain(&self) -> bool {
        match self.explicit.as_ref().map(|explicit| explicit.state) {
            Some(ExplicitSessionState::Tombstoned) => {
                Utc::now() - self.last_used <= Duration::days(25)
            }
            Some(_) => true,
            None => self.valid && !self.is_expired(),
        }
    }

    /// Get the last assistant UUID (parent for next turn)
    pub fn last_parent_uuid(&self) -> Option<&str> {
        self.turns.last().map(|t| t.assistant_uuid.as_str())
    }

    /// Truncate turns from `from_index` onward (for fork scenarios)
    pub fn truncate_turns(&mut self, from_index: usize) {
        self.turns.truncate(from_index);
    }
}

/// Cache key for an implicit request family.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub key_index: usize,
    pub request_fingerprint: u64,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplicitSessionKey {
    session_principal: String,
    session_digest: String,
}

impl ExplicitSessionKey {
    pub fn new(session_principal: impl Into<String>, session_digest: impl Into<String>) -> Self {
        Self {
            session_principal: session_principal.into(),
            session_digest: session_digest.into(),
        }
    }

    pub fn session_ref(&self) -> String {
        format!("{}:{}", self.session_principal, self.session_digest)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum StoredCacheKey {
    Legacy(CacheKey),
    ExplicitSession(ExplicitSessionKey),
}

#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedConversation {
    conv_uuid: String,
    org_uuid: String,
    cookie_id: String,
    model: String,
    is_pro: bool,
    system_hash: u64,
    turns: Vec<CachedTurn>,
    #[serde_as(as = "TimestampSecondsWithFrac")]
    created_at: DateTime<Utc>,
    #[serde_as(as = "TimestampSecondsWithFrac")]
    last_used: DateTime<Utc>,
    valid: bool,
    last_stream_healthy: bool,
    #[serde(default)]
    explicit: Option<ExplicitConversation>,
}

impl From<&CachedConversation> for PersistedConversation {
    fn from(conv: &CachedConversation) -> Self {
        Self {
            conv_uuid: conv.conv_uuid.clone(),
            org_uuid: conv.org_uuid.clone(),
            cookie_id: conv.cookie_id.clone(),
            model: conv.model.clone(),
            is_pro: conv.is_pro,
            system_hash: conv.system_hash,
            turns: conv.turns.clone(),
            created_at: conv.created_at,
            last_used: conv.last_used,
            valid: conv.valid,
            last_stream_healthy: conv.last_stream_healthy.load(Ordering::Relaxed),
            explicit: conv.explicit.clone(),
        }
    }
}

impl From<PersistedConversation> for CachedConversation {
    fn from(conv: PersistedConversation) -> Self {
        Self {
            conv_uuid: conv.conv_uuid,
            org_uuid: conv.org_uuid,
            cookie_id: conv.cookie_id,
            model: conv.model,
            is_pro: conv.is_pro,
            system_hash: conv.system_hash,
            turns: conv.turns,
            created_at: conv.created_at,
            last_used: conv.last_used,
            valid: conv.valid,
            last_stream_healthy: Arc::new(AtomicBool::new(conv.last_stream_healthy)),
            explicit: conv.explicit,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCacheEntry {
    key: StoredCacheKey,
    conversation: PersistedConversation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedConversationCache {
    version: u32,
    conversations: Vec<PersistedCacheEntry>,
}

impl PersistedConversationCache {
    fn from_map(map: &HashMap<StoredCacheKey, CachedConversation>) -> Self {
        Self {
            version: CACHE_FILE_VERSION,
            conversations: map
                .iter()
                .map(|(key, conversation)| PersistedCacheEntry {
                    key: key.clone(),
                    conversation: conversation.into(),
                })
                .collect(),
        }
    }
}

fn enforce_explicit_capacity(
    map: &HashMap<StoredCacheKey, CachedConversation>,
    principal: &str,
) -> Result<(), ProtocolError> {
    let sessions = map.iter().filter_map(|(key, conversation)| match key {
        StoredCacheKey::ExplicitSession(key) if key.session_principal == principal => {
            Some(conversation)
        }
        _ => None,
    });
    let sessions = sessions.collect::<Vec<_>>();
    let live = sessions
        .iter()
        .filter(|conversation| {
            !matches!(
                conversation
                    .explicit
                    .as_ref()
                    .map(|explicit| explicit.state),
                Some(ExplicitSessionState::Tombstoned)
            )
        })
        .count();
    if sessions.len() >= MAX_SESSION_RECORDS_PER_PRINCIPAL
        || live >= MAX_LIVE_SESSIONS_PER_PRINCIPAL
    {
        return Err(ProtocolError::new(
            http::StatusCode::SERVICE_UNAVAILABLE,
            "session_capacity_exceeded",
            "Authenticated principal has reached the session capacity limit",
        ));
    }
    Ok(())
}

/// Thread-safe conversation cache
#[derive(Clone)]
pub struct ConversationCache {
    inner: Arc<Mutex<HashMap<StoredCacheKey, CachedConversation>>>,
    operation_locks: Arc<Mutex<HashMap<StoredCacheKey, Weak<Mutex<()>>>>>,
    persist_path: Option<Arc<PathBuf>>,
    persist_lock: Arc<Mutex<()>>,
    explicit_mutation_lock: Arc<Mutex<()>>,
    #[cfg(test)]
    persistence_attempts: Arc<AtomicUsize>,
}

impl ConversationCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            operation_locks: Arc::new(Mutex::new(HashMap::new())),
            persist_path: None,
            persist_lock: Arc::new(Mutex::new(())),
            explicit_mutation_lock: Arc::new(Mutex::new(())),
            #[cfg(test)]
            persistence_attempts: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Default for ConversationCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationCache {
    pub async fn persistent(path: impl Into<PathBuf>) -> Self {
        let persist_path = path.into();
        let inner = match Self::load_from_path(&persist_path).await {
            Ok(map) => map,
            Err(err) => {
                warn!(
                    "[CACHE] failed to load conversation cache from {}: {}",
                    persist_path.display(),
                    err
                );
                HashMap::new()
            }
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
            operation_locks: Arc::new(Mutex::new(HashMap::new())),
            persist_path: Some(Arc::new(persist_path)),
            persist_lock: Arc::new(Mutex::new(())),
            explicit_mutation_lock: Arc::new(Mutex::new(())),
            #[cfg(test)]
            persistence_attempts: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub async fn get(&self, key: &CacheKey) -> Option<CachedConversation> {
        self.get_stored(&StoredCacheKey::Legacy(key.clone())).await
    }

    pub async fn get_explicit(&self, key: &ExplicitSessionKey) -> Option<CachedConversation> {
        self.get_stored(&StoredCacheKey::ExplicitSession(key.clone()))
            .await
    }

    async fn get_stored(&self, key: &StoredCacheKey) -> Option<CachedConversation> {
        let map = self.inner.lock().await;
        map.get(key).filter(|c| c.should_retain()).cloned()
    }

    pub async fn lock_operation(&self, key: &CacheKey) -> OwnedMutexGuard<()> {
        self.lock_stored_operation(StoredCacheKey::Legacy(key.clone()))
            .await
    }

    pub async fn lock_explicit_operation(&self, key: &ExplicitSessionKey) -> OwnedMutexGuard<()> {
        self.lock_stored_operation(StoredCacheKey::ExplicitSession(key.clone()))
            .await
    }

    pub async fn try_lock_explicit_operation(
        &self,
        key: &ExplicitSessionKey,
    ) -> Result<OwnedMutexGuard<()>, ProtocolError> {
        let lock = self
            .stored_operation_lock(StoredCacheKey::ExplicitSession(key.clone()))
            .await;
        lock.try_lock_owned().map_err(|_| {
            ProtocolError::new(
                http::StatusCode::CONFLICT,
                "session_busy",
                "Another operation is already using this session",
            )
        })
    }

    async fn lock_stored_operation(&self, key: StoredCacheKey) -> OwnedMutexGuard<()> {
        self.stored_operation_lock(key).await.lock_owned().await
    }

    async fn stored_operation_lock(&self, key: StoredCacheKey) -> Arc<Mutex<()>> {
        {
            let mut locks = self.operation_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(key, Arc::downgrade(&lock));
                lock
            }
        }
    }

    pub async fn set(&self, key: CacheKey, conv: CachedConversation) {
        self.set_stored(StoredCacheKey::Legacy(key), conv).await;
    }

    pub async fn set_explicit(&self, key: ExplicitSessionKey, conv: CachedConversation) {
        self.set_stored(StoredCacheKey::ExplicitSession(key), conv)
            .await;
    }

    pub async fn set_explicit_checked(
        &self,
        key: ExplicitSessionKey,
        conv: CachedConversation,
    ) -> Result<(), ProtocolError> {
        self.mutate_explicit(&key, |map, stored_key| {
            if !map.contains_key(stored_key) {
                enforce_explicit_capacity(map, &key.session_principal)?;
            }
            map.insert(stored_key.clone(), conv);
            Ok(true)
        })
        .await?;
        Ok(())
    }

    async fn set_stored(&self, key: StoredCacheKey, conv: CachedConversation) {
        {
            let mut map = self.inner.lock().await;
            map.insert(key, conv);
        }
        self.persist().await;
    }

    pub async fn stage_explicit_turn(
        &self,
        key: &ExplicitSessionKey,
        pending: PendingExplicitTurn,
    ) -> Result<(), ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| {
            let conversation = map
                .get_mut(stored_key)
                .ok_or_else(|| explicit_missing("Session disappeared before completion"))?;
            let explicit = conversation
                .explicit
                .as_mut()
                .ok_or_else(|| explicit_missing("Session metadata is unavailable"))?;
            explicit.state = ExplicitSessionState::InFlight;
            explicit.pending = Some(pending);
            conversation.last_used = Utc::now();
            Ok(true)
        })
        .await?;
        Ok(())
    }

    pub async fn commit_explicit_turn(
        &self,
        key: &ExplicitSessionKey,
        assistant_digest_after: Option<String>,
    ) -> Result<(), ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| {
            let conversation = map
                .get_mut(stored_key)
                .ok_or_else(|| explicit_missing("Session disappeared before commit"))?;
            let explicit = conversation
                .explicit
                .as_mut()
                .ok_or_else(|| explicit_missing("Session metadata is unavailable"))?;
            let pending = explicit
                .pending
                .take()
                .ok_or_else(|| explicit_missing("Session has no pending turn"))?;
            conversation.turns.truncate(pending.replace_from_turn);
            conversation.turns.push(CachedTurn {
                user_hashes: Vec::new(),
                assistant_uuid: pending.assistant_uuid_after.clone(),
            });
            explicit.turns.truncate(pending.replace_from_turn);
            explicit.turns.push(ExplicitTurn {
                parent_uuid_before: pending.parent_uuid_before,
                user_digests: pending.user_digests,
                assistant_uuid_after: pending.assistant_uuid_after,
                parent_timeline: pending.parent_timeline,
                request_timeline: pending.request_timeline,
                assistant_digest_after,
            });
            explicit.state = ExplicitSessionState::Committed;
            conversation.last_used = Utc::now();
            Ok(true)
        })
        .await?;
        Ok(())
    }

    pub async fn mark_explicit_uncertain(
        &self,
        key: &ExplicitSessionKey,
    ) -> Result<(), ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| {
            let conversation = map
                .get_mut(stored_key)
                .ok_or_else(|| explicit_missing("Session disappeared before uncertain state"))?;
            let explicit = conversation
                .explicit
                .as_mut()
                .ok_or_else(|| explicit_missing("Session metadata is unavailable"))?;
            explicit.state = ExplicitSessionState::Uncertain;
            Ok(true)
        })
        .await?;
        Ok(())
    }

    pub async fn mark_explicit_uncertain_in_memory(&self, key: &ExplicitSessionKey) {
        let mut map = self.inner.lock().await;
        if let Some(explicit) = map
            .get_mut(&StoredCacheKey::ExplicitSession(key.clone()))
            .and_then(|conversation| conversation.explicit.as_mut())
        {
            explicit.state = ExplicitSessionState::Uncertain;
        }
    }

    pub async fn tombstone_explicit(&self, key: &ExplicitSessionKey) -> Result<(), ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| {
            let conversation = map
                .get_mut(stored_key)
                .ok_or_else(|| explicit_missing("Session disappeared before tombstone"))?;
            let explicit = conversation
                .explicit
                .as_mut()
                .ok_or_else(|| explicit_missing("Session metadata is unavailable"))?;
            explicit.state = ExplicitSessionState::Tombstoned;
            explicit.pending = None;
            conversation.last_used = Utc::now();
            Ok(true)
        })
        .await?;
        Ok(())
    }

    pub async fn reset_explicit(&self, key: &ExplicitSessionKey) -> Result<bool, ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| Ok(map.remove(stored_key).is_some()))
            .await
    }

    pub async fn explicit_file_mapping(
        &self,
        key: &ExplicitSessionKey,
        staged_file_id: &str,
    ) -> Option<String> {
        self.get_explicit(key)
            .await
            .and_then(|conversation| conversation.explicit)
            .and_then(|explicit| explicit.file_mappings.get(staged_file_id).cloned())
    }

    pub async fn put_explicit_file_mapping(
        &self,
        key: &ExplicitSessionKey,
        staged_file_id: &str,
        upstream_file_id: &str,
    ) -> Result<(), ProtocolError> {
        self.mutate_explicit(key, |map, stored_key| {
            let explicit = map
                .get_mut(stored_key)
                .and_then(|conversation| conversation.explicit.as_mut())
                .ok_or_else(|| explicit_missing("Session disappeared while uploading a file"))?;
            explicit
                .file_mappings
                .insert(staged_file_id.to_owned(), upstream_file_id.to_owned());
            Ok(true)
        })
        .await?;
        Ok(())
    }

    pub async fn existing_explicit_session_refs(&self) -> BTreeSet<String> {
        self.inner
            .lock()
            .await
            .keys()
            .filter_map(|key| match key {
                StoredCacheKey::ExplicitSession(key) => Some(key.session_ref()),
                StoredCacheKey::Legacy(_) => None,
            })
            .collect()
    }

    async fn mutate_explicit(
        &self,
        key: &ExplicitSessionKey,
        mutation: impl FnOnce(
            &mut HashMap<StoredCacheKey, CachedConversation>,
            &StoredCacheKey,
        ) -> Result<bool, ProtocolError>,
    ) -> Result<bool, ProtocolError> {
        let _mutation = self.explicit_mutation_lock.lock().await;
        let _persist = self.persist_lock.lock().await;
        let stored_key = StoredCacheKey::ExplicitSession(key.clone());
        let previous = {
            let mut map = self.inner.lock().await;
            let previous = map.get(&stored_key).cloned();
            match mutation(&mut map, &stored_key) {
                Ok(true) => previous,
                Ok(false) => return Ok(false),
                Err(error) => {
                    restore_record(&mut map, stored_key, previous);
                    return Err(error);
                }
            }
        };
        if let Err(error) = self.persist_snapshot().await {
            let mut map = self.inner.lock().await;
            restore_record(&mut map, stored_key, previous);
            return Err(storage_error(error));
        }
        Ok(true)
    }

    /// Append a new turn to an existing cached conversation
    pub async fn append_turn(&self, key: &CacheKey, turn: CachedTurn) {
        self.append_stored_turn(&StoredCacheKey::Legacy(key.clone()), turn)
            .await;
    }

    pub async fn append_explicit_turn(&self, key: &ExplicitSessionKey, turn: CachedTurn) {
        self.append_stored_turn(&StoredCacheKey::ExplicitSession(key.clone()), turn)
            .await;
    }

    async fn append_stored_turn(&self, key: &StoredCacheKey, turn: CachedTurn) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conv) = map.get_mut(key) {
                conv.turns.push(turn);
                conv.last_used = Utc::now();
                true
            } else {
                false
            }
        };
        if updated {
            self.persist().await;
        }
    }

    /// Truncate turns and append a new one (fork scenario)
    pub async fn fork_and_append(&self, key: &CacheKey, from_index: usize, turn: CachedTurn) {
        self.fork_and_append_stored(&StoredCacheKey::Legacy(key.clone()), from_index, turn)
            .await;
    }

    pub async fn fork_and_append_explicit(
        &self,
        key: &ExplicitSessionKey,
        from_index: usize,
        turn: CachedTurn,
    ) {
        self.fork_and_append_stored(
            &StoredCacheKey::ExplicitSession(key.clone()),
            from_index,
            turn,
        )
        .await;
    }

    async fn fork_and_append_stored(
        &self,
        key: &StoredCacheKey,
        from_index: usize,
        turn: CachedTurn,
    ) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conv) = map.get_mut(key) {
                conv.truncate_turns(from_index);
                conv.turns.push(turn);
                conv.last_used = Utc::now();
                true
            } else {
                false
            }
        };
        if updated {
            self.persist().await;
        }
    }

    /// Mark a cached conversation as invalid
    pub async fn invalidate(&self, key: &CacheKey) {
        self.invalidate_stored(&StoredCacheKey::Legacy(key.clone()))
            .await;
    }

    pub async fn invalidate_explicit(&self, key: &ExplicitSessionKey) {
        self.invalidate_stored(&StoredCacheKey::ExplicitSession(key.clone()))
            .await;
    }

    async fn invalidate_stored(&self, key: &StoredCacheKey) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conv) = map.get_mut(key) {
                conv.valid = false;
                true
            } else {
                false
            }
        };
        if updated {
            self.persist().await;
        }
    }

    /// Remove expired entries (call periodically)
    pub async fn cleanup(&self) {
        let _explicit_mutation = self.explicit_mutation_lock.lock().await;
        let removed = {
            let mut map = self.inner.lock().await;
            let old_len = map.len();
            map.retain(|_, conversation| conversation.should_retain());
            map.len() != old_len
        };
        if removed {
            self.persist().await;
        }
    }

    /// Invalidate all entries for a given cookie_id (cookie rotation)
    pub async fn invalidate_by_cookie(&self, cookie_id: &str) {
        let updated = {
            let mut map = self.inner.lock().await;
            let mut updated = false;
            for conv in map.values_mut() {
                if conv.cookie_id == cookie_id {
                    conv.valid = false;
                    updated = true;
                }
            }
            updated
        };
        if updated {
            self.persist().await;
        }
    }

    /// Update the stream health flag on an existing cached conversation
    pub async fn update_stream_health(&self, key: &CacheKey, flag: Arc<AtomicBool>) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conv) = map.get_mut(&StoredCacheKey::Legacy(key.clone())) {
                conv.last_stream_healthy = flag;
                true
            } else {
                false
            }
        };
        if updated {
            self.persist().await;
        }
    }

    /// Check if the last stream completed healthily for a given cache key
    pub async fn is_last_stream_healthy(&self, key: &CacheKey) -> bool {
        let map = self.inner.lock().await;
        map.get(&StoredCacheKey::Legacy(key.clone()))
            .map(|c| c.last_stream_healthy.load(Ordering::Relaxed))
            .unwrap_or(true)
    }

    pub async fn flush(&self) {
        self.persist().await;
    }

    async fn persist(&self) {
        if let Err(err) = self.persist_result().await {
            let Some(path) = self.persist_path.as_deref() else {
                return;
            };
            warn!(
                "[CACHE] failed to persist conversation cache to {}: {}",
                path.display(),
                err
            );
        }
    }

    async fn persist_result(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(path) = self.persist_path.as_deref() else {
            return Ok(());
        };
        let _guard = self.persist_lock.lock().await;
        self.persist_snapshot_to(path).await
    }

    async fn persist_snapshot(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let Some(path) = self.persist_path.as_deref() else {
            return Ok(());
        };
        self.persist_snapshot_to(path).await
    }

    async fn persist_snapshot_to(
        &self,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        #[cfg(test)]
        self.persistence_attempts.fetch_add(1, Ordering::Relaxed);
        let snapshot = {
            let map = self.inner.lock().await;
            PersistedConversationCache::from_map(&map)
        };
        write_json_atomically(path, &snapshot).await
    }

    async fn load_from_path(
        path: &Path,
    ) -> Result<HashMap<StoredCacheKey, CachedConversation>, Box<dyn std::error::Error + Send + Sync>>
    {
        let data = match tokio::fs::read_to_string(path).await {
            Ok(data) if data.trim().is_empty() => return Ok(HashMap::new()),
            Ok(data) => data,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(HashMap::new()),
            Err(err) => return Err(Box::new(err)),
        };
        let persisted: PersistedConversationCache = serde_json::from_str(&data)?;
        if persisted.version != CACHE_FILE_VERSION {
            warn!(
                "[CACHE] ignoring unsupported conversation cache version {} from {}",
                persisted.version,
                path.display()
            );
            return Ok(HashMap::new());
        }

        let mut map = HashMap::new();
        for entry in persisted.conversations {
            let conversation = CachedConversation::from(entry.conversation);
            if conversation.should_retain() {
                map.insert(entry.key, conversation);
            }
        }
        Ok(map)
    }
}

#[cfg(test)]
mod explicit_tests {
    use super::*;

    fn conversation(state: ExplicitSessionState) -> CachedConversation {
        CachedConversation {
            conv_uuid: "conversation".into(),
            org_uuid: "org".into(),
            cookie_id: "cookie".into(),
            model: "model".into(),
            is_pro: false,
            system_hash: 0,
            turns: Vec::new(),
            created_at: Utc::now(),
            last_used: Utc::now(),
            valid: true,
            last_stream_healthy: Arc::new(AtomicBool::new(true)),
            explicit: Some(ExplicitConversation {
                state,
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: Vec::new(),
                pending: None,
                file_mappings: Default::default(),
            }),
        }
    }

    fn pending() -> PendingExplicitTurn {
        PendingExplicitTurn {
            parent_uuid_before: None,
            user_digests: vec!["user".into()],
            assistant_uuid_after: "assistant".into(),
            replace_from_turn: 0,
            parent_timeline: Vec::new(),
            request_timeline: vec!["user:user".into()],
        }
    }

    async fn fill_sessions(cache: &ConversationCache, count: usize, state: ExplicitSessionState) {
        let mut map = cache.inner.lock().await;
        for index in 0..count {
            map.insert(
                StoredCacheKey::ExplicitSession(ExplicitSessionKey::new(
                    "principal",
                    index.to_string(),
                )),
                conversation(state),
            );
        }
    }

    #[tokio::test]
    async fn enforces_live_and_record_capacity_per_principal() {
        let cache = ConversationCache::new();
        fill_sessions(
            &cache,
            MAX_LIVE_SESSIONS_PER_PRINCIPAL,
            ExplicitSessionState::Committed,
        )
        .await;
        let error = cache
            .set_explicit_checked(
                ExplicitSessionKey::new("principal", "overflow"),
                conversation(ExplicitSessionState::InFlight),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, "session_capacity_exceeded");

        let cache = ConversationCache::new();
        fill_sessions(
            &cache,
            MAX_SESSION_RECORDS_PER_PRINCIPAL,
            ExplicitSessionState::Tombstoned,
        )
        .await;
        let error = cache
            .set_explicit_checked(
                ExplicitSessionKey::new("principal", "overflow"),
                conversation(ExplicitSessionState::InFlight),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code, "session_capacity_exceeded");
    }

    #[tokio::test]
    async fn cleanup_expires_only_old_tombstones() {
        let cache = ConversationCache::new();
        let tombstone = ExplicitSessionKey::new("principal", "tombstone");
        let uncertain = ExplicitSessionKey::new("principal", "uncertain");
        let mut old_tombstone = conversation(ExplicitSessionState::Tombstoned);
        old_tombstone.last_used = Utc::now() - Duration::days(26);
        let mut old_uncertain = conversation(ExplicitSessionState::Uncertain);
        old_uncertain.last_used = Utc::now() - Duration::days(26);
        cache.set_explicit(tombstone.clone(), old_tombstone).await;
        cache.set_explicit(uncertain.clone(), old_uncertain).await;
        cache.cleanup().await;
        assert!(cache.get_explicit(&tombstone).await.is_none());
        assert!(cache.get_explicit(&uncertain).await.is_some());
    }

    #[tokio::test]
    async fn explicit_mutations_return_storage_errors_and_roll_back() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"block").unwrap();
        let cache = ConversationCache::persistent(blocker.join("cache.json")).await;
        let key = ExplicitSessionKey::new("principal", "session");
        cache
            .set_explicit(key.clone(), conversation(ExplicitSessionState::Committed))
            .await;

        let new_key = ExplicitSessionKey::new("principal", "new");
        assert_eq!(
            cache
                .set_explicit_checked(
                    new_key.clone(),
                    conversation(ExplicitSessionState::InFlight)
                )
                .await
                .unwrap_err()
                .code,
            "session_storage_unavailable"
        );
        assert!(cache.get_explicit(&new_key).await.is_none());

        assert_eq!(
            cache
                .stage_explicit_turn(&key, pending())
                .await
                .unwrap_err()
                .code,
            "session_storage_unavailable"
        );
        assert_eq!(
            cache.mark_explicit_uncertain(&key).await.unwrap_err().code,
            "session_storage_unavailable"
        );
        assert_eq!(
            cache.tombstone_explicit(&key).await.unwrap_err().code,
            "session_storage_unavailable"
        );
        assert_eq!(
            cache.reset_explicit(&key).await.unwrap_err().code,
            "session_storage_unavailable"
        );
        assert_eq!(
            cache
                .get_explicit(&key)
                .await
                .unwrap()
                .explicit
                .unwrap()
                .state,
            ExplicitSessionState::Committed
        );

        let mut in_flight = conversation(ExplicitSessionState::InFlight);
        in_flight.explicit.as_mut().unwrap().pending = Some(pending());
        cache.set_explicit(key.clone(), in_flight).await;
        assert_eq!(
            cache
                .commit_explicit_turn(&key, Some("assistant".into()))
                .await
                .unwrap_err()
                .code,
            "session_storage_unavailable"
        );
        assert_eq!(
            cache
                .get_explicit(&key)
                .await
                .unwrap()
                .explicit
                .unwrap()
                .state,
            ExplicitSessionState::InFlight
        );
    }

    #[tokio::test]
    async fn reset_is_durable_across_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("conversation-cache.json");
        let key = ExplicitSessionKey::new("principal", "session");
        let cache = ConversationCache::persistent(&path).await;
        cache
            .set_explicit_checked(key.clone(), conversation(ExplicitSessionState::Uncertain))
            .await
            .unwrap();
        let cache = ConversationCache::persistent(&path).await;
        assert!(cache.get_explicit(&key).await.is_some());
        assert!(cache.reset_explicit(&key).await.unwrap());
        let cache = ConversationCache::persistent(&path).await;
        assert!(cache.get_explicit(&key).await.is_none());
    }

    #[tokio::test]
    async fn lifecycle_drop_attempts_persistence_once_and_releases_operation() {
        use crate::claude_web_state::explicit_session::ExplicitLifecycle;

        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"block").unwrap();
        let cache = ConversationCache::persistent(blocker.join("cache.json")).await;
        let key = ExplicitSessionKey::new("principal", "session");
        cache
            .set_explicit(key.clone(), conversation(ExplicitSessionState::InFlight))
            .await;
        let baseline = cache.persistence_attempts.load(Ordering::Relaxed);
        let operation = cache.try_lock_explicit_operation(&key).await.unwrap();
        drop(ExplicitLifecycle::new(
            cache.clone(),
            key.clone(),
            operation,
        ));

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if cache.try_lock_explicit_operation(&key).await.is_ok() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("operation lock must be released after one failed persistence attempt");
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert_eq!(
            cache.persistence_attempts.load(Ordering::Relaxed),
            baseline + 1
        );
        assert_eq!(
            cache
                .get_explicit(&key)
                .await
                .unwrap()
                .explicit
                .unwrap()
                .state,
            ExplicitSessionState::Uncertain
        );
    }

    #[tokio::test]
    async fn explicit_mutation_waits_for_persist_transaction_before_map_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.json");
        let cache = ConversationCache::persistent(&path).await;
        let key = ExplicitSessionKey::new("principal", "session");
        let persist_guard = cache.persist_lock.lock().await;
        let update_cache = cache.clone();
        let update_key = key.clone();
        let update = tokio::spawn(async move {
            update_cache
                .set_explicit_checked(update_key, conversation(ExplicitSessionState::InFlight))
                .await
        });
        tokio::task::yield_now().await;
        assert!(cache.get_explicit(&key).await.is_none());
        drop(persist_guard);
        update.await.unwrap().unwrap();
        cache.flush().await;
        let reloaded = ConversationCache::persistent(&path).await;
        assert!(reloaded.get_explicit(&key).await.is_some());
    }
}
