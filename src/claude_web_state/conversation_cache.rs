use std::collections::HashMap;
use std::path::{Path, PathBuf};
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

fn explicit_missing(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(
        http::StatusCode::CONFLICT,
        "conversation_reuse_failed",
        message,
    )
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

/// Thread-safe conversation cache
#[derive(Clone)]
pub struct ConversationCache {
    inner: Arc<Mutex<HashMap<StoredCacheKey, CachedConversation>>>,
    operation_locks: Arc<Mutex<HashMap<StoredCacheKey, Weak<Mutex<()>>>>>,
    persist_path: Option<Arc<PathBuf>>,
    persist_lock: Arc<Mutex<()>>,
}

impl ConversationCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            operation_locks: Arc::new(Mutex::new(HashMap::new())),
            persist_path: None,
            persist_lock: Arc::new(Mutex::new(())),
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
        let updated = {
            let mut map = self.inner.lock().await;
            let conversation = map
                .get_mut(&StoredCacheKey::ExplicitSession(key.clone()))
                .ok_or_else(|| explicit_missing("Session disappeared before completion"))?;
            let explicit = conversation
                .explicit
                .as_mut()
                .ok_or_else(|| explicit_missing("Session metadata is unavailable"))?;
            explicit.state = ExplicitSessionState::InFlight;
            explicit.pending = Some(pending);
            conversation.last_used = Utc::now();
            true
        };
        if updated {
            self.persist().await;
        }
        Ok(())
    }

    pub async fn commit_explicit_turn(
        &self,
        key: &ExplicitSessionKey,
        assistant_digest_after: Option<String>,
    ) -> Result<(), ProtocolError> {
        {
            let mut map = self.inner.lock().await;
            let conversation = map
                .get_mut(&StoredCacheKey::ExplicitSession(key.clone()))
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
        }
        self.persist().await;
        Ok(())
    }

    pub async fn mark_explicit_uncertain(&self, key: &ExplicitSessionKey) {
        let updated = {
            let mut map = self.inner.lock().await;
            map.get_mut(&StoredCacheKey::ExplicitSession(key.clone()))
                .and_then(|conversation| conversation.explicit.as_mut())
                .map(|explicit| {
                    explicit.state = ExplicitSessionState::Uncertain;
                })
                .is_some()
        };
        if updated {
            self.persist().await;
        }
    }

    pub async fn tombstone_explicit(&self, key: &ExplicitSessionKey) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conversation) = map.get_mut(&StoredCacheKey::ExplicitSession(key.clone()))
                && let Some(explicit) = conversation.explicit.as_mut()
            {
                explicit.state = ExplicitSessionState::Tombstoned;
                explicit.pending = None;
                conversation.last_used = Utc::now();
                true
            } else {
                false
            }
        };
        if updated {
            self.persist().await;
        }
    }

    pub async fn reset_explicit(&self, key: &ExplicitSessionKey) -> bool {
        let removed = self
            .inner
            .lock()
            .await
            .remove(&StoredCacheKey::ExplicitSession(key.clone()))
            .is_some();
        if removed {
            self.persist().await;
        }
        removed
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
        let Some(path) = self.persist_path.as_deref() else {
            return;
        };
        let _guard = self.persist_lock.lock().await;
        let snapshot = {
            let map = self.inner.lock().await;
            PersistedConversationCache::from_map(&map)
        };
        if let Err(err) = write_json_atomically(path, &snapshot).await {
            warn!(
                "[CACHE] failed to persist conversation cache to {}: {}",
                path.display(),
                err
            );
        }
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
