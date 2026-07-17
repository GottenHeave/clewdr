use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_with::{TimestampSecondsWithFrac, serde_as};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::warn;

use crate::utils::write_json_atomically;

const CACHE_FILE_VERSION: u32 = 1;

fn legacy_stream_health_default() -> bool {
    true
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
    /// Whether this cache entry is currently valid for reuse.
    pub valid: bool,
}

impl CachedConversation {
    /// Check if this cached conversation has expired (conservative 25-day TTL)
    pub fn is_expired(&self) -> bool {
        Utc::now() - self.created_at > Duration::days(25)
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
///
/// The downstream key index and request fingerprint together identify one
/// conversation slot. The fingerprint keeps auxiliary request families from
/// sharing the conversation used by ordinary chat requests.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Index of the downstream API key in configuration.
    pub key_index: usize,
    /// Stable fingerprint of the request family and its relevant options.
    pub request_fingerprint: u64,
}

/// Identity of a caller-managed explicit session.
///
/// Explicit sessions use a separate key space from implicit request families,
/// so a session cannot accidentally reuse an unrelated cached conversation.
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
    /// Legacy cache entries addressed by downstream key and request family.
    Legacy(CacheKey),
    /// Explicit sessions addressed by their authenticated principal and token.
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
    #[serde(
        rename = "last_stream_healthy",
        default = "legacy_stream_health_default"
    )]
    _last_stream_healthy: bool,
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
            _last_stream_healthy: true,
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
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCacheEntry {
    /// The key is persisted with the conversation so entries retain their
    /// isolation after a process restart.
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

/// Thread-safe conversation cache.
///
/// `inner` protects the in-memory entries, `persist_lock` serializes snapshots
/// written to disk, and `operation_locks` provides one async lock per key so
/// concurrent turns for the same conversation cannot interleave.
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
        map.get(key).filter(|c| c.valid && !c.is_expired()).cloned()
    }

    pub async fn lock_operation(&self, key: &CacheKey) -> OwnedMutexGuard<()> {
        self.lock_stored_operation(StoredCacheKey::Legacy(key.clone()))
            .await
    }

    pub async fn lock_explicit_operation(&self, key: &ExplicitSessionKey) -> OwnedMutexGuard<()> {
        self.lock_stored_operation(StoredCacheKey::ExplicitSession(key.clone()))
            .await
    }

    async fn lock_stored_operation(&self, key: StoredCacheKey) -> OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.operation_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);
            if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(Mutex::new(()));
                locks.insert(key, Arc::downgrade(&lock));
                lock
            }
        };
        lock.lock_owned().await
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
            map.retain(|_, v| v.valid && !v.is_expired());
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

    pub async fn flush(&self) {
        self.persist().await;
    }

    async fn persist(&self) {
        let Some(path) = self.persist_path.as_deref() else {
            return;
        };
        // Hold the persistence lock while taking and writing one snapshot so
        // concurrent mutations cannot publish partially ordered cache files.
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
        // Invalid, expired, or unknown-version entries are omitted at load
        // time; callers only observe reusable conversations.
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
            if conversation.valid && !conversation.is_expired() {
                map.insert(entry.key, conversation);
            }
        }
        Ok(map)
    }
}
