use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_with::{TimestampSecondsWithFrac, serde_as};
use tokio::sync::Mutex;
use tracing::warn;

const CACHE_FILE_VERSION: u32 = 1;

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

/// Cache key: identifies a unique "conversation slot"
/// First version: one conversation per (cookie, key_index) pair
/// This means each downstream API key gets one cached conversation per cookie
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    /// Index of the downstream API key in config (from self.key)
    pub key_index: usize,
    /// Request-family fingerprint used to isolate auxiliary requests.
    pub request_fingerprint: u64,
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
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedCacheEntry {
    key: CacheKey,
    conversation: PersistedConversation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PersistedConversationCache {
    version: u32,
    conversations: Vec<PersistedCacheEntry>,
}

impl PersistedConversationCache {
    fn from_map(map: &HashMap<CacheKey, CachedConversation>) -> Self {
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
    inner: Arc<Mutex<HashMap<CacheKey, CachedConversation>>>,
    persist_path: Option<Arc<PathBuf>>,
    persist_lock: Arc<Mutex<()>>,
}

impl ConversationCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
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
            persist_path: Some(Arc::new(persist_path)),
            persist_lock: Arc::new(Mutex::new(())),
        }
    }

    pub async fn get(&self, key: &CacheKey) -> Option<CachedConversation> {
        let map = self.inner.lock().await;
        map.get(key).filter(|c| c.valid && !c.is_expired()).cloned()
    }

    pub async fn set(&self, key: CacheKey, conv: CachedConversation) {
        {
            let mut map = self.inner.lock().await;
            map.insert(key, conv);
        }
        self.persist().await;
    }

    /// Append a new turn to an existing cached conversation
    pub async fn append_turn(&self, key: &CacheKey, turn: CachedTurn) {
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

    /// Update the stream health flag on an existing cached conversation
    pub async fn update_stream_health(&self, key: &CacheKey, flag: Arc<AtomicBool>) {
        let updated = {
            let mut map = self.inner.lock().await;
            if let Some(conv) = map.get_mut(key) {
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
        map.get(key)
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
        if let Err(err) = Self::write_to_path(path, &snapshot).await {
            warn!(
                "[CACHE] failed to persist conversation cache to {}: {}",
                path.display(),
                err
            );
        }
    }

    async fn load_from_path(
        path: &Path,
    ) -> Result<HashMap<CacheKey, CachedConversation>, Box<dyn std::error::Error + Send + Sync>>
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
            if conversation.valid && !conversation.is_expired() {
                map.insert(entry.key, conversation);
            }
        }
        Ok(map)
    }

    async fn write_to_path(
        path: &Path,
        snapshot: &PersistedConversationCache,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp_path = path.with_extension("json.tmp");
        let data = serde_json::to_vec_pretty(snapshot)?;
        tokio::fs::write(&tmp_path, data).await?;
        match tokio::fs::rename(&tmp_path, path).await {
            Ok(()) => Ok(()),
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::AlreadyExists | std::io::ErrorKind::PermissionDenied
                ) =>
            {
                let _ = tokio::fs::remove_file(path).await;
                tokio::fs::rename(&tmp_path, path).await?;
                Ok(())
            }
            Err(err) => Err(Box::new(err)),
        }
    }
}
