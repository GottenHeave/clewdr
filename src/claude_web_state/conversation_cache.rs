use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use chrono::{DateTime, Duration, Utc};

/// Represents one round-trip (ClewdR request → Claude response) in a cached conversation
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CacheKey {
    /// Index of the downstream API key in config (from self.key)
    pub key_index: usize,
}

/// Thread-safe conversation cache
#[derive(Clone)]
pub struct ConversationCache {
    inner: Arc<Mutex<HashMap<CacheKey, CachedConversation>>>,
}

impl ConversationCache {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn get(&self, key: &CacheKey) -> Option<CachedConversation> {
        let map = self.inner.lock().await;
        map.get(key).filter(|c| c.valid && !c.is_expired()).cloned()
    }

    pub async fn set(&self, key: CacheKey, conv: CachedConversation) {
        let mut map = self.inner.lock().await;
        map.insert(key, conv);
    }

    /// Append a new turn to an existing cached conversation
    pub async fn append_turn(&self, key: &CacheKey, turn: CachedTurn) {
        let mut map = self.inner.lock().await;
        if let Some(conv) = map.get_mut(key) {
            conv.turns.push(turn);
            conv.last_used = Utc::now();
        }
    }

    /// Truncate turns and append a new one (fork scenario)
    pub async fn fork_and_append(&self, key: &CacheKey, from_index: usize, turn: CachedTurn) {
        let mut map = self.inner.lock().await;
        if let Some(conv) = map.get_mut(key) {
            conv.truncate_turns(from_index);
            conv.turns.push(turn);
            conv.last_used = Utc::now();
        }
    }

    /// Mark a cached conversation as invalid
    pub async fn invalidate(&self, key: &CacheKey) {
        let mut map = self.inner.lock().await;
        if let Some(conv) = map.get_mut(key) {
            conv.valid = false;
        }
    }

    /// Remove expired entries (call periodically)
    pub async fn cleanup(&self) {
        let mut map = self.inner.lock().await;
        map.retain(|_, v| v.valid && !v.is_expired());
    }

    /// Invalidate all entries for a given cookie_id (cookie rotation)
    pub async fn invalidate_by_cookie(&self, cookie_id: &str) {
        let mut map = self.inner.lock().await;
        for conv in map.values_mut() {
            if conv.cookie_id == cookie_id {
                conv.valid = false;
            }
        }
    }

    /// Update the stream health flag on an existing cached conversation
    pub async fn update_stream_health(&self, key: &CacheKey, flag: Arc<AtomicBool>) {
        let mut map = self.inner.lock().await;
        if let Some(conv) = map.get_mut(key) {
            conv.last_stream_healthy = flag;
        }
    }

    /// Check if the last stream completed healthily for a given cache key
    pub async fn is_last_stream_healthy(&self, key: &CacheKey) -> bool {
        let map = self.inner.lock().await;
        map.get(key)
            .map(|c| c.last_stream_healthy.load(Ordering::Relaxed))
            .unwrap_or(true)
    }
}
