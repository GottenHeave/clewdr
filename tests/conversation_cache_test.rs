use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use clewdr::claude_web_state::conversation_cache::{
    CacheKey, CachedConversation, CachedTurn, ConversationCache,
};
use clewdr::claude_web_state::diff::{
    self, DiffResult, extract_user_hashes, hash_system, hash_user_message,
};
use clewdr::types::claude::{Message, Role};

fn make_user_msg(text: &str) -> Message {
    Message::new_text(Role::User, text)
}

fn make_cached(conv_uuid: &str, turns: Vec<CachedTurn>, system_hash: u64) -> CachedConversation {
    CachedConversation {
        conv_uuid: conv_uuid.to_string(),
        org_uuid: "org".to_string(),
        cookie_id: "cookie".to_string(),
        model: "model".to_string(),
        is_pro: false,
        system_hash,
        turns,
        created_at: chrono::Utc::now(),
        last_used: chrono::Utc::now(),
        valid: true,
        last_stream_healthy: Arc::new(AtomicBool::new(true)),
    }
}

/// Test: 3 sequential requests, verify 2nd and 3rd use cache
#[tokio::test]
async fn test_sequential_requests_use_cache() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    // Request 1: full messages [u1, u2, u3]
    let msgs1 = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3")];
    let hashes1 = extract_user_hashes(&msgs1);
    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: hashes1.iter().map(|(_, h)| *h).collect(),
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Request 2: same prefix + new message [u1, u2, u3, u4]
    let msgs2 = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3"), make_user_msg("u4")];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);
    match result {
        DiffResult::Append { parent_uuid, new_user_indices, new_user_hashes } => {
            assert_eq!(parent_uuid, "asst0");
            assert_eq!(new_user_indices, vec![3]);
            assert_eq!(new_user_hashes.len(), 1);
        }
        _ => panic!("Expected Append, got {:?}", result),
    }

    // Simulate successful append: update cache
    cache.append_turn(&key, CachedTurn {
        user_hashes: vec![hashes2[3].1],
        assistant_uuid: "asst1".to_string(),
    }).await;

    // Request 3: same prefix + another new message [u1, u2, u3, u4, u5]
    let msgs3 = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3"), make_user_msg("u4"), make_user_msg("u5")];
    let hashes3 = extract_user_hashes(&msgs3);
    let cached = cache.get(&key).await.unwrap();
    assert_eq!(cached.turns.len(), 2);
    let result = diff::diff_messages(&cached, sys_hash, &hashes3);
    match result {
        DiffResult::Append { parent_uuid, new_user_indices, .. } => {
            assert_eq!(parent_uuid, "asst1");
            assert_eq!(new_user_indices, vec![4]);
        }
        _ => panic!("Expected Append, got {:?}", result),
    }
}

/// Test: edit scenario (message modification → fork)
#[tokio::test]
async fn test_edit_scenario_fork() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    // Initial: [u1, u2, u3]
    let msgs1 = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3")];
    let hashes1 = extract_user_hashes(&msgs1);
    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: hashes1.iter().map(|(_, h)| *h).collect(),
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Edit: [u1, u2_edited, u3]
    let msgs2 = vec![make_user_msg("u1"), make_user_msg("u2_edited"), make_user_msg("u3")];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);

    // Turn 0 has the mismatch (u2_edited vs u2) → FullRebuild
    assert!(matches!(result, DiffResult::FullRebuild));
}

/// Test: edit scenario with multi-turn fork
#[tokio::test]
async fn test_edit_scenario_fork_multi_turn() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    // Turn 0: [u1, u2, u3], Turn 1: [u4]
    let msgs1 = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3")];
    let hashes1 = extract_user_hashes(&msgs1);
    let u4_hash = hash_user_message(&make_user_msg("u4"));
    let conv = make_cached(
        "conv1",
        vec![
            CachedTurn {
                user_hashes: hashes1.iter().map(|(_, h)| *h).collect(),
                assistant_uuid: "asst0".to_string(),
            },
            CachedTurn {
                user_hashes: vec![u4_hash],
                assistant_uuid: "asst1".to_string(),
            },
        ],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Edit u4 → [u1, u2, u3, u4_edited, u5]
    let msgs2 = vec![
        make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3"),
        make_user_msg("u4_edited"), make_user_msg("u5"),
    ];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);

    match result {
        DiffResult::Fork { parent_uuid, fork_turn_index, remaining_user_indices, .. } => {
            assert_eq!(parent_uuid, "asst0");
            assert_eq!(fork_turn_index, 1);
            assert!(remaining_user_indices.contains(&3)); // u4_edited
            assert!(remaining_user_indices.contains(&4)); // u5
        }
        _ => panic!("Expected Fork, got {:?}", result),
    }
}

/// Test: system prompt change → full rebuild
#[tokio::test]
async fn test_system_prompt_change_full_rebuild() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash1 = hash_system(&Some(serde_json::json!("system v1")));
    let sys_hash2 = hash_system(&Some(serde_json::json!("system v2")));

    let msgs = vec![make_user_msg("u1"), make_user_msg("u2")];
    let hashes = extract_user_hashes(&msgs);
    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: hashes.iter().map(|(_, h)| *h).collect(),
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash1,
    );
    cache.set(key.clone(), conv).await;

    // Same messages but different system prompt
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash2, &hashes);
    assert!(matches!(result, DiffResult::FullRebuild));
}

/// Test: model switch → cache invalidated
#[tokio::test]
async fn test_model_switch_invalidation() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Verify cache is valid
    let cached = cache.get(&key).await.unwrap();
    assert_eq!(cached.model, "model");

    // Simulate model change: the caller invalidates and creates new
    cache.invalidate(&key).await;
    assert!(cache.get(&key).await.is_none());
}

/// Test: incremental failure → fallback to full rebuild
#[tokio::test]
async fn test_incremental_failure_fallback() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    // Set up cache
    let msgs = vec![make_user_msg("u1"), make_user_msg("u2")];
    let hashes = extract_user_hashes(&msgs);
    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: hashes.iter().map(|(_, h)| *h).collect(),
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Simulate failure: invalidate cache
    cache.invalidate(&key).await;

    // Next request should get cache miss
    assert!(cache.get(&key).await.is_none());

    // Caller falls back to send_full and creates new cache entry
    let new_msgs = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3")];
    let new_hashes = extract_user_hashes(&new_msgs);
    let new_conv = make_cached(
        "conv2",
        vec![CachedTurn {
            user_hashes: new_hashes.iter().map(|(_, h)| *h).collect(),
            assistant_uuid: "asst_new".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), new_conv).await;

    // Verify new cache works
    let cached = cache.get(&key).await.unwrap();
    assert_eq!(cached.conv_uuid, "conv2");
}

/// Test: cookie rotation → cache invalidation
#[tokio::test]
async fn test_cookie_rotation_invalidation() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    let conv = make_cached(
        "conv1",
        vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Simulate cookie rotation
    cache.invalidate_by_cookie("cookie").await;
    let cached = cache.get(&key).await;
    assert!(cached.is_none());
}

/// Test: cache cleanup removes expired entries
#[tokio::test]
async fn test_cache_cleanup() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    // Create a conversation that's already expired (created 26 days ago)
    let mut conv = make_cached(
        "conv_expired",
        vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    conv.created_at = chrono::Utc::now() - chrono::Duration::days(26);
    cache.set(key.clone(), conv).await;

    // Before cleanup, it exists but is expired
    let cached = cache.get(&key).await;
    assert!(cached.is_none()); // get() filters expired

    // Cleanup removes it
    cache.cleanup().await;
}

/// Test: stream health flag is shared between cache and stream
#[tokio::test]
async fn test_stream_health_flag() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    let flag = Arc::new(AtomicBool::new(false));
    let conv = CachedConversation {
        conv_uuid: "conv1".to_string(),
        org_uuid: "org".to_string(),
        cookie_id: "cookie".to_string(),
        model: "model".to_string(),
        is_pro: false,
        system_hash: sys_hash,
        turns: vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        created_at: chrono::Utc::now(),
        last_used: chrono::Utc::now(),
        valid: true,
        last_stream_healthy: flag.clone(),
    };
    cache.set(key.clone(), conv).await;

    // Initially unhealthy
    assert!(!cache.is_last_stream_healthy(&key).await);

    // Simulate stream completion
    flag.store(true, std::sync::atomic::Ordering::Relaxed);

    // Now healthy
    assert!(cache.is_last_stream_healthy(&key).await);
}

/// Test: stream health flag update on append
#[tokio::test]
async fn test_stream_health_update_on_append() {
    let cache = ConversationCache::new();
    let key = CacheKey { key_index: 0 };
    let sys_hash = hash_system(&None);

    let flag = Arc::new(AtomicBool::new(true)); // initially healthy (stream completed)
    let conv = CachedConversation {
        conv_uuid: "conv1".to_string(),
        org_uuid: "org".to_string(),
        cookie_id: "cookie".to_string(),
        model: "model".to_string(),
        is_pro: false,
        system_hash: sys_hash,
        turns: vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        created_at: chrono::Utc::now(),
        last_used: chrono::Utc::now(),
        valid: true,
        last_stream_healthy: flag,
    };
    cache.set(key.clone(), conv).await;

    // New request with new flag
    let new_flag = Arc::new(AtomicBool::new(false));
    cache.update_stream_health(&key, new_flag.clone()).await;

    // Not yet healthy
    assert!(!cache.is_last_stream_healthy(&key).await);

    // Stream completes
    new_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(cache.is_last_stream_healthy(&key).await);
}

/// Test: cache key isolation
#[tokio::test]
async fn test_cache_key_isolation() {
    let cache = ConversationCache::new();
    let key0 = CacheKey { key_index: 0 };
    let key1 = CacheKey { key_index: 1 };
    let sys_hash = hash_system(&None);

    let conv0 = make_cached(
        "conv_key0",
        vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst0".to_string(),
        }],
        sys_hash,
    );
    let conv1 = make_cached(
        "conv_key1",
        vec![CachedTurn {
            user_hashes: vec![hash_user_message(&make_user_msg("u1"))],
            assistant_uuid: "asst1".to_string(),
        }],
        sys_hash,
    );

    cache.set(key0.clone(), conv0).await;
    cache.set(key1.clone(), conv1).await;

    let c0 = cache.get(&key0).await.unwrap();
    let c1 = cache.get(&key1).await.unwrap();
    assert_eq!(c0.conv_uuid, "conv_key0");
    assert_eq!(c1.conv_uuid, "conv_key1");

    // Invalidate one doesn't affect the other
    cache.invalidate(&key0).await;
    assert!(cache.get(&key0).await.is_none());
    assert!(cache.get(&key1).await.is_some());
}
