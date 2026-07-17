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

fn cache_key(key_index: usize, request_fingerprint: u64) -> CacheKey {
    CacheKey {
        key_index,
        request_fingerprint,
    }
}

fn turn(user_hashes: Vec<u64>, assistant_uuid: &str) -> CachedTurn {
    CachedTurn {
        user_hashes,
        assistant_uuid: assistant_uuid.to_owned(),
    }
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
    }
}

#[tokio::test]
async fn test_sequential_requests_use_cache() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Request 1: full messages [u1, u2, u3]
    let msgs1 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ];
    let hashes1 = extract_user_hashes(&msgs1);
    let conv = make_cached(
        "conv1",
        vec![turn(hashes1.iter().map(|(_, h)| *h).collect(), "asst0")],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Request 2: same prefix + new message [u1, u2, u3, u4]
    let msgs2 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
        make_user_msg("u4"),
    ];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);
    match result {
        DiffResult::Append {
            parent_uuid,
            new_user_indices,
            new_user_hashes,
        } => {
            assert_eq!(parent_uuid, "asst0");
            assert_eq!(new_user_indices, vec![3]);
            assert_eq!(new_user_hashes.len(), 1);
        }
        _ => panic!("Expected Append, got {result:?}"),
    }

    // Simulate successful append: update cache
    cache
        .append_turn(&key, turn(vec![hashes2[3].1], "asst1"))
        .await;

    // Request 3: same prefix + another new message [u1, u2, u3, u4, u5]
    let msgs3 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
        make_user_msg("u4"),
        make_user_msg("u5"),
    ];
    let hashes3 = extract_user_hashes(&msgs3);
    let cached = cache.get(&key).await.unwrap();
    assert_eq!(cached.turns.len(), 2);
    let result = diff::diff_messages(&cached, sys_hash, &hashes3);
    match result {
        DiffResult::Append {
            parent_uuid,
            new_user_indices,
            ..
        } => {
            assert_eq!(parent_uuid, "asst1");
            assert_eq!(new_user_indices, vec![4]);
        }
        _ => panic!("Expected Append, got {result:?}"),
    }
}

#[tokio::test]
async fn test_edit_scenario_fork() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Initial: [u1, u2, u3]
    let msgs1 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ];
    let hashes1 = extract_user_hashes(&msgs1);
    let conv = make_cached(
        "conv1",
        vec![turn(hashes1.iter().map(|(_, h)| *h).collect(), "asst0")],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Edit: [u1, u2_edited, u3]
    let msgs2 = vec![
        make_user_msg("u1"),
        make_user_msg("u2_edited"),
        make_user_msg("u3"),
    ];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);

    // Turn 0 has the mismatch (u2_edited vs u2) → FullRebuild
    assert!(matches!(result, DiffResult::FullRebuild));
}

#[tokio::test]
async fn test_edit_scenario_fork_multi_turn() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Turn 0: [u1, u2, u3], Turn 1: [u4]
    let msgs1 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ];
    let hashes1 = extract_user_hashes(&msgs1);
    let u4_hash = hash_user_message(&make_user_msg("u4"));
    let conv = make_cached(
        "conv1",
        vec![
            turn(hashes1.iter().map(|(_, h)| *h).collect(), "asst0"),
            turn(vec![u4_hash], "asst1"),
        ],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Edit u4 → [u1, u2, u3, u4_edited, u5]
    let msgs2 = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
        make_user_msg("u4_edited"),
        make_user_msg("u5"),
    ];
    let hashes2 = extract_user_hashes(&msgs2);
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash, &hashes2);

    match result {
        DiffResult::Fork {
            parent_uuid,
            fork_turn_index,
            remaining_user_indices,
            ..
        } => {
            assert_eq!(parent_uuid, "asst0");
            assert_eq!(fork_turn_index, 1);
            assert!(remaining_user_indices.contains(&3)); // u4_edited
            assert!(remaining_user_indices.contains(&4)); // u5
        }
        _ => panic!("Expected Fork, got {result:?}"),
    }
}

#[tokio::test]
async fn test_system_prompt_change_full_rebuild() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash1 = hash_system(&Some(serde_json::json!("system v1")));
    let sys_hash2 = hash_system(&Some(serde_json::json!("system v2")));

    let msgs = vec![make_user_msg("u1"), make_user_msg("u2")];
    let hashes = extract_user_hashes(&msgs);
    let conv = make_cached(
        "conv1",
        vec![turn(hashes.iter().map(|(_, h)| *h).collect(), "asst0")],
        sys_hash1,
    );
    cache.set(key.clone(), conv).await;

    // Same messages but different system prompt
    let cached = cache.get(&key).await.unwrap();
    let result = diff::diff_messages(&cached, sys_hash2, &hashes);
    assert!(matches!(result, DiffResult::FullRebuild));
}

#[tokio::test]
async fn test_model_switch_invalidation() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    let conv = make_cached(
        "conv1",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
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

#[tokio::test]
async fn test_incremental_failure_fallback() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Set up cache
    let msgs = vec![make_user_msg("u1"), make_user_msg("u2")];
    let hashes = extract_user_hashes(&msgs);
    let conv = make_cached(
        "conv1",
        vec![turn(hashes.iter().map(|(_, h)| *h).collect(), "asst0")],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Simulate failure: invalidate cache
    cache.invalidate(&key).await;

    // Next request should get cache miss
    assert!(cache.get(&key).await.is_none());

    // Caller falls back to send_full and creates new cache entry
    let new_msgs = vec![
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ];
    let new_hashes = extract_user_hashes(&new_msgs);
    let new_conv = make_cached(
        "conv2",
        vec![turn(
            new_hashes.iter().map(|(_, h)| *h).collect(),
            "asst_new",
        )],
        sys_hash,
    );
    cache.set(key.clone(), new_conv).await;

    // Verify new cache works
    let cached = cache.get(&key).await.unwrap();
    assert_eq!(cached.conv_uuid, "conv2");
}

#[tokio::test]
async fn test_cookie_rotation_invalidation() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    let conv = make_cached(
        "conv1",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // Simulate cookie rotation
    cache.invalidate_by_cookie("cookie").await;
    let cached = cache.get(&key).await;
    assert!(cached.is_none());
}

#[tokio::test]
async fn test_cache_cleanup() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Create a conversation that's already expired (created 26 days ago)
    let mut conv = make_cached(
        "conv_expired",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
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

#[tokio::test]
async fn test_cache_key_isolation() {
    let cache = ConversationCache::new();
    let key0 = cache_key(0, 0);
    let key1 = cache_key(1, 0);
    let sys_hash = hash_system(&None);

    let conv0 = make_cached(
        "conv_key0",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
        sys_hash,
    );
    let conv1 = make_cached(
        "conv_key1",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst1")],
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

#[tokio::test]
async fn test_cache_key_request_fingerprint_isolation() {
    let cache = ConversationCache::new();
    let chat_key = cache_key(0, 1);
    let diagnostic_key = cache_key(0, 2);
    let sys_hash = hash_system(&None);

    let chat_conv = make_cached("chat_conv", vec![], sys_hash);
    let diagnostic_conv = make_cached("diagnostic_conv", vec![], sys_hash);

    cache.set(chat_key.clone(), chat_conv).await;
    cache.set(diagnostic_key.clone(), diagnostic_conv).await;

    assert_eq!(cache.get(&chat_key).await.unwrap().conv_uuid, "chat_conv");
    assert_eq!(
        cache.get(&diagnostic_key).await.unwrap().conv_uuid,
        "diagnostic_conv"
    );
}

#[tokio::test]
async fn test_persistent_cache_reloads_valid_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation_cache.json");
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);
    let cookie_id = "hashed-cookie-id";

    let cache = ConversationCache::persistent(&path).await;
    let mut conv = make_cached(
        "conv_persisted",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
        sys_hash,
    );
    conv.cookie_id = cookie_id.to_string();
    cache.set(key.clone(), conv).await;

    let persisted = std::fs::read_to_string(&path).unwrap();
    assert!(persisted.contains(cookie_id));
    assert!(!persisted.contains("sessionKey="));

    let reloaded = ConversationCache::persistent(&path).await;
    let cached = reloaded.get(&key).await.unwrap();
    assert_eq!(cached.conv_uuid, "conv_persisted");
    assert_eq!(cached.cookie_id, cookie_id);
    assert_eq!(cached.turns.len(), 1);
}

#[tokio::test]
async fn test_persistent_cache_skips_expired_and_invalid_entries() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation_cache.json");
    let sys_hash = hash_system(&None);

    let expired_key = cache_key(0, 0);
    let invalid_key = cache_key(1, 0);
    let valid_key = cache_key(2, 0);
    let cache = ConversationCache::persistent(&path).await;

    let mut expired = make_cached("conv_expired", vec![], sys_hash);
    expired.created_at = chrono::Utc::now() - chrono::Duration::days(26);
    cache.set(expired_key.clone(), expired).await;

    let mut invalid = make_cached("conv_invalid", vec![], sys_hash);
    invalid.valid = false;
    cache.set(invalid_key.clone(), invalid).await;

    cache
        .set(
            valid_key.clone(),
            make_cached("conv_valid", vec![], sys_hash),
        )
        .await;

    let reloaded = ConversationCache::persistent(&path).await;
    assert!(reloaded.get(&expired_key).await.is_none());
    assert!(reloaded.get(&invalid_key).await.is_none());
    assert_eq!(
        reloaded.get(&valid_key).await.unwrap().conv_uuid,
        "conv_valid"
    );
}

#[tokio::test]
async fn test_persistent_cache_preserves_legacy_stream_health_shape() {
    #[derive(serde::Deserialize)]
    struct LegacyCache {
        conversations: Vec<LegacyEntry>,
    }

    #[derive(serde::Deserialize)]
    struct LegacyEntry {
        conversation: LegacyConversation,
    }

    #[derive(serde::Deserialize)]
    struct LegacyConversation {
        conv_uuid: String,
        last_stream_healthy: bool,
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation_cache.json");
    let key = cache_key(0, 0);
    let cache = ConversationCache::persistent(&path).await;
    cache
        .set(
            key.clone(),
            make_cached("conv_legacy", vec![], hash_system(&None)),
        )
        .await;

    let serialized = std::fs::read_to_string(&path).unwrap();
    let legacy: LegacyCache = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        legacy.conversations[0].conversation.conv_uuid,
        "conv_legacy"
    );
    assert!(legacy.conversations[0].conversation.last_stream_healthy);

    let mut persisted: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    persisted["conversations"][0]["conversation"]["last_stream_healthy"] = serde_json::json!(false);
    std::fs::write(&path, serde_json::to_vec(&persisted).unwrap()).unwrap();

    let reloaded = ConversationCache::persistent(&path).await;
    assert_eq!(reloaded.get(&key).await.unwrap().conv_uuid, "conv_legacy");

    persisted["conversations"][0]["conversation"]
        .as_object_mut()
        .unwrap()
        .remove("last_stream_healthy");
    std::fs::write(&path, serde_json::to_vec(&persisted).unwrap()).unwrap();
    let reloaded_without_field = ConversationCache::persistent(&path).await;
    assert_eq!(
        reloaded_without_field.get(&key).await.unwrap().conv_uuid,
        "conv_legacy"
    );
}
