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
        explicit: None,
    }
}

async fn cached_diff(
    turns: Vec<CachedTurn>,
    cached_system: u64,
    requested_system: u64,
    messages: &[&str],
) -> DiffResult {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    cache
        .set(key.clone(), make_cached("conv1", turns, cached_system))
        .await;
    let messages = messages
        .iter()
        .map(|text| make_user_msg(text))
        .collect::<Vec<_>>();
    diff::diff_messages(
        &cache.get(&key).await.unwrap(),
        requested_system,
        &extract_user_hashes(&messages),
    )
}

#[tokio::test]
async fn test_sequential_requests_use_cache() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Request 1 stores the full user prefix and its assistant parent.
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

    // Request 2 appends one user while preserving the cached parent.
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

    // Simulate the successful append before the next request.
    cache
        .append_turn(&key, turn(vec![hashes2[3].1], "asst1"))
        .await;

    // Request 3 appends another user after the new cached turn.
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
    let sys_hash = hash_system(&None);
    // Editing a message in the first turn requires a complete rebuild.
    let hashes = extract_user_hashes(&[
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ]);
    assert!(matches!(
        cached_diff(
            vec![turn(
                hashes.into_iter().map(|(_, hash)| hash).collect(),
                "asst0"
            )],
            sys_hash,
            sys_hash,
            &["u1", "u2_edited", "u3"],
        )
        .await,
        DiffResult::FullRebuild
    ));
}

#[tokio::test]
async fn test_edit_scenario_fork_multi_turn() {
    let sys_hash = hash_system(&None);
    // A mismatch in turn one forks from turn zero and retains the edited suffix.
    let hashes = extract_user_hashes(&[
        make_user_msg("u1"),
        make_user_msg("u2"),
        make_user_msg("u3"),
    ]);
    match cached_diff(
        vec![
            turn(hashes.into_iter().map(|(_, hash)| hash).collect(), "asst0"),
            turn(vec![hash_user_message(&make_user_msg("u4"))], "asst1"),
        ],
        sys_hash,
        sys_hash,
        &["u1", "u2", "u3", "u4_edited", "u5"],
    )
    .await
    {
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
        result => panic!("Expected Fork, got {result:?}"),
    }
}

#[tokio::test]
async fn test_system_prompt_change_full_rebuild() {
    let sys_hash1 = hash_system(&Some(serde_json::json!("system v1")));
    let sys_hash2 = hash_system(&Some(serde_json::json!("system v2")));
    // A changed system digest invalidates reuse even when user messages match.
    let hash = hash_user_message(&make_user_msg("u1"));
    assert!(matches!(
        cached_diff(
            vec![turn(vec![hash], "asst0")],
            sys_hash1,
            sys_hash2,
            &["u1"]
        )
        .await,
        DiffResult::FullRebuild
    ));
}

#[tokio::test]
async fn test_incremental_failure_fallback() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    let msgs = vec![make_user_msg("u1"), make_user_msg("u2")];
    let hashes = extract_user_hashes(&msgs);
    let conv = make_cached(
        "conv1",
        vec![turn(hashes.iter().map(|(_, h)| *h).collect(), "asst0")],
        sys_hash,
    );
    cache.set(key.clone(), conv).await;

    // A failed incremental request invalidates the entry before send_full recreates it.
    cache.invalidate(&key).await;
    assert!(cache.get(&key).await.is_none());
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

    // Rotating a cookie invalidates every conversation bound to the old identity.
    cache.invalidate_by_cookie("cookie").await;
    let cached = cache.get(&key).await;
    assert!(cached.is_none());
}

#[tokio::test]
async fn test_cache_cleanup() {
    let cache = ConversationCache::new();
    let key = cache_key(0, 0);
    let sys_hash = hash_system(&None);

    // Implicit entries use the 25-day TTL and are removed by cleanup.
    let mut conv = make_cached(
        "conv_expired",
        vec![turn(vec![hash_user_message(&make_user_msg("u1"))], "asst0")],
        sys_hash,
    );
    conv.created_at = chrono::Utc::now() - chrono::Duration::days(26);
    cache.set(key.clone(), conv).await;

    let cached = cache.get(&key).await;
    assert!(cached.is_none());
    cache.cleanup().await;
}

#[tokio::test]
async fn test_cache_key_isolation() {
    for (first, second) in [((0, 0), (1, 0)), ((0, 1), (0, 2))] {
        // Both key dimensions isolate cache records from each other.
        let cache = ConversationCache::new();
        let first = cache_key(first.0, first.1);
        let second = cache_key(second.0, second.1);
        cache
            .set(first.clone(), make_cached("first", vec![], 0))
            .await;
        cache
            .set(second.clone(), make_cached("second", vec![], 0))
            .await;
        assert_eq!(cache.get(&first).await.unwrap().conv_uuid, "first");
        assert_eq!(cache.get(&second).await.unwrap().conv_uuid, "second");
        cache.invalidate(&first).await;
        assert!(cache.get(&first).await.is_none());
        assert!(cache.get(&second).await.is_some());
    }
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
