use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use clewdr::claude_web_state::conversation_cache::{
    CacheKey, CachedConversation, ConversationCache,
};
use clewdr::protocol::{AuthPrincipal, parse_session_id};
use clewdr::utils::write_json_atomically;
use serde_json::json;

fn cached_conversation(id: &str) -> CachedConversation {
    CachedConversation {
        conv_uuid: id.to_owned(),
        org_uuid: "org".to_owned(),
        cookie_id: "cookie".to_owned(),
        model: "model".to_owned(),
        is_pro: false,
        system_hash: 0,
        turns: Vec::new(),
        created_at: chrono::Utc::now(),
        last_used: chrono::Utc::now(),
        valid: true,
        last_stream_healthy: Arc::new(AtomicBool::new(true)),
    }
}

#[tokio::test]
async fn reads_v1_cache_keys_without_a_scope_field() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation_cache.json");
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1_000.0;
    let document = json!({
        "version": 1,
        "conversations": [{
            "key": { "key_index": 3, "request_fingerprint": 42 },
            "conversation": {
                "conv_uuid": "legacy-conversation",
                "org_uuid": "org",
                "cookie_id": "cookie",
                "model": "model",
                "is_pro": false,
                "system_hash": 0,
                "turns": [],
                "created_at": now,
                "last_used": now,
                "valid": true,
                "last_stream_healthy": true
            }
        }]
    });
    std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();

    let cache = ConversationCache::persistent(path).await;
    let conversation = cache.get(&CacheKey::legacy(3, 42)).await.unwrap();

    assert_eq!(conversation.conv_uuid, "legacy-conversation");
}

#[tokio::test]
async fn explicit_sessions_are_separate_cache_scopes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conversation_cache.json");
    let cache = ConversationCache::persistent(&path).await;
    let first = CacheKey::explicit_session("principal", "a".repeat(64));
    let second = CacheKey::explicit_session("principal", "b".repeat(64));
    cache.set(first.clone(), cached_conversation("first")).await;
    cache
        .set(second.clone(), cached_conversation("second"))
        .await;
    let cache = ConversationCache::persistent(&path).await;

    assert_eq!(cache.get(&first).await.unwrap().conv_uuid, "first");
    assert_eq!(cache.get(&second).await.unwrap().conv_uuid, "second");
    assert!(cache.get(&CacheKey::legacy(0, 0)).await.is_none());
}

#[tokio::test]
async fn operation_locks_serialize_only_matching_keys() {
    let cache = ConversationCache::new();
    let key = CacheKey::explicit_session("principal", "a".repeat(64));
    let other_key = CacheKey::explicit_session("principal", "b".repeat(64));
    let held = cache.lock_operation(&key).await;

    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            cache.lock_operation(&key)
        )
        .await
        .is_err()
    );
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            cache.lock_operation(&other_key)
        )
        .await
        .is_ok()
    );

    drop(held);
    assert!(
        tokio::time::timeout(
            std::time::Duration::from_millis(20),
            cache.lock_operation(&key)
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn atomic_json_writer_overwrites_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");

    write_json_atomically(&path, &json!({ "revision": 1 }))
        .await
        .unwrap();
    write_json_atomically(&path, &json!({ "revision": 2 }))
        .await
        .unwrap();

    let persisted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(persisted, json!({ "revision": 2 }));
}

#[test]
fn parses_only_versioned_lowercase_sha256_session_ids() {
    let valid = format!("cherry_topic_v1_{}", "ab".repeat(32));
    assert_eq!(
        parse_session_id(Some(&valid)).unwrap(),
        Some("ab".repeat(32))
    );
    assert_eq!(parse_session_id(Some("legacy-client")).unwrap(), None);
    assert!(parse_session_id(Some(&format!("cherry_topic_v1_{}", "AB".repeat(32)))).is_err());
    assert!(parse_session_id(Some("cherry_topic_v1_1234")).is_err());
}

#[test]
fn authenticated_principal_is_stable_and_versioned() {
    let principal = AuthPrincipal::for_authenticated_user();
    assert_eq!(principal.as_str().len(), 64);
    assert_eq!(principal, AuthPrincipal::for_authenticated_user());
}
