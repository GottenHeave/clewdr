use std::hash::{DefaultHasher, Hash, Hasher};
use serde_json::Value;
use crate::types::claude::{Message, MessageContent, ContentBlock, Role};
use super::conversation_cache::CachedConversation;

/// Hash a single user message's text content
pub fn hash_user_message(msg: &Message) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Hash the text content of the message
    match &msg.content {
        MessageContent::Text { content } => {
            content.hash(&mut hasher);
        }
        MessageContent::Blocks { content } => {
            for block in content {
                match block {
                    ContentBlock::Text { text, .. } => {
                        text.hash(&mut hasher);
                    }
                    ContentBlock::Image { source, .. } => {
                        // Hash the image source for change detection
                        // For base64: hash the data
                        // This ensures image changes are detected
                        format!("{:?}", source).hash(&mut hasher);
                    }
                    _ => {}
                }
            }
        }
    }
    hasher.finish()
}

/// Hash system prompt
pub fn hash_system(system: &Option<Value>) -> u64 {
    let mut hasher = DefaultHasher::new();
    match system {
        Some(v) => format!("{}", v).hash(&mut hasher),
        None => 0u64.hash(&mut hasher),
    }
    hasher.finish()
}

/// Extract (message_ref, hash) pairs for all Role::User messages in order
pub fn extract_user_hashes(messages: &[Message]) -> Vec<(usize, u64)> {
    messages.iter().enumerate()
        .filter(|(_, m)| m.role == Role::User)
        .map(|(idx, m)| (idx, hash_user_message(m)))
        .collect()
}

/// Result of diffing new messages against cached conversation
#[derive(Debug)]
pub enum DiffResult {
    /// All cached turns match; new user messages should be appended
    Append {
        /// parent_message_uuid = last cached turn's assistant_uuid
        parent_uuid: String,
        /// Indices into the original messages array for new Role::User messages
        new_user_indices: Vec<usize>,
        /// Hashes of the new user messages (for cache write)
        new_user_hashes: Vec<u64>,
    },
    /// A cached turn doesn't match; fork from an earlier point
    Fork {
        /// parent_message_uuid = the turn before the mismatch
        parent_uuid: String,
        /// Which turn index to truncate from
        fork_turn_index: usize,
        /// Indices into the original messages array for all user messages from fork point
        remaining_user_indices: Vec<usize>,
        /// Hashes of those user messages
        remaining_user_hashes: Vec<u64>,
    },
    /// Cannot reuse; must do full rebuild
    FullRebuild,
}

/// Compare new user messages against cached turns to determine reuse strategy
///
/// # Arguments
/// * `cached` - The cached conversation state
/// * `new_system_hash` - Hash of the current system prompt
/// * `user_hashes` - (original_index, hash) pairs for all Role::User messages
///
/// # Logic
/// Walk through cached turns in order. Each turn has a list of user_hashes.
/// For each turn, check if the corresponding user messages match.
///
/// - If all turns match and there are remaining user messages → Append
/// - If a turn mismatches at its start → Fork from previous turn
/// - If turn[0] mismatches internally → FullRebuild (can't partially reuse paste)
/// - If system prompt changed → FullRebuild
pub fn diff_messages(
    cached: &CachedConversation,
    new_system_hash: u64,
    user_hashes: &[(usize, u64)], // (original_msg_index, hash)
) -> DiffResult {
    // System prompt changed → full rebuild
    if cached.system_hash != new_system_hash {
        return DiffResult::FullRebuild;
    }

    // No cached turns (shouldn't happen, but defensive)
    if cached.turns.is_empty() {
        return DiffResult::FullRebuild;
    }

    let mut cursor: usize = 0; // position in user_hashes

    for (turn_idx, turn) in cached.turns.iter().enumerate() {
        for (hash_idx_in_turn, &cached_hash) in turn.user_hashes.iter().enumerate() {
            if cursor >= user_hashes.len() {
                // New messages are shorter than cached → can't reuse
                // (user deleted messages from the end)
                return DiffResult::FullRebuild;
            }

            let (_orig_idx, new_hash) = user_hashes[cursor];

            if new_hash != cached_hash {
                // === MISMATCH DETECTED ===

                if turn_idx == 0 {
                    // Turn 0 is the full paste — can't partially reuse
                    return DiffResult::FullRebuild;
                }

                // Fork from the previous turn
                let parent_uuid = cached.turns[turn_idx - 1].assistant_uuid.clone();

                // Rewind cursor to the start of this turn
                let rewind_count = hash_idx_in_turn;
                let fork_cursor = cursor - rewind_count;

                let remaining_user_indices: Vec<usize> =
                    user_hashes[fork_cursor..].iter().map(|(idx, _)| *idx).collect();
                let remaining_user_hashes: Vec<u64> =
                    user_hashes[fork_cursor..].iter().map(|(_, h)| *h).collect();

                return DiffResult::Fork {
                    parent_uuid,
                    fork_turn_index: turn_idx,
                    remaining_user_indices,
                    remaining_user_hashes,
                };
            }

            cursor += 1;
        }
    }

    // All cached turns matched
    if cursor >= user_hashes.len() {
        // No new messages — this shouldn't normally happen
        // (implies exact same request as before)
        return DiffResult::FullRebuild;
    }

    // Remaining messages are new → Append
    let parent_uuid = cached.turns.last().unwrap().assistant_uuid.clone();
    let new_user_indices: Vec<usize> =
        user_hashes[cursor..].iter().map(|(idx, _)| *idx).collect();
    let new_user_hashes: Vec<u64> =
        user_hashes[cursor..].iter().map(|(_, h)| *h).collect();

    DiffResult::Append {
        parent_uuid,
        new_user_indices,
        new_user_hashes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude::Message;
    use crate::claude_web_state::conversation_cache::{CachedConversation, CachedTurn};
    use chrono::Utc;

    fn make_cached(conv_uuid: &str, turns: Vec<CachedTurn>, system_hash: u64) -> CachedConversation {
        CachedConversation {
            conv_uuid: conv_uuid.to_string(),
            org_uuid: "org".to_string(),
            cookie_id: "cookie".to_string(),
            model: "model".to_string(),
            is_pro: false,
            system_hash,
            turns,
            created_at: Utc::now(),
            last_used: Utc::now(),
            valid: true,
        }
    }

    fn make_user_msg(text: &str) -> Message {
        Message::new_text(Role::User, text)
    }

    #[test]
    fn test_hash_user_message_text() {
        let msg = make_user_msg("hello");
        let h1 = hash_user_message(&msg);
        let msg2 = make_user_msg("hello");
        let h2 = hash_user_message(&msg2);
        assert_eq!(h1, h2);

        let msg3 = make_user_msg("world");
        let h3 = hash_user_message(&msg3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_system_matching() {
        let sys = Some(serde_json::json!("system prompt"));
        let h1 = hash_system(&sys);
        let h2 = hash_system(&sys);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_system_different() {
        let h1 = hash_system(&Some(serde_json::json!("sys1")));
        let h2 = hash_system(&Some(serde_json::json!("sys2")));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_extract_user_hashes() {
        let messages = vec![
            make_user_msg("u1"),
            Message::new_text(Role::Assistant, "a1"),
            make_user_msg("u2"),
        ];
        let hashes = extract_user_hashes(&messages);
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0].0, 0);
        assert_eq!(hashes[1].0, 2);
    }

    #[test]
    fn test_diff_messages_append() {
        let msgs = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3")];
        let hashes = extract_user_hashes(&msgs);
        let sys_hash = hash_system(&None);

        let cached = make_cached("conv1", vec![CachedTurn {
            user_hashes: vec![hashes[0].1, hashes[1].1],
            assistant_uuid: "asst1".to_string(),
        }], sys_hash);

        let result = diff_messages(&cached, sys_hash, &hashes);
        match result {
            DiffResult::Append { parent_uuid, new_user_indices, new_user_hashes } => {
                assert_eq!(parent_uuid, "asst1");
                assert_eq!(new_user_indices, vec![2]);
                assert_eq!(new_user_hashes.len(), 1);
                assert_eq!(new_user_hashes[0], hashes[2].1);
            }
            _ => panic!("Expected Append, got {:?}", result),
        }
    }

    #[test]
    fn test_diff_messages_full_rebuild_system_change() {
        let msgs = vec![make_user_msg("u1")];
        let hashes = extract_user_hashes(&msgs);
        let sys_hash1 = hash_system(&Some(serde_json::json!("sys1")));
        let sys_hash2 = hash_system(&Some(serde_json::json!("sys2")));

        let cached = make_cached("conv1", vec![CachedTurn {
            user_hashes: vec![hashes[0].1],
            assistant_uuid: "asst1".to_string(),
        }], sys_hash1);

        let result = diff_messages(&cached, sys_hash2, &hashes);
        assert!(matches!(result, DiffResult::FullRebuild));
    }

    #[test]
    fn test_diff_messages_full_rebuild_turn0_mismatch() {
        let msgs = vec![make_user_msg("u1_changed")];
        let hashes = extract_user_hashes(&msgs);

        let cached = make_cached("conv1", vec![CachedTurn {
            user_hashes: vec![12345u64], // mismatching hash
            assistant_uuid: "asst1".to_string(),
        }], hash_system(&None));

        let result = diff_messages(&cached, hash_system(&None), &hashes);
        assert!(matches!(result, DiffResult::FullRebuild));
    }

    #[test]
    fn test_diff_messages_fork() {
        let msgs = vec![make_user_msg("u1"), make_user_msg("u2"), make_user_msg("u3_edited")];
        let hashes = extract_user_hashes(&msgs);

        // turn 0 has u1, turn 1 has u2_original
        let u2_original_hash = hash_user_message(&make_user_msg("u2_original"));
        let cached = make_cached("conv1", vec![
            CachedTurn {
                user_hashes: vec![hashes[0].1],
                assistant_uuid: "asst0".to_string(),
            },
            CachedTurn {
                user_hashes: vec![u2_original_hash], // mismatch at turn 1
                assistant_uuid: "asst1".to_string(),
            },
        ], hash_system(&None));

        let result = diff_messages(&cached, hash_system(&None), &hashes);
        match result {
            DiffResult::Fork { parent_uuid, fork_turn_index, remaining_user_indices, .. } => {
                assert_eq!(parent_uuid, "asst0");
                assert_eq!(fork_turn_index, 1);
                // remaining starts from u2 in the new messages
                assert!(remaining_user_indices.contains(&1));
                assert!(remaining_user_indices.contains(&2));
            }
            _ => panic!("Expected Fork, got {:?}", result),
        }
    }

    #[test]
    fn test_diff_messages_full_rebuild_same_request() {
        let msgs = vec![make_user_msg("u1")];
        let hashes = extract_user_hashes(&msgs);
        let sys_hash = hash_system(&None);

        let cached = make_cached("conv1", vec![CachedTurn {
            user_hashes: vec![hashes[0].1],
            assistant_uuid: "asst1".to_string(),
        }], sys_hash);

        // Same messages, no new ones → FullRebuild
        let result = diff_messages(&cached, sys_hash, &hashes);
        assert!(matches!(result, DiffResult::FullRebuild));
    }

    #[test]
    fn test_diff_messages_full_rebuild_shorter_messages() {
        let msgs = vec![make_user_msg("u1")];
        let hashes = extract_user_hashes(&msgs);
        let sys_hash = hash_system(&None);

        // Cache has more turns than new messages
        let cached = make_cached("conv1", vec![
            CachedTurn {
                user_hashes: vec![hashes[0].1],
                assistant_uuid: "asst0".to_string(),
            },
            CachedTurn {
                user_hashes: vec![99999u64], // additional turn that new messages don't have
                assistant_uuid: "asst1".to_string(),
            },
        ], sys_hash);

        let result = diff_messages(&cached, sys_hash, &hashes);
        assert!(matches!(result, DiffResult::FullRebuild));
    }
}
