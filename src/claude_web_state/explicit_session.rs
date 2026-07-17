use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::{
    claude_web_state::conversation_cache::{ConversationCache, ExplicitSessionKey},
    protocol::ProtocolError,
    types::{
        claude::{Message, Role},
        claude_web::request::{
            ExplicitContentError, ExplicitContentErrorKind, normalize_explicit_message,
        },
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplicitSessionState {
    InFlight,
    Committed,
    Uncertain,
    Tombstoned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplicitTurn {
    pub parent_uuid_before: Option<String>,
    pub user_digests: Vec<String>,
    pub assistant_uuid_after: String,
    pub parent_timeline: Vec<String>,
    pub request_timeline: Vec<String>,
    pub assistant_digest_after: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingExplicitTurn {
    pub parent_uuid_before: Option<String>,
    pub user_digests: Vec<String>,
    pub assistant_uuid_after: String,
    pub replace_from_turn: usize,
    pub parent_timeline: Vec<String>,
    pub request_timeline: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplicitConversation {
    pub state: ExplicitSessionState,
    pub model_digest: String,
    pub system_digest: String,
    pub turns: Vec<ExplicitTurn>,
    pub pending: Option<PendingExplicitTurn>,
    #[serde(default)]
    pub file_mappings: HashMap<String, String>,
}

#[derive(Clone)]
pub struct ExplicitLifecycle {
    inner: Arc<ExplicitLifecycleInner>,
}

struct ExplicitLifecycleInner {
    cache: ConversationCache,
    key: ExplicitSessionKey,
    operation: Mutex<Option<OwnedMutexGuard<()>>>,
    finalized: AtomicBool,
}

impl ExplicitLifecycle {
    pub fn new(
        cache: ConversationCache,
        key: ExplicitSessionKey,
        operation: OwnedMutexGuard<()>,
    ) -> Self {
        Self {
            inner: Arc::new(ExplicitLifecycleInner {
                cache,
                key,
                operation: Mutex::new(Some(operation)),
                finalized: AtomicBool::new(false),
            }),
        }
    }

    pub async fn commit(&self, assistant_digest: Option<String>) -> Result<(), ProtocolError> {
        let mut operation = self.inner.operation.lock().await;
        if operation.is_none() {
            return Ok(());
        }
        self.inner
            .cache
            .commit_explicit_turn(&self.inner.key, assistant_digest)
            .await?;
        operation.take();
        self.inner.finalized.store(true, Ordering::Release);
        Ok(())
    }

    pub async fn uncertain(&self) -> Result<(), ProtocolError> {
        finalize_uncertain(&self.inner).await
    }
}

impl Drop for ExplicitLifecycle {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 || self.inner.finalized.load(Ordering::Acquire) {
            return;
        }
        let inner = self.inner.clone();
        tokio::spawn(async move {
            if let Err(error) = finalize_uncertain(&inner).await {
                tracing::warn!("Failed to persist explicit session uncertainty: {error}");
            }
        });
    }
}

async fn finalize_uncertain(inner: &ExplicitLifecycleInner) -> Result<(), ProtocolError> {
    let mut operation = inner.operation.lock().await;
    let result = if operation.is_some() {
        inner.cache.mark_explicit_uncertain(&inner.key).await
    } else {
        Ok(())
    };
    if result.is_err() {
        inner
            .cache
            .mark_explicit_uncertain_in_memory(&inner.key)
            .await;
    }
    operation.take();
    inner.finalized.store(true, Ordering::Release);
    result
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExplicitReusePlan {
    Create,
    Append {
        parent_uuid: String,
        suffix_start: usize,
    },
    Fork {
        parent_uuid: Option<String>,
        suffix_start: usize,
        replace_from_turn: usize,
    },
    Regenerate {
        parent_uuid: Option<String>,
        suffix_start: usize,
        replace_from_turn: usize,
    },
}

#[derive(Debug)]
pub struct DigestedMessages {
    pub users: Vec<(usize, String)>,
    pub timeline: Vec<String>,
}

pub fn digest_model(model: &str) -> String {
    digest_tagged(b"clewdr-model-v1\0", model.as_bytes())
}

pub fn digest_system(system: &Option<serde_json::Value>) -> String {
    let value = system.as_ref().unwrap_or(&serde_json::Value::Null);
    digest_json(value)
}

pub fn digest_messages(messages: &[Message]) -> Result<DigestedMessages, ProtocolError> {
    let mut users = Vec::new();
    let mut timeline = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        if message.role == Role::System {
            return Err(invalid_request(
                "Explicit sessions require system instructions in the top-level system field",
            ));
        }
        let digest = digest_message(message)?;
        let role = match message.role {
            Role::User => {
                users.push((index, digest.clone()));
                "user"
            }
            Role::Assistant => "assistant",
            Role::System => unreachable!(),
        };
        timeline.push(format!("{role}:{digest}"));
    }
    if users.is_empty() {
        return Err(invalid_request(
            "An explicit session request must contain user content",
        ));
    }
    Ok(DigestedMessages { users, timeline })
}

pub fn digest_assistant_output(text: &str) -> String {
    digest_json(&serde_json::json!([{ "type": "text", "text": text.trim() }]))
}

pub fn plan(
    explicit: Option<&ExplicitConversation>,
    user_digests: &[String],
    timeline: &[String],
    model_digest: &str,
    system_digest: &str,
) -> Result<ExplicitReusePlan, ProtocolError> {
    let Some(explicit) = explicit else {
        return Ok(ExplicitReusePlan::Create);
    };
    match explicit.state {
        ExplicitSessionState::InFlight | ExplicitSessionState::Uncertain => {
            return Err(ProtocolError::new(
                StatusCode::CONFLICT,
                "conversation_state_uncertain",
                "The previous session operation did not reach a verified terminal state",
            ));
        }
        ExplicitSessionState::Tombstoned => {
            return Err(ProtocolError::new(
                StatusCode::GONE,
                "conversation_expired",
                "The upstream conversation has expired and must be reset",
            ));
        }
        ExplicitSessionState::Committed => {}
    }
    if explicit.model_digest != model_digest || explicit.system_digest != system_digest {
        return Err(reuse_failed("Model or system prompt changed"));
    }
    let result = plan_turns(&explicit.turns, user_digests)?;
    validate_timeline(&explicit.turns, user_digests, timeline, &result)?;
    Ok(result)
}

pub fn parent_timeline(turns: &[ExplicitTurn], plan: &ExplicitReusePlan) -> Vec<String> {
    match plan {
        ExplicitReusePlan::Create => Vec::new(),
        ExplicitReusePlan::Append { .. } => {
            let latest = turns.last().expect("append requires a committed turn");
            let mut timeline = latest.request_timeline.clone();
            if let Some(digest) = &latest.assistant_digest_after {
                timeline.push(format!("assistant:{digest}"));
            }
            timeline
        }
        ExplicitReusePlan::Fork {
            replace_from_turn, ..
        }
        | ExplicitReusePlan::Regenerate {
            replace_from_turn, ..
        } => turns[*replace_from_turn].parent_timeline.clone(),
    }
}

fn plan_turns(
    turns: &[ExplicitTurn],
    requested: &[String],
) -> Result<ExplicitReusePlan, ProtocolError> {
    if turns.is_empty() {
        return Err(reuse_failed("Committed session has no turns"));
    }
    let committed = turns
        .iter()
        .flat_map(|turn| turn.user_digests.iter().cloned())
        .collect::<Vec<_>>();
    if requested.len() < committed.len() {
        return Err(reuse_failed("Request is shorter than committed history"));
    }
    if requested.starts_with(&committed) {
        if requested.len() > committed.len() {
            return Ok(ExplicitReusePlan::Append {
                parent_uuid: turns
                    .last()
                    .expect("non-empty")
                    .assistant_uuid_after
                    .clone(),
                suffix_start: committed.len(),
            });
        }
        let last = turns.last().expect("non-empty");
        return Ok(ExplicitReusePlan::Regenerate {
            parent_uuid: last.parent_uuid_before.clone(),
            suffix_start: committed.len() - last.user_digests.len(),
            replace_from_turn: turns.len() - 1,
        });
    }
    let mut offset = 0;
    for (turn_index, turn) in turns.iter().enumerate() {
        let end = offset + turn.user_digests.len();
        if requested.get(offset..end) != Some(turn.user_digests.as_slice()) {
            if turn_index == 0 {
                return Err(reuse_failed("Cannot fork inside bootstrap history"));
            }
            return Ok(ExplicitReusePlan::Fork {
                parent_uuid: turn.parent_uuid_before.clone(),
                suffix_start: offset,
                replace_from_turn: turn_index,
            });
        }
        offset = end;
    }
    Err(reuse_failed(
        "Request history is not aligned to committed turns",
    ))
}

fn validate_timeline(
    turns: &[ExplicitTurn],
    users: &[String],
    requested: &[String],
    plan: &ExplicitReusePlan,
) -> Result<(), ProtocolError> {
    let mut expected = parent_timeline(turns, plan);
    let suffix = match plan {
        ExplicitReusePlan::Append { suffix_start, .. }
        | ExplicitReusePlan::Fork { suffix_start, .. }
        | ExplicitReusePlan::Regenerate { suffix_start, .. } => *suffix_start,
        ExplicitReusePlan::Create => return Ok(()),
    };
    expected.extend(
        users[suffix..]
            .iter()
            .map(|digest| format!("user:{digest}")),
    );
    if expected != requested {
        return Err(reuse_failed(
            "Message order or assistant content cannot be forwarded from the selected parent",
        ));
    }
    Ok(())
}

fn digest_message(message: &Message) -> Result<String, ProtocolError> {
    normalize_explicit_message(message)
        .map(|normalized| digest_json(&normalized.identity))
        .map_err(content_error)
}

pub(super) fn content_error(error: ExplicitContentError) -> ProtocolError {
    match error.kind {
        ExplicitContentErrorKind::InvalidRequest => invalid_request(error.message),
    }
}

fn digest_json(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).expect("JSON values serialize");
    hex::encode(Sha256::digest(bytes))
}

fn digest_tagged(tag: &[u8], value: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(tag);
    digest.update(value);
    hex::encode(digest.finalize())
}

fn reuse_failed(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(StatusCode::CONFLICT, "conversation_reuse_failed", message)
}

fn invalid_request(message: impl Into<String>) -> ProtocolError {
    ProtocolError::new(
        StatusCode::BAD_REQUEST,
        "conversation_reuse_failed",
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::claude::Message;

    fn turn(parent: Option<&str>, users: &[&str], assistant: &str) -> ExplicitTurn {
        ExplicitTurn {
            parent_uuid_before: parent.map(str::to_owned),
            user_digests: users.iter().map(|value| (*value).to_owned()).collect(),
            assistant_uuid_after: assistant.to_owned(),
            parent_timeline: Vec::new(),
            request_timeline: users.iter().map(|value| format!("user:{value}")).collect(),
            assistant_digest_after: Some(format!("answer-{assistant}")),
        }
    }

    #[test]
    fn plans_append_fork_and_regenerate_at_turn_boundaries() {
        let turns = vec![turn(None, &["u1"], "a1"), turn(Some("a1"), &["u2"], "a2")];
        assert!(matches!(
            plan_turns(&turns, &["u1".into(), "u2".into(), "u3".into()]).unwrap(),
            ExplicitReusePlan::Append {
                suffix_start: 2,
                ..
            }
        ));
        assert!(matches!(
            plan_turns(&turns, &["u1".into(), "changed".into()]).unwrap(),
            ExplicitReusePlan::Fork {
                replace_from_turn: 1,
                ..
            }
        ));
        assert!(matches!(
            plan_turns(&turns, &["u1".into(), "u2".into()]).unwrap(),
            ExplicitReusePlan::Regenerate {
                replace_from_turn: 1,
                ..
            }
        ));
    }

    #[test]
    fn timeline_model_system_and_prefill_mismatches_are_rejected() {
        let explicit = ExplicitConversation {
            state: ExplicitSessionState::Committed,
            model_digest: "model".into(),
            system_digest: "system".into(),
            turns: vec![turn(None, &["u1"], "a1")],
            pending: None,
            file_mappings: Default::default(),
        };
        let reject = |users: &[&str], timeline: &[&str], model, system| {
            assert_eq!(
                plan(
                    Some(&explicit),
                    &users
                        .iter()
                        .map(|value| (*value).into())
                        .collect::<Vec<_>>(),
                    &timeline
                        .iter()
                        .map(|value| (*value).into())
                        .collect::<Vec<_>>(),
                    model,
                    system,
                )
                .unwrap_err()
                .code,
                "conversation_reuse_failed"
            );
        };
        let valid = ["user:u1", "assistant:answer-a1", "user:u2"];
        reject(
            &["u1", "u2"],
            &["user:u1", "assistant:changed", "user:u2"],
            "model",
            "system",
        );
        reject(&["u1", "u2"], &valid, "changed-model", "system");
        reject(&["u1", "u2"], &valid, "model", "changed-system");
        reject(
            &["u1"],
            &["user:u1", "assistant:prefill"],
            "model",
            "system",
        );
    }

    #[test]
    fn staged_file_reference_is_part_of_explicit_identity() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type":"container_upload", "file_id":"file_clewdr_v1_abc"}]
        }))
        .unwrap();
        assert!(digest_messages(&[message]).is_ok());
    }

    #[test]
    fn malformed_explicit_requests_are_bad_requests() {
        let invalid = [
            vec![Message::new_text(Role::System, "system")],
            vec![Message::new_text(Role::Assistant, "assistant")],
            vec![Message::new_text(Role::User, "   ")],
            vec![serde_json::from_value(serde_json::json!({
                "role":"user",
                "content":[{"type":"image_url", "image_url":{"url":"https://example.com/a.png"}}]
            }))
            .unwrap()],
        ];
        for messages in invalid {
            let error = digest_messages(&messages).unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.code, "conversation_reuse_failed");
        }
    }
}
