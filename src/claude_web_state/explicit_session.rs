use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::{
    claude_web_state::conversation_cache::{ConversationCache, ExplicitSessionKey},
    protocol::ProtocolError,
    types::claude::{Message, MessageContent, Role},
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

    pub async fn uncertain(&self) {
        let mut operation = self.inner.operation.lock().await;
        if operation.is_some() {
            self.inner
                .cache
                .mark_explicit_uncertain(&self.inner.key)
                .await;
        }
        operation.take();
        self.inner.finalized.store(true, Ordering::Release);
    }
}

impl Drop for ExplicitLifecycle {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 || self.inner.finalized.load(Ordering::Acquire) {
            return;
        }
        let lifecycle = self.clone();
        tokio::spawn(async move {
            lifecycle.uncertain().await;
        });
    }
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
            return Err(reuse_failed(
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
        return Err(reuse_failed(
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
    let value = match &message.content {
        MessageContent::Text { content } if !content.trim().is_empty() => {
            serde_json::json!([{ "type": "text", "text": content.trim() }])
        }
        MessageContent::Text { .. } => {
            return Err(reuse_failed(
                "Explicit session message has no forwardable content",
            ));
        }
        MessageContent::Blocks { content } => {
            serde_json::to_value(content).expect("content blocks serialize")
        }
    };
    reject_unsupported_content(&value)?;
    Ok(digest_json(&value))
}

fn reject_unsupported_content(value: &serde_json::Value) -> Result<(), ProtocolError> {
    let blocks = match value {
        serde_json::Value::String(text) if !text.trim().is_empty() => return Ok(()),
        serde_json::Value::Array(blocks) => blocks,
        _ => {
            return Err(reuse_failed(
                "Explicit session message has no forwardable content",
            ));
        }
    };
    for block in blocks {
        let kind = block
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        match kind {
            "text" | "image" | "image_url" | "document" | "container_upload" => {}
            _ => {
                return Err(reuse_failed(
                    "Explicit session content block is not forwardable",
                ));
            }
        }
        if contains_staged_reference(block) {
            return Err(ProtocolError::new(
                StatusCode::NOT_IMPLEMENTED,
                "staged_files_unavailable",
                "Staged file references require the staged-files extension",
            ));
        }
        if matches!(kind, "image" | "image_url") && contains_remote_url(block) {
            return Err(reuse_failed(
                "Explicit sessions do not support remote image URLs",
            ));
        }
    }
    Ok(())
}

fn contains_remote_url(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(value) => {
            value.starts_with("http://") || value.starts_with("https://")
        }
        serde_json::Value::Array(values) => values.iter().any(contains_remote_url),
        serde_json::Value::Object(values) => values.values().any(contains_remote_url),
        _ => false,
    }
}

fn contains_staged_reference(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(value) => value.starts_with("file_clewdr_v1_"),
        serde_json::Value::Array(values) => values.iter().any(contains_staged_reference),
        serde_json::Value::Object(values) => values.values().any(contains_staged_reference),
        _ => false,
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
    fn full_timeline_rejects_changed_assistant_content() {
        let mut first = turn(None, &["u1"], "a1");
        first.request_timeline = vec!["user:u1".into()];
        let explicit = ExplicitConversation {
            state: ExplicitSessionState::Committed,
            model_digest: "model".into(),
            system_digest: "system".into(),
            turns: vec![first],
            pending: None,
        };
        let error = plan(
            Some(&explicit),
            &["u1".into(), "u2".into()],
            &[
                "user:u1".into(),
                "assistant:changed".into(),
                "user:u2".into(),
            ],
            "model",
            "system",
        )
        .unwrap_err();
        assert_eq!(error.code, "conversation_reuse_failed");
    }

    #[test]
    fn model_system_and_assistant_prefill_mismatches_are_rejected() {
        let first = turn(None, &["u1"], "a1");
        let explicit = ExplicitConversation {
            state: ExplicitSessionState::Committed,
            model_digest: "model".into(),
            system_digest: "system".into(),
            turns: vec![first],
            pending: None,
        };
        assert_eq!(
            plan(
                Some(&explicit),
                &["u1".into(), "u2".into()],
                &[
                    "user:u1".into(),
                    "assistant:answer-a1".into(),
                    "user:u2".into()
                ],
                "changed-model",
                "system",
            )
            .unwrap_err()
            .code,
            "conversation_reuse_failed"
        );
        assert_eq!(
            plan(
                Some(&explicit),
                &["u1".into(), "u2".into()],
                &[
                    "user:u1".into(),
                    "assistant:answer-a1".into(),
                    "user:u2".into()
                ],
                "model",
                "changed-system",
            )
            .unwrap_err()
            .code,
            "conversation_reuse_failed"
        );
        assert_eq!(
            plan(
                Some(&explicit),
                &["u1".into()],
                &["user:u1".into(), "assistant:prefill".into()],
                "model",
                "system",
            )
            .unwrap_err()
            .code,
            "conversation_reuse_failed"
        );
    }

    #[test]
    fn staged_file_reference_is_explicitly_unavailable() {
        let message: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": [{"type":"container_upload", "file_id":"file_clewdr_v1_abc"}]
        }))
        .unwrap();
        assert_eq!(
            digest_messages(&[message]).unwrap_err().code,
            "staged_files_unavailable"
        );
    }
}
