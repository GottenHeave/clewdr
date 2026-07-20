use std::sync::Arc;

use colored::Colorize;
use futures::{StreamExt, TryFutureExt};
use serde_json::json;
use sha2::{Digest, Sha256};
use snafu::ResultExt;
use tracing::{Instrument, debug, error, info, info_span, warn};
use wreq::{Method, Response, header::ACCEPT};

use super::{ClaudeWebState, PendingCacheWrite};
use crate::{
    claude_web_state::conversation_cache::{
        CachedConversation, CachedTurn, ConversationCache, ExplicitOperationPermit,
        ExplicitResponseSnapshot, ExplicitSessionKey,
    },
    claude_web_state::diff::{self, DiffResult, extract_user_hashes, hash_system},
    claude_web_state::explicit_session::{
        ExplicitConversation, ExplicitLifecycle, ExplicitReusePlan, ExplicitSessionState,
        PendingExplicitTurn, content_error, digest_messages, digest_model, digest_system,
        parent_timeline, plan,
    },
    config::CLEWDR_CONFIG,
    error::{CheckClaudeErr, ClewdrError, WreqSnafu},
    protocol::{ProtocolError, parse_session_id},
    types::claude::{CreateMessageParams, ImageSource, Message},
    types::claude_web::request::{
        Attachment, CreateConversationParams, TurnMessageUuids, normalize_explicit_message,
    },
    utils::{TIME_ZONE, print_out_json},
};

const MAX_EXPLICIT_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

struct PreservedFileReferencesGuard {
    cache: ConversationCache,
    files: Arc<crate::protocol_files::StagedFileStore>,
    key: ExplicitSessionKey,
    armed: bool,
}

impl PreservedFileReferencesGuard {
    fn new(
        cache: ConversationCache,
        files: Arc<crate::protocol_files::StagedFileStore>,
        key: ExplicitSessionKey,
    ) -> Self {
        Self {
            cache,
            files,
            key,
            armed: true,
        }
    }

    async fn reconcile_exclusive(&mut self) -> Result<(), ProtocolError> {
        reconcile_explicit_file_references(&self.cache, &self.files, &self.key).await?;
        self.armed = false;
        Ok(())
    }

    async fn reconcile_after_operation(&mut self) -> Result<(), ProtocolError> {
        let operation = self.cache.lock_explicit_operation(&self.key).await;
        let result = self.reconcile_exclusive().await;
        drop(operation);
        result
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PreservedFileReferencesGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let cache = self.cache.clone();
        let files = self.files.clone();
        let key = self.key.clone();
        tokio::spawn(async move {
            let operation = cache.lock_explicit_operation(&key).await;
            if let Err(error) = reconcile_explicit_file_references(&cache, &files, &key).await {
                warn!("Failed to reconcile staged files after cancelled recovery: {error}");
            }
            drop(operation);
        });
    }
}

async fn reconcile_explicit_file_references(
    cache: &ConversationCache,
    files: &crate::protocol_files::StagedFileStore,
    key: &ExplicitSessionKey,
) -> Result<(), ProtocolError> {
    let _files = cache.lock_explicit_files().await;
    let staged_file_ids = cache.explicit_staged_file_ids(key).await;
    files
        .set_session_references(&key.session_ref(), &staged_file_ids)
        .await
}

fn explicit_request_fingerprint(state: &ClaudeWebState, request: &CreateMessageParams) -> String {
    let mut digest = Sha256::new();
    digest.update(b"clewdr-explicit-request-v1\0");
    digest.update(state.api_format.to_string().as_bytes());
    digest.update([u8::from(state.stream)]);
    digest.update(serde_json::to_vec(request).expect("request parameters serialize"));
    hex::encode(digest.finalize())
}

fn replay_response(snapshot: ExplicitResponseSnapshot) -> axum::response::Response {
    let mut response = axum::response::Response::new(axum::body::Body::from(snapshot.body));
    *response.status_mut() = snapshot.status;
    *response.headers_mut() = snapshot.headers;
    response
}

async fn materialize_explicit_response(
    response: axum::response::Response,
) -> Result<(axum::response::Response, ExplicitResponseSnapshot), ClewdrError> {
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let headers = parts.headers.clone();
    let mut stream = body.into_data_stream();
    let mut buffered = bytes::BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| {
            ProtocolError::new(
                http::StatusCode::BAD_GATEWAY,
                "conversation_state_uncertain",
                "Claude Web response ended before a verified completion",
            )
        })?;
        if buffered.len().saturating_add(chunk.len()) > MAX_EXPLICIT_RESPONSE_BYTES {
            return Err(ProtocolError::new(
                http::StatusCode::BAD_GATEWAY,
                "explicit_response_too_large",
                format!(
                    "Explicit session response exceeds the {MAX_EXPLICIT_RESPONSE_BYTES} byte buffer limit"
                ),
            )
            .into());
        }
        buffered.extend_from_slice(&chunk);
    }
    let body = buffered.freeze();
    let snapshot = ExplicitResponseSnapshot {
        status,
        headers,
        body: body.clone(),
    };
    Ok((
        axum::response::Response::from_parts(parts, axum::body::Body::from(body)),
        snapshot,
    ))
}

fn recoverable_explicit_error(error: &ClewdrError) -> Option<&'static str> {
    let ClewdrError::Protocol { source } = error else {
        return None;
    };
    is_recoverable_protocol_error(source).then_some(source.code)
}

fn is_recoverable_protocol_error(error: &ProtocolError) -> bool {
    matches!(
        (error.status, error.code),
        (http::StatusCode::CONFLICT, "conversation_reuse_failed")
            | (http::StatusCode::CONFLICT, "conversation_state_uncertain")
            | (
                http::StatusCode::BAD_GATEWAY,
                "conversation_state_uncertain"
            )
            | (http::StatusCode::GONE, "conversation_expired")
    )
}

fn explicit_recovery_still_needed(
    existing: Option<&CachedConversation>,
    request: &CreateMessageParams,
) -> Result<bool, ProtocolError> {
    let Some(explicit) = existing.and_then(|conversation| conversation.explicit.as_ref()) else {
        return Ok(false);
    };
    let digested = digest_messages(&request.messages)?;
    let user_digests = digested
        .users
        .iter()
        .map(|(_, digest)| digest.clone())
        .collect::<Vec<_>>();
    match plan(
        Some(explicit),
        &user_digests,
        &digested.timeline,
        &digest_model(&request.model),
        &digest_system(&request.system),
    ) {
        Ok(_) => Ok(false),
        Err(error) if is_recoverable_protocol_error(&error) => Ok(true),
        Err(error) => Err(error),
    }
}

/// User content split into the Claude Web prompt, text attachments, and images.
struct BundledMessages {
    /// Short text stays in the prompt; long text is represented by an attachment.
    prompt: String,
    /// Text documents extracted from user content.
    attachments: Vec<Attachment>,
    /// Images uploaded before the completion request.
    #[allow(dead_code)]
    images: Vec<ImageSource>,
}

#[derive(Clone)]
struct PreparedExplicitTurn {
    conversation_uuid: String,
    human_uuid: String,
    assistant_uuid: String,
    cache_write: PendingCacheWrite,
}

struct PreparedExplicitOperation<'a> {
    reuse: &'a ExplicitReusePlan,
    existing: Option<&'a CachedConversation>,
    parent_uuid: Option<String>,
    replace_from_turn: usize,
    user_indices: Vec<usize>,
    user_hashes: Vec<u64>,
    request: &'a CreateMessageParams,
    turn: PreparedExplicitTurn,
    model_digest: String,
    system_digest: String,
    user_digests: Vec<String>,
    parent_timeline: Vec<String>,
    request_timeline: Vec<String>,
}

struct IncrementalTurn<'a> {
    organization_uuid: &'a str,
    conversation_uuid: &'a str,
    parent_uuid: Option<&'a str>,
    human_uuid: &'a str,
    assistant_uuid: &'a str,
}

fn model_selector_state_body(p: &CreateMessageParams) -> serde_json::Value {
    let mut body = json!({ "model": p.model });
    if let (Some(effort), Some(mode)) = (p.web_thinking_effort(), p.web_thinking_mode()) {
        body["thinking"] = json!({
            "type": "effort_and_mode",
            "effort": effort,
            "mode": mode,
        });
    }
    body
}

fn create_conversation_params(
    p: &CreateMessageParams,
    is_temporary: bool,
    is_pro: bool,
) -> CreateConversationParams {
    CreateConversationParams {
        name: if is_temporary {
            String::new()
        } else {
            format!("ClewdR-{}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"))
        },
        model: p.model.clone(),
        include_conversation_preferences: true,
        paprika_mode: p
            .web_thinking_mode()
            .is_some_and(|mode| mode == crate::types::claude::ThinkingMode::Auto && is_pro)
            .then(|| "auto".to_string()),
        compass_mode: None,
        tool_search_mode: "auto".to_string(),
        is_temporary,
        enabled_imagine: true,
    }
}

impl ClaudeWebState {
    /// Routes explicit metadata sessions to the lifecycle path and retries implicit requests.
    /// Cache validation happens before any upload or completion side effect.
    pub async fn try_chat(
        &mut self,
        p: CreateMessageParams,
    ) -> Result<axum::response::Response, ClewdrError> {
        let session_id = p
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.fields.get("user_id"))
            .map(String::as_str);
        let session_digest = parse_session_id(session_id).map_err(|error| {
            ProtocolError::new(
                http::StatusCode::BAD_REQUEST,
                "invalid_session_id",
                error.to_string(),
            )
        })?;
        if let Some(session_digest) = session_digest {
            return self.try_explicit_chat(p, session_digest).await;
        }
        for i in 0..CLEWDR_CONFIG.load().max_retries + 1 {
            if i > 0 {
                info!("[RETRY] attempt: {}", i.to_string().green());
            }
            let mut state = self.to_owned();
            let p = p.to_owned();

            let cookie = state.request_cookie().await?;
            let web_res = async {
                state.bootstrap().await?;
                state.send_chat(p).await
            };
            let transform_res = web_res
                .and_then(async |r| self.transform_response(r).await)
                .instrument(info_span!("claude_web", "cookie" = cookie.cookie.mask()));

            match transform_res.await {
                Ok(b) => {
                    // Commit cache state only after the response has been transformed successfully.
                    if let Some(pending) = state.pending_cache_write.take() {
                        state.commit_cache_write(pending).await;
                    }

                    if let Err(e) = state.clean_chat().await {
                        warn!("Failed to clean chat: {}", e);
                    }
                    return Ok(b);
                }
                Err(e) => {
                    // Failed implicit requests cannot be reused on the next retry.
                    state.conv_cache.invalidate(&state.cache_key()).await;
                    state.pending_cache_write = None;

                    if let Err(e) = state.clean_chat().await {
                        warn!("Failed to clean chat: {}", e);
                    }
                    error!("{e}");
                    if let ClewdrError::InvalidCookie { reason } = e {
                        state.return_cookie(Some(reason.to_owned())).await;
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        error!("Max retries exceeded");
        Err(ClewdrError::TooManyRetries)
    }

    /// Plans and stages an explicit operation before invoking the existing send paths.
    async fn try_explicit_chat(
        &mut self,
        p: CreateMessageParams,
        session_digest: String,
    ) -> Result<axum::response::Response, ClewdrError> {
        let principal = self.principal.clone().ok_or(ClewdrError::InvalidAuth)?;
        let key = ExplicitSessionKey::new(principal.as_str(), &session_digest);
        let fingerprint = explicit_request_fingerprint(self, &p);
        let mut recovery_attempted = false;
        let mut preserved_file_references: Option<PreservedFileReferencesGuard> = None;
        let mut operation = None;
        loop {
            match self
                .try_explicit_chat_once(
                    p.clone(),
                    session_digest.clone(),
                    operation.take(),
                    preserved_file_references.is_some(),
                )
                .await
            {
                Ok(response) => {
                    if let Some(mut references) = preserved_file_references.take() {
                        references.disarm();
                    }
                    return Ok(response);
                }
                Err(error @ ClewdrError::InvalidCookie { .. }) => {
                    let permit = self
                        .conv_cache
                        .lock_explicit_request(&key, fingerprint.clone())
                        .await;
                    if let Some(response) = permit.observed_replay() {
                        if let Some(mut references) = preserved_file_references.take() {
                            references.reconcile_after_operation().await?;
                        }
                        return Ok(replay_response(response));
                    }
                    if recovery_attempted {
                        if let Some(mut references) = preserved_file_references.take() {
                            references.reconcile_exclusive().await?;
                        }
                        return Err(error);
                    }
                    let ClewdrError::InvalidCookie { reason } = &error else {
                        unreachable!()
                    };
                    self.return_cookie(Some(reason.clone())).await;
                    preserved_file_references = self.reset_explicit_for_rebuild(&key).await?;
                    self.pause_after_explicit_reset().await;
                    self.clear_explicit_attempt_state();
                    recovery_attempted = true;
                    operation = Some(permit);
                }
                Err(error) => {
                    let Some(code) = recoverable_explicit_error(&error) else {
                        if let Some(mut references) = preserved_file_references.take() {
                            references.reconcile_after_operation().await?;
                        }
                        return Err(error);
                    };
                    warn!(
                        "[SESSION] recovering {code} by rebuilding explicit session {}",
                        session_digest
                    );
                    #[cfg(test)]
                    if let Some(barrier) = &self.explicit_recovery_barrier {
                        barrier.wait().await;
                    }
                    let permit = self
                        .conv_cache
                        .lock_explicit_request(&key, fingerprint.clone())
                        .await;
                    if let Some(response) = permit.observed_replay() {
                        if let Some(mut references) = preserved_file_references.take() {
                            references.reconcile_after_operation().await?;
                        }
                        return Ok(replay_response(response));
                    }
                    if recovery_attempted {
                        if let Some(mut references) = preserved_file_references.take() {
                            references.reconcile_exclusive().await?;
                        }
                        return Err(error);
                    }
                    recovery_attempted = true;
                    let existing = self.conv_cache.get_explicit(&key).await;
                    let reset = explicit_recovery_still_needed(existing.as_ref(), &p)?;
                    if reset {
                        preserved_file_references = self.reset_explicit_for_rebuild(&key).await?;
                        self.pause_after_explicit_reset().await;
                    }
                    self.clear_explicit_attempt_state();
                    operation = Some(permit);
                }
            }
        }
    }

    fn clear_explicit_attempt_state(&mut self) {
        self.explicit_lifecycle = None;
        self.explicit_file_key = None;
        self.pending_cache_write = None;
        self.explicit_completion_started = false;
        self.conv_uuid = None;
    }

    #[cfg(test)]
    async fn pause_after_explicit_reset(&self) {
        if let Some((reached, release)) = &self.explicit_after_reset {
            let released = release.notified();
            reached.notify_one();
            released.await;
        }
    }

    #[cfg(not(test))]
    async fn pause_after_explicit_reset(&self) {}

    async fn reset_explicit_for_rebuild(
        &self,
        key: &ExplicitSessionKey,
    ) -> Result<Option<PreservedFileReferencesGuard>, ProtocolError> {
        let mut references = self.staged_files.as_ref().map(|files| {
            PreservedFileReferencesGuard::new(self.conv_cache.clone(), files.clone(), key.clone())
        });
        let _files = self.conv_cache.lock_explicit_files().await;
        if let Err(error) = self.conv_cache.reset_explicit(key).await {
            if let Some(references) = &mut references {
                references.disarm();
            }
            return Err(error);
        }
        Ok(references)
    }

    async fn try_explicit_chat_once(
        &mut self,
        p: CreateMessageParams,
        session_digest: String,
        operation: Option<ExplicitOperationPermit>,
        reconcile_file_references: bool,
    ) -> Result<axum::response::Response, ClewdrError> {
        let principal = self.principal.clone().ok_or(ClewdrError::InvalidAuth)?;
        let key = ExplicitSessionKey::new(principal.as_str(), &session_digest);
        let fingerprint = explicit_request_fingerprint(self, &p);
        self.explicit_file_key = Some(key.clone());
        self.explicit_completion_started = false;
        let operation = match operation {
            Some(operation) => operation,
            None => {
                self.conv_cache
                    .lock_explicit_request(&key, fingerprint.clone())
                    .await
            }
        };
        if let Some(response) = operation.observed_replay() {
            return Ok(replay_response(response));
        }
        let (operation, replay) = operation.into_operation();
        let replay = replay.expect("request operation owns a replay generation");
        let digested = digest_messages(&p.messages)?;
        let user_digests = digested
            .users
            .iter()
            .map(|(_, digest)| digest.clone())
            .collect::<Vec<_>>();
        let model_digest = digest_model(&p.model);
        let system_digest = digest_system(&p.system);
        let existing = self.conv_cache.get_explicit(&key).await;
        let reuse = plan(
            existing
                .as_ref()
                .and_then(|conversation| conversation.explicit.as_ref()),
            &user_digests,
            &digested.timeline,
            &model_digest,
            &system_digest,
        )?;
        let selected_parent_timeline = existing
            .as_ref()
            .and_then(|conversation| conversation.explicit.as_ref())
            .map(|explicit| parent_timeline(&explicit.turns, &reuse))
            .unwrap_or_default();

        let cookie = self
            .request_session_cookie(
                &session_digest,
                existing
                    .as_ref()
                    .map(|conversation| conversation.cookie_id.as_str()),
            )
            .await;
        if matches!(cookie, Err(ClewdrError::NoCookieAvailable)) && existing.is_some() {
            self.conv_cache.tombstone_explicit(&key).await?;
            return Err(ProtocolError::new(
                http::StatusCode::GONE,
                "conversation_expired",
                "The session Cookie is no longer available",
            )
            .into());
        }
        cookie?;
        self.bootstrap().await?;
        let organization_uuid = self.org_uuid.clone().ok_or(ClewdrError::UnexpectedNone {
            msg: "Organization UUID is not set",
        })?;
        if let Some(existing) = &existing
            && explicit_binding_error(existing, &self.cookie_id(), &organization_uuid).is_some()
        {
            self.conv_cache.tombstone_explicit(&key).await?;
            return Err(ProtocolError::new(
                http::StatusCode::GONE,
                "conversation_expired",
                "The persisted Cookie or organization is no longer available",
            )
            .into());
        }

        let (suffix_start, replace_from_turn, parent_uuid) = match &reuse {
            ExplicitReusePlan::Create => (0, 0, None),
            ExplicitReusePlan::Append {
                parent_uuid,
                suffix_start,
            } => (
                *suffix_start,
                existing
                    .as_ref()
                    .and_then(|conversation| conversation.explicit.as_ref())
                    .map(|explicit| explicit.turns.len())
                    .unwrap_or_default(),
                Some(parent_uuid.clone()),
            ),
            ExplicitReusePlan::Fork {
                parent_uuid,
                suffix_start,
                replace_from_turn,
            }
            | ExplicitReusePlan::Regenerate {
                parent_uuid,
                suffix_start,
                replace_from_turn,
            } => (*suffix_start, *replace_from_turn, parent_uuid.clone()),
        };
        let hashes = extract_user_hashes(&p.messages);
        let indices = digested.users[suffix_start..]
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();
        let suffix_hashes = hashes[suffix_start..]
            .iter()
            .map(|(_, hash)| *hash)
            .collect::<Vec<_>>();

        let turn = self.prepare_explicit_turn(
            &reuse,
            existing.as_ref(),
            replace_from_turn,
            &suffix_hashes,
            &p,
            &organization_uuid,
        );
        let prepared = PreparedExplicitOperation {
            reuse: &reuse,
            existing: existing.as_ref(),
            parent_uuid,
            replace_from_turn,
            user_indices: indices,
            user_hashes: suffix_hashes,
            request: &p,
            turn,
            model_digest,
            system_digest,
            user_digests: user_digests[suffix_start..].to_vec(),
            parent_timeline: selected_parent_timeline,
            request_timeline: digested.timeline,
        };
        self.stage_explicit_operation(&key, &prepared).await?;

        let send_result = self.send_explicit_operation(&prepared).await;
        self.pending_cache_write.take();
        let response = self
            .finish_explicit_send(&key, existing.is_some(), send_result)
            .await?;
        let lifecycle = ExplicitLifecycle::new(self.conv_cache.clone(), key.clone(), operation);
        self.explicit_lifecycle = Some(lifecycle.clone());
        match self.transform_response(response).await {
            Ok(response) => match materialize_explicit_response(response).await {
                Ok((response, snapshot)) => {
                    if reconcile_file_references {
                        let Some(files) = &self.staged_files else {
                            unreachable!("file reconciliation requires staged storage")
                        };
                        if let Err(error) =
                            reconcile_explicit_file_references(&self.conv_cache, files, &key).await
                        {
                            replay.fail();
                            lifecycle.release().await;
                            return Err(error.into());
                        }
                    }
                    #[cfg(test)]
                    if let Some((reached, release)) = &self.explicit_before_replay {
                        let released = release.notified();
                        reached.notify_one();
                        released.await;
                    }
                    replay.complete(snapshot);
                    lifecycle.release().await;
                    Ok(response)
                }
                Err(error) => {
                    replay.fail();
                    lifecycle.release().await;
                    Err(error)
                }
            },
            Err(error) => {
                replay.fail();
                lifecycle.release().await;
                Err(error)
            }
        }
    }

    async fn finish_explicit_send(
        &self,
        key: &ExplicitSessionKey,
        existing_session: bool,
        result: Result<Response, ClewdrError>,
    ) -> Result<Response, ClewdrError> {
        match result {
            Ok(response) => Ok(response),
            Err(ClewdrError::ClaudeHttpError { code, .. })
                if code == http::StatusCode::NOT_FOUND || code == http::StatusCode::GONE =>
            {
                self.conv_cache.tombstone_explicit(key).await?;
                Err(ProtocolError::new(
                    http::StatusCode::GONE,
                    "conversation_expired",
                    "The upstream conversation no longer exists",
                )
                .into())
            }
            Err(error) => {
                if existing_session && !self.explicit_completion_started {
                    self.conv_cache.restore_explicit_committed(key).await?;
                } else {
                    self.conv_cache.mark_explicit_uncertain(key).await?;
                }
                Err(error)
            }
        }
    }

    async fn send_explicit_operation(
        &mut self,
        operation: &PreparedExplicitOperation<'_>,
    ) -> Result<Response, ClewdrError> {
        match operation.reuse {
            ExplicitReusePlan::Create => {
                self.send_full(operation.request.clone(), true, Some(&operation.turn))
                    .await
            }
            ExplicitReusePlan::Append { parent_uuid, .. } => {
                self.send_incremental(
                    operation.existing.expect("append requires cache"),
                    parent_uuid,
                    &operation.user_indices,
                    &operation.user_hashes,
                    operation.request,
                    Some(&operation.turn),
                )
                .await
            }
            ExplicitReusePlan::Fork { .. } | ExplicitReusePlan::Regenerate { .. } => {
                self.send_incremental_fork(
                    operation.existing.expect("reuse requires cache"),
                    operation.parent_uuid.as_deref(),
                    operation.replace_from_turn,
                    &operation.user_indices,
                    &operation.user_hashes,
                    operation.request,
                    Some(&operation.turn),
                )
                .await
            }
        }
    }

    fn prepare_explicit_turn(
        &self,
        reuse: &ExplicitReusePlan,
        existing: Option<&CachedConversation>,
        replace_from_turn: usize,
        user_hashes: &[u64],
        p: &CreateMessageParams,
        organization_uuid: &str,
    ) -> PreparedExplicitTurn {
        let conversation_uuid = existing
            .map(|conversation| conversation.conv_uuid.clone())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let assistant_uuid = uuid::Uuid::new_v4().to_string();
        let cache_write = match reuse {
            ExplicitReusePlan::Create => PendingCacheWrite::Init {
                key: self.cache_key_for(p),
                conv: Box::new(CachedConversation {
                    conv_uuid: conversation_uuid.clone(),
                    org_uuid: organization_uuid.to_owned(),
                    cookie_id: self.cookie_id(),
                    model: p.model.clone(),
                    is_pro: self.is_pro(),
                    system_hash: hash_system(&p.system),
                    turns: vec![CachedTurn {
                        user_hashes: user_hashes.to_vec(),
                        assistant_uuid: assistant_uuid.clone(),
                        model: Some(p.model.clone()),
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                    explicit: None,
                }),
            },
            ExplicitReusePlan::Append { .. } => PendingCacheWrite::AppendTurn {
                key: self.cache_key_for(p),
                turn: CachedTurn {
                    user_hashes: user_hashes.to_vec(),
                    assistant_uuid: assistant_uuid.clone(),
                    model: Some(p.model.clone()),
                },
            },
            ExplicitReusePlan::Fork { .. } | ExplicitReusePlan::Regenerate { .. } => {
                PendingCacheWrite::ForkAndAppend {
                    key: self.cache_key_for(p),
                    fork_turn_index: replace_from_turn,
                    turn: CachedTurn {
                        user_hashes: user_hashes.to_vec(),
                        assistant_uuid: assistant_uuid.clone(),
                        model: Some(p.model.clone()),
                    },
                }
            }
        };
        PreparedExplicitTurn {
            conversation_uuid,
            human_uuid: uuid::Uuid::new_v4().to_string(),
            assistant_uuid,
            cache_write,
        }
    }

    async fn stage_explicit_operation(
        &self,
        key: &ExplicitSessionKey,
        operation: &PreparedExplicitOperation<'_>,
    ) -> Result<(), ProtocolError> {
        let (assistant_uuid_after, initial) = match operation.turn.cache_write.clone() {
            PendingCacheWrite::Init { conv, .. } => {
                let assistant_uuid = conv
                    .turns
                    .last()
                    .expect("initial cache write has a turn")
                    .assistant_uuid
                    .clone();
                (assistant_uuid, Some(conv))
            }
            PendingCacheWrite::AppendTurn { turn, .. }
            | PendingCacheWrite::ForkAndAppend { turn, .. } => (turn.assistant_uuid, None),
        };
        let pending = PendingExplicitTurn {
            model: Some(operation.request.model.clone()),
            model_digest: Some(operation.model_digest.clone()),
            parent_uuid_before: operation.parent_uuid.clone(),
            user_digests: operation.user_digests.clone(),
            assistant_uuid_after,
            replace_from_turn: operation.replace_from_turn,
            parent_timeline: operation.parent_timeline.clone(),
            request_timeline: operation.request_timeline.clone(),
        };
        if let Some(mut conversation) = initial {
            conversation.turns.clear();
            conversation.explicit = Some(ExplicitConversation {
                state: ExplicitSessionState::InFlight,
                model_digest: operation.model_digest.clone(),
                system_digest: operation.system_digest.clone(),
                turns: Vec::new(),
                pending: Some(pending),
                file_mappings: Default::default(),
            });
            self.conv_cache
                .set_explicit_checked(key.clone(), *conversation)
                .await?;
        } else {
            self.conv_cache.stage_explicit_turn(key, pending).await?;
        }
        Ok(())
    }

    async fn send_chat(&mut self, p: CreateMessageParams) -> Result<Response, ClewdrError> {
        let _org_uuid = self
            .org_uuid
            .to_owned()
            .ok_or(ClewdrError::UnexpectedNone {
                msg: "Organization UUID is not set",
            })?;

        let can_reuse =
            CLEWDR_CONFIG.load().reuse_conversation && !CLEWDR_CONFIG.load().preserve_chats;

        if can_reuse {
            if let Some(result) = self.try_reuse_conversation(&p).await {
                match result {
                    Ok(response) => return Ok(response),
                    Err(e) => {
                        warn!("Reuse failed, falling back to full: {}", e);
                        self.conv_cache.invalidate(&self.cache_key_for(&p)).await;
                    }
                }
            }
        }

        self.send_full(p, can_reuse, None).await
    }

    async fn try_reuse_conversation(
        &mut self,
        p: &CreateMessageParams,
    ) -> Option<Result<Response, ClewdrError>> {
        let key = self.cache_key_for(p);
        let cached = self.conv_cache.get(&key).await?;

        if cached.cookie_id != self.cookie_id() {
            info!("[CACHE] cookie mismatch, invalidating");
            self.conv_cache.invalidate(&key).await;
            return None;
        }
        if cached.model != p.model {
            info!(
                "[CACHE] switching model for conversation: {} -> {}",
                cached.model, p.model
            );
        }
        if cached.is_pro != self.is_pro() {
            info!("[CACHE] pro status changed");
            self.conv_cache.invalidate(&key).await;
            return None;
        }

        let user_hashes = extract_user_hashes(&p.messages);
        let sys_hash = hash_system(&p.system);

        let diff = diff::diff_messages(&cached, sys_hash, &user_hashes);

        match diff {
            DiffResult::Append {
                parent_uuid,
                new_user_indices,
                new_user_hashes,
            } => {
                info!(
                    "[CACHE HIT] appending {} new user message(s)",
                    new_user_indices.len()
                );
                let result = self
                    .send_incremental(
                        &cached,
                        &parent_uuid,
                        &new_user_indices,
                        &new_user_hashes,
                        p,
                        None,
                    )
                    .await;
                Some(result)
            }
            DiffResult::Fork {
                parent_uuid,
                fork_turn_index,
                remaining_user_indices,
                remaining_user_hashes,
            } => {
                info!(
                    "[CACHE FORK] forking at turn {}, {} user message(s)",
                    fork_turn_index,
                    remaining_user_indices.len()
                );
                let result = self
                    .send_incremental_fork(
                        &cached,
                        Some(&parent_uuid),
                        fork_turn_index,
                        &remaining_user_indices,
                        &remaining_user_hashes,
                        p,
                        None,
                    )
                    .await;
                Some(result)
            }
            DiffResult::FullRebuild => {
                info!("[CACHE MISS] full rebuild required");
                self.conv_cache.invalidate(&key).await;
                None // fall through to send_full
            }
        }
    }

    /// Sends a new conversation request, including UUID generation and file uploads.
    async fn send_full(
        &mut self,
        p: CreateMessageParams,
        write_cache: bool,
        prepared: Option<&PreparedExplicitTurn>,
    ) -> Result<Response, ClewdrError> {
        let org_uuid = self
            .org_uuid
            .to_owned()
            .ok_or(ClewdrError::UnexpectedNone {
                msg: "Organization UUID is not set",
            })?;
        let user_hashes = extract_user_hashes(&p.messages)
            .into_iter()
            .map(|(_, hash)| hash)
            .collect::<Vec<_>>();
        let generated = prepared.is_none().then(|| {
            self.prepare_explicit_turn(
                &ExplicitReusePlan::Create,
                None,
                0,
                &user_hashes,
                &p,
                &org_uuid,
            )
        });
        let prepared = prepared.or(generated.as_ref()).unwrap();

        self.sync_model_selector_state(&p).await?;

        // The client-generated UUID is sent with the first completion and becomes the
        // conversation identity used by subsequent incremental requests.
        let new_uuid = prepared.conversation_uuid.clone();
        let is_temporary = !CLEWDR_CONFIG.load().preserve_chats;
        self.conv_uuid = Some(new_uuid.clone());
        self.last_params = Some(p.clone());
        debug!("Generated conversation UUID: {}", new_uuid);

        let mut body = self
            .transform_request(p.clone())
            .ok_or(ClewdrError::BadRequest {
                msg: "Request body is empty",
            })?;
        body.create_conversation_params =
            Some(create_conversation_params(&p, is_temporary, self.is_pro()));

        let human_uuid = prepared.human_uuid.clone();
        let assistant_uuid = prepared.assistant_uuid.clone();
        body.turn_message_uuids = Some(TurnMessageUuids {
            human_message_uuid: human_uuid.clone(),
            assistant_message_uuid: assistant_uuid.clone(),
        });

        if write_cache {
            self.pending_cache_write = Some(prepared.cache_write.clone());
        }

        // Upload images before completion; a failed upload must not produce an upstream turn.
        let images = body.images.drain(..).collect::<Vec<_>>();
        let files = self.upload_files(images, &org_uuid, &new_uuid).await?;
        body.files = files;

        print_out_json(&body, "claude_web_clewdr_req.json");
        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{org_uuid}/chat_conversations/{new_uuid}/completion"
            ))
            .expect("Url parse error");

        self.explicit_completion_started = true;
        let response = self
            .build_request(Method::POST, endpoint)
            .json(&body)
            .header(ACCEPT, "text/event-stream")
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to send chat request",
            })?
            .check_claude()
            .await?;

        Ok(response)
    }

    /// Appends new user messages to a validated conversation parent.
    async fn send_incremental(
        &mut self,
        cached: &CachedConversation,
        parent_uuid: &str,
        new_user_indices: &[usize],
        new_user_hashes: &[u64],
        p: &CreateMessageParams,
        prepared: Option<&PreparedExplicitTurn>,
    ) -> Result<Response, ClewdrError> {
        self.conv_uuid = Some(cached.conv_uuid.clone());
        self.last_params = Some(p.clone());
        let is_explicit = prepared.is_some();
        let generated = prepared.is_none().then(|| {
            self.prepare_explicit_turn(
                &ExplicitReusePlan::Append {
                    parent_uuid: parent_uuid.to_owned(),
                    suffix_start: 0,
                },
                Some(cached),
                0,
                new_user_hashes,
                p,
                &cached.org_uuid,
            )
        });
        let prepared = prepared.or(generated.as_ref()).unwrap();

        self.sync_model_selector_state(p).await?;

        let need_thinking = p
            .web_thinking_mode()
            .is_some_and(|mode| mode == crate::types::claude::ThinkingMode::Auto)
            && self.is_pro();
        self.update_paprika(&cached.conv_uuid, need_thinking).await;

        // Only the suffix selected by cache diff is forwarded; cached history stays upstream.
        let new_user_msgs: Vec<&Message> = new_user_indices
            .iter()
            .map(|&idx| &p.messages[idx])
            .collect();

        let bundled =
            self.bundle_user_messages(&new_user_msgs, if is_explicit { "\n" } else { "\n\n" })?;

        let human_uuid = prepared.human_uuid.clone();
        let assistant_uuid = prepared.assistant_uuid.clone();

        let body = self
            .build_incremental_body(
                &bundled,
                IncrementalTurn {
                    organization_uuid: &cached.org_uuid,
                    conversation_uuid: &cached.conv_uuid,
                    parent_uuid: Some(parent_uuid),
                    human_uuid: &human_uuid,
                    assistant_uuid: &assistant_uuid,
                },
                p,
            )
            .await?;

        self.pending_cache_write = Some(prepared.cache_write.clone());

        print_out_json(&body, "claude_web_incremental_req.json");

        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}/completion",
                cached.org_uuid, cached.conv_uuid
            ))
            .expect("Url parse error");

        self.explicit_completion_started = true;
        let response = self
            .build_request(Method::POST, endpoint)
            .json(&body)
            .header(ACCEPT, "text/event-stream")
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to send incremental chat",
            })?
            .check_claude()
            .await?;

        Ok(response)
    }

    /// Sends an edited suffix from a parent UUID after truncating the cached timeline.
    async fn send_incremental_fork(
        &mut self,
        cached: &CachedConversation,
        parent_uuid: Option<&str>,
        fork_turn_index: usize,
        remaining_user_indices: &[usize],
        remaining_user_hashes: &[u64],
        p: &CreateMessageParams,
        prepared: Option<&PreparedExplicitTurn>,
    ) -> Result<Response, ClewdrError> {
        self.conv_uuid = Some(cached.conv_uuid.clone());
        self.last_params = Some(p.clone());
        let is_explicit = prepared.is_some();
        let generated = prepared.is_none().then(|| {
            self.prepare_explicit_turn(
                &ExplicitReusePlan::Fork {
                    parent_uuid: parent_uuid.map(str::to_owned),
                    suffix_start: 0,
                    replace_from_turn: fork_turn_index,
                },
                Some(cached),
                fork_turn_index,
                remaining_user_hashes,
                p,
                &cached.org_uuid,
            )
        });
        let prepared = prepared.or(generated.as_ref()).unwrap();

        self.sync_model_selector_state(p).await?;

        let need_thinking = p
            .web_thinking_mode()
            .is_some_and(|mode| mode == crate::types::claude::ThinkingMode::Auto)
            && self.is_pro();
        self.update_paprika(&cached.conv_uuid, need_thinking).await;

        let remaining_user_msgs: Vec<&Message> = remaining_user_indices
            .iter()
            .map(|&idx| &p.messages[idx])
            .collect();

        let bundled = self.bundle_user_messages(
            &remaining_user_msgs,
            if is_explicit { "\n" } else { "\n\n" },
        )?;

        let human_uuid = prepared.human_uuid.clone();
        let assistant_uuid = prepared.assistant_uuid.clone();

        let body = self
            .build_incremental_body(
                &bundled,
                IncrementalTurn {
                    organization_uuid: &cached.org_uuid,
                    conversation_uuid: &cached.conv_uuid,
                    parent_uuid,
                    human_uuid: &human_uuid,
                    assistant_uuid: &assistant_uuid,
                },
                p,
            )
            .await?;

        self.pending_cache_write = Some(prepared.cache_write.clone());

        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}/completion",
                cached.org_uuid, cached.conv_uuid
            ))
            .expect("Url parse error");

        self.explicit_completion_started = true;
        let response = self
            .build_request(Method::POST, endpoint)
            .json(&body)
            .header(ACCEPT, "text/event-stream")
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to send forked chat",
            })?
            .check_claude()
            .await?;

        Ok(response)
    }

    async fn update_paprika(&self, conv_uuid: &str, need_thinking: bool) {
        // Claude Web keeps paprika mode on the conversation, so synchronize it before completion.
        let paprika = if need_thinking {
            "auto".into()
        } else {
            json!(null)
        };
        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}",
                self.org_uuid.as_ref().unwrap(),
                conv_uuid
            ))
            .expect("Url parse error");
        let body = json!({ "settings": { "paprika_mode": paprika } });
        let _ = self
            .build_request(Method::PUT, endpoint)
            .json(&body)
            .send()
            .await;
    }

    async fn sync_model_selector_state(&self, p: &CreateMessageParams) -> Result<(), ClewdrError> {
        if !self.is_pro() {
            return Ok(());
        }
        let org_uuid = self.org_uuid.as_ref().ok_or(ClewdrError::UnexpectedNone {
            msg: "Organization UUID is not set",
        })?;
        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{org_uuid}/model_selector_state/chat"
            ))
            .map_err(|e| ClewdrError::Whatever {
                message: format!("Parse URL error: {e}"),
                source: Some(Box::new(e)),
            })?;
        let body = model_selector_state_body(p);
        // Model selector state is a pro-account side request and must precede the completion.

        self.build_request(Method::PATCH, endpoint)
            .json(&body)
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to sync Claude Web model selector",
            })?
            .check_claude()
            .await?;

        Ok(())
    }

    /// Builds the incremental body, preserving model, tools, thinking, and paprika settings.
    async fn build_incremental_body(
        &mut self,
        bundled: &BundledMessages,
        turn: IncrementalTurn<'_>,
        p: &CreateMessageParams,
    ) -> Result<serde_json::Value, ClewdrError> {
        let files = self
            .upload_files(
                bundled.images.clone(),
                turn.organization_uuid,
                turn.conversation_uuid,
            )
            .await?;
        let mut body = json!({
            "prompt": bundled.prompt,
            "timezone": TIME_ZONE.to_string(),
            "turn_message_uuids": {
                "human_message_uuid": turn.human_uuid,
                "assistant_message_uuid": turn.assistant_uuid,
            },
            "attachments": bundled.attachments,
            "files": files,
            "rendering_mode": if p.stream.unwrap_or_default() { "messages" } else { "raw" },
        });
        if let Some(parent_uuid) = turn.parent_uuid {
            body["parent_message_uuid"] = json!(parent_uuid);
        }
        // Pro requests carry the model; all requests preserve effort and thinking controls.
        if self.is_pro() {
            body["model"] = json!(p.model);
        }
        if let Some(effort) = p.web_thinking_effort() {
            body["effort"] = json!(effort);
        }
        if let Some(mode) = p.web_thinking_mode() {
            body["thinking_mode"] = json!(mode);
        }
        let mut tools = vec![];
        if CLEWDR_CONFIG.load().web_search {
            tools.push(json!({"type": "web_search_v0", "name": "web_search"}));
        }
        // Web search is opt-in and is forwarded only when enabled in ClewdR configuration.
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        Ok(body)
    }

    /// Extracts text/documents/images and applies the short-prompt versus attachment boundary.
    fn bundle_user_messages(
        &self,
        user_msgs: &[&Message],
        message_separator: &str,
    ) -> Result<BundledMessages, ProtocolError> {
        let mut texts: Vec<String> = vec![];
        let mut attachments: Vec<Attachment> = vec![];
        let mut images: Vec<ImageSource> = vec![];

        for msg in user_msgs {
            let normalized = normalize_explicit_message(msg).map_err(content_error)?;
            if !normalized.text_blocks.is_empty() {
                texts.push(normalized.text_blocks.join("\n"));
            }
            attachments.extend(normalized.attachments);
            images.extend(normalized.images);
        }

        // Explicit sessions use one newline for consecutive users; implicit callers pass two.
        let combined = texts.join(message_separator);

        const PROMPT_THRESHOLD: usize = 4000;

        // Keep short text in the prompt; long text becomes one attachment for Claude Web.
        if combined.len() <= PROMPT_THRESHOLD {
            let mut prompt = combined;
            if prompt.is_empty() && (!attachments.is_empty() || !images.is_empty()) {
                prompt = "Please answer using the attached content.".to_string();
            }
            Ok(BundledMessages {
                prompt,
                attachments,
                images,
            })
        } else {
            attachments.push(Attachment::new(combined));
            let mut p_str = CLEWDR_CONFIG.load().custom_prompt.clone();
            if p_str.is_empty() {
                p_str = "Please answer using the attached content.".to_string();
            }
            Ok(BundledMessages {
                prompt: p_str,
                attachments,
                images,
            })
        }
    }

    async fn commit_cache_write(&self, pending: PendingCacheWrite) {
        match pending {
            PendingCacheWrite::Init { key, conv } => {
                info!("[CACHE] initialized for conv {}", conv.conv_uuid);
                self.conv_cache.set(key, *conv).await;
            }
            PendingCacheWrite::AppendTurn { key, turn } => {
                info!("[CACHE] appended turn (assistant={})", turn.assistant_uuid);
                self.conv_cache.append_turn(&key, turn).await;
            }
            PendingCacheWrite::ForkAndAppend {
                key,
                fork_turn_index,
                turn,
            } => {
                info!(
                    "[CACHE] forked at turn {}, new assistant={}",
                    fork_turn_index, turn.assistant_uuid
                );
                self.conv_cache
                    .fork_and_append(&key, fork_turn_index, turn)
                    .await;
            }
        }
    }
}

fn explicit_binding_error(
    existing: &CachedConversation,
    cookie_id: &str,
    organization_uuid: &str,
) -> Option<ProtocolError> {
    (existing.cookie_id != cookie_id || existing.org_uuid != organization_uuid).then(|| {
        ProtocolError::new(
            http::StatusCode::GONE,
            "conversation_expired",
            "The persisted Cookie or organization is no longer available",
        )
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use crate::claude_web_state::conversation_cache::{
        ConversationCache, explicit_test_conversation, explicit_test_seed, explicit_test_state,
    };
    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::{Request, State},
        http::{StatusCode, header::CONTENT_TYPE},
        response::Response as AxumResponse,
    };
    use serde_json::json;

    use super::*;
    use crate::config::{CLEWDR_CONFIG, ClewdrConfig};
    use crate::types::claude::{
        ContentBlock, ImageSource, Metadata, OutputConfig, OutputEffort, Role, Thinking,
    };

    static CONFIG_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn serve(app: Router) -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url::Url::parse(&format!("http://{address}/")).unwrap()
    }

    fn explicit_conversation(state: ExplicitSessionState) -> CachedConversation {
        let mut conversation = explicit_test_conversation(state);
        conversation.explicit.as_mut().unwrap().model_digest = digest_model(&conversation.model);
        conversation.explicit.as_mut().unwrap().system_digest = digest_system(&None);
        conversation
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        path: String,
        body: serde_json::Value,
    }

    #[derive(Clone, Default)]
    struct MockClaudeServer {
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        uploads: Arc<AtomicUsize>,
        fail_second_upload: bool,
        incomplete_completion: bool,
        incomplete_first_completion: bool,
        fail_first_completion_gone: bool,
        always_fail_completion_gone: bool,
        completion_delay: std::time::Duration,
        fail_first_bootstrap_invalid: bool,
        bootstrap_calls: Arc<AtomicUsize>,
        response_padding_bytes: usize,
    }

    impl MockClaudeServer {
        async fn start(&self) -> url::Url {
            serve(
                Router::new()
                    .fallback(mock_claude_request)
                    .with_state(self.clone()),
            )
            .await
        }

        fn recorded(&self) -> Vec<RecordedRequest> {
            self.requests.lock().unwrap().clone()
        }

        fn completions(&self) -> Vec<RecordedRequest> {
            self.recorded()
                .into_iter()
                .filter(|request| request.path.contains("/completion"))
                .collect()
        }
    }

    async fn mock_claude_request(
        State(mock): State<MockClaudeServer>,
        request: Request,
    ) -> AxumResponse {
        let path = request.uri().path().to_owned();
        let bytes = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        mock.requests.lock().unwrap().push(RecordedRequest {
            path: path.clone(),
            body,
        });
        let (content_type, status, body) = if path == "/api/bootstrap" {
            let bootstrap = mock.bootstrap_calls.fetch_add(1, Ordering::SeqCst);
            if mock.fail_first_bootstrap_invalid && bootstrap == 0 {
                return AxumResponse::builder()
                    .status(StatusCode::OK)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(json!({"account":null}).to_string()))
                    .unwrap();
            }
            (
                "application/json",
                StatusCode::OK,
                json!({"account":{"email_address":"test@example.com","memberships":[{"organization":{"capabilities":["chat"]}}]}}).to_string(),
            )
        } else if path == "/api/organizations" {
            (
                "application/json",
                StatusCode::OK,
                json!([{"uuid":"org","capabilities":["chat"],"active_flags":[]}]).to_string(),
            )
        } else if path.contains("/upload-file") {
            let upload = mock.uploads.fetch_add(1, Ordering::SeqCst);
            if mock.fail_second_upload && upload == 1 {
                (
                    "text/plain",
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "upload failed".into(),
                )
            } else {
                (
                    "application/json",
                    StatusCode::OK,
                    "{\"file_uuid\":\"uploaded-file\"}".into(),
                )
            }
        } else if path.contains("/completion") {
            if !mock.completion_delay.is_zero() {
                tokio::time::sleep(mock.completion_delay).await;
            }
            if mock.always_fail_completion_gone
                || (mock.fail_first_completion_gone && mock.completions().len() == 1)
            {
                return AxumResponse::builder()
                    .status(StatusCode::GONE)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({"error":{"type":"not_found","message":"missing"}}).to_string(),
                    ))
                    .unwrap();
            }
            let stop = if mock.incomplete_completion
                || (mock.incomplete_first_completion && mock.completions().len() == 1)
            {
                ""
            } else {
                "data: {\"type\":\"message_stop\"}\n\n"
            };
            let padding = if mock.response_padding_bytes == 0 {
                String::new()
            } else {
                format!(
                    "data: {}\n\n",
                    json!({"type":"padding","padding":"x".repeat(mock.response_padding_bytes)})
                )
            };
            (
                "text/event-stream",
                StatusCode::OK,
                format!("data: {{\"completion\":\"answer\"}}\n\n{padding}{stop}"),
            )
        } else {
            ("application/json", StatusCode::OK, "{}".into())
        };
        AxumResponse::builder()
            .status(status)
            .header(CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap()
    }

    async fn mock_endpoint() -> (url::Url, MockClaudeServer) {
        let mock = MockClaudeServer::default();
        let endpoint = mock.start().await;
        (endpoint, mock)
    }

    struct ConfigRestore(crate::config::ClewdrConfig);

    impl Drop for ConfigRestore {
        fn drop(&mut self) {
            let config = self.0.clone();
            crate::config::CLEWDR_CONFIG.rcu(|_| config.clone());
        }
    }

    fn params(messages: Vec<Message>) -> CreateMessageParams {
        CreateMessageParams {
            model: "claude-sonnet-4-6".into(),
            messages,
            ..Default::default()
        }
    }

    fn session_request(digest: &str, messages: Vec<Message>) -> CreateMessageParams {
        let mut request = params(messages);
        request.metadata = Some(Metadata {
            fields: [("user_id".into(), format!("cherry_topic_v1_{digest}"))]
                .into_iter()
                .collect(),
        });
        request
    }

    fn session_request_with_model(
        digest: &str,
        messages: Vec<Message>,
        model: &str,
    ) -> CreateMessageParams {
        let mut request = params(messages);
        request.model = model.into();
        request.metadata = Some(Metadata {
            fields: [("user_id".into(), format!("cherry_topic_v1_{digest}"))]
                .into_iter()
                .collect(),
        });
        request
    }

    async fn stage_file(
        store: &crate::protocol_files::StagedFileStore,
        principal: &crate::protocol::AuthPrincipal,
        name: &str,
        bytes: &'static [u8],
    ) -> crate::protocol_files::FileResponse {
        store
            .stage_stream(
                principal,
                name,
                "application/octet-stream",
                futures::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from_static(bytes))]),
            )
            .await
            .unwrap()
    }

    async fn send_session(state: &mut ClaudeWebState, digest: &str, messages: Vec<Message>) {
        state
            .try_chat(session_request(digest, messages))
            .await
            .unwrap();
    }

    async fn configured_explicit_state(
        mock: &MockClaudeServer,
    ) -> (
        ConfigRestore,
        crate::services::cookie_actor::CookieActorHandle,
        ConversationCache,
        crate::protocol::AuthPrincipal,
    ) {
        let endpoint = mock.start().await;
        let original = crate::config::CLEWDR_CONFIG.load().as_ref().clone();
        let restore = ConfigRestore(original.clone());
        crate::config::CLEWDR_CONFIG.rcu(|_| {
            let mut config = original.clone();
            config.rproxy = Some(endpoint.clone());
            config.no_fs = true;
            config.skip_non_pro = false;
            config.skip_normal_pro = false;
            config
        });
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cookie =
            crate::config::CookieStatus::new(&format!("{}-bbbbbbAA", "a".repeat(86)), None)
                .unwrap();
        handle.submit(cookie).await.unwrap();
        tokio::task::yield_now().await;
        (
            restore,
            handle,
            ConversationCache::new(),
            crate::protocol::AuthPrincipal::for_authenticated_user(),
        )
    }

    #[tokio::test]
    async fn explicit_session_state_errors_rebuild_once_without_reaching_the_client() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer::default();
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let user = Message::new_text(Role::User, "hello");

        for (index, state) in [
            ExplicitSessionState::Uncertain,
            ExplicitSessionState::Committed,
            ExplicitSessionState::Tombstoned,
        ]
        .into_iter()
        .enumerate()
        {
            let digest = format!("{index:02x}").repeat(32);
            let key = ExplicitSessionKey::new(principal.as_str(), &digest);
            cache
                .set_explicit_checked(key.clone(), explicit_conversation(state))
                .await
                .unwrap();
            let mut request_state = ClaudeWebState::new(handle.clone(), cache.clone());
            request_state.principal = Some(principal.clone());

            let response = request_state
                .try_chat(session_request(&digest, vec![user.clone()]))
                .await
                .unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                explicit_test_state(&cache, &key).await,
                ExplicitSessionState::Committed
            );
        }

        assert_eq!(mock.completions().len(), 3);
    }

    #[tokio::test]
    async fn concurrent_waiters_share_one_state_recovery() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            completion_delay: std::time::Duration::from_millis(50),
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let invoke = |handle, cache, principal, request: CreateMessageParams, recovery_barrier| async move {
            let mut state = ClaudeWebState::new(handle, cache);
            state.principal = Some(principal);
            state.explicit_recovery_barrier = Some(recovery_barrier);
            let response = state.try_chat(request).await.unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
        };

        for (index, session_state) in [
            ExplicitSessionState::Uncertain,
            ExplicitSessionState::Tombstoned,
        ]
        .into_iter()
        .enumerate()
        {
            let digest = format!("e{index}").repeat(32);
            let key = ExplicitSessionKey::new(principal.as_str(), &digest);
            cache
                .set_explicit_checked(key.clone(), explicit_conversation(session_state))
                .await
                .unwrap();
            let request =
                session_request(&digest, vec![Message::new_text(Role::User, "concurrent")]);
            let expected_completions = index + 1;
            let recovery_barrier = Arc::new(tokio::sync::Barrier::new(2));
            let first = tokio::spawn(invoke(
                handle.clone(),
                cache.clone(),
                principal.clone(),
                request.clone(),
                recovery_barrier.clone(),
            ));
            let second = tokio::spawn(invoke(
                handle.clone(),
                cache.clone(),
                principal.clone(),
                request,
                recovery_barrier,
            ));
            let (first, second) = tokio::join!(first, second);

            assert_eq!(first.unwrap(), second.unwrap());
            assert_eq!(mock.completions().len(), expected_completions);
            assert_eq!(
                explicit_test_state(&cache, &key).await,
                ExplicitSessionState::Committed
            );
        }
    }

    #[tokio::test]
    async fn invalid_explicit_request_does_not_reset_committed_session() {
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cache = ConversationCache::new();
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let digest = "aa".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        cache
            .set_explicit_checked(
                key.clone(),
                explicit_conversation(ExplicitSessionState::Committed),
            )
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal);

        let error = state
            .try_chat(session_request(
                &digest,
                vec![Message::new_text(Role::System, "invalid")],
            ))
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol error");
        };

        assert_eq!(source.status, StatusCode::BAD_REQUEST);
        assert!(cache.get_explicit(&key).await.is_some());
    }

    #[tokio::test]
    async fn upstream_expiration_rebuilds_once() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            fail_first_completion_gone: true,
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "ba".repeat(32);
        let mut state = ClaudeWebState::new(handle, cache);
        state.principal = Some(principal);

        let response = state
            .try_chat(session_request(
                &digest,
                vec![Message::new_text(Role::User, "hello")],
            ))
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        assert_eq!(mock.completions().len(), 2);
    }

    #[tokio::test]
    async fn invalid_bound_cookie_rebuilds_once_with_another_cookie() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            fail_first_bootstrap_invalid: true,
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let replacement =
            crate::config::CookieStatus::new(&format!("{}-bbbbbbAA", "b".repeat(86)), None)
                .unwrap();
        handle.submit(replacement).await.unwrap();
        tokio::task::yield_now().await;
        let digest = "bd".repeat(32);
        let mut state = ClaudeWebState::new(handle, cache);
        state.principal = Some(principal);

        let response = state
            .try_chat(session_request(
                &digest,
                vec![Message::new_text(Role::User, "hello")],
            ))
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        assert_eq!(mock.bootstrap_calls.load(Ordering::SeqCst), 2);
        assert_eq!(mock.completions().len(), 1);
    }

    #[tokio::test]
    async fn rebuild_preserves_files_until_exact_reference_reconciliation() {
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let digest = "be".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let cache = ConversationCache::new();
        let directory = tempfile::tempdir().unwrap();
        let files =
            crate::protocol_files::StagedFileStore::persistent_with_limits(directory.path(), 1, 2)
                .await
                .unwrap();
        let a = stage_file(&files, &principal, "a", b"a").await;
        let b = stage_file(&files, &principal, "b", b"b").await;
        let session_ref = key.session_ref();
        let mut existing = explicit_conversation(ExplicitSessionState::Committed);
        existing.explicit.as_mut().unwrap().file_mappings.extend([
            (a.id.clone(), "upstream-a".into()),
            (b.id.clone(), "upstream-b".into()),
        ]);
        cache
            .set_explicit_checked(key.clone(), existing)
            .await
            .unwrap();
        files.add_reference(&a.id, &session_ref).await.unwrap();
        files.add_reference(&b.id, &session_ref).await.unwrap();
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal.clone());
        state.staged_files = Some(files.clone());

        let mut references = state
            .reset_explicit_for_rebuild(&key)
            .await
            .unwrap()
            .unwrap();
        let blocked = files
            .stage_stream(
                &principal,
                "c",
                "application/octet-stream",
                futures::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"c"))]),
            )
            .await
            .unwrap_err();
        assert_eq!(blocked.code, "staged_storage_full");

        let mut rebuilt = explicit_conversation(ExplicitSessionState::Committed);
        rebuilt
            .explicit
            .as_mut()
            .unwrap()
            .file_mappings
            .insert(a.id.clone(), "replacement-a".into());
        cache
            .set_explicit_checked(key.clone(), rebuilt)
            .await
            .unwrap();
        references.reconcile_exclusive().await.unwrap();
        assert!(files.has_session_reference(&a.id, &session_ref).await);
        assert!(!files.has_session_reference(&b.id, &session_ref).await);
        stage_file(&files, &principal, "c", b"c").await;

        let mut references = state
            .reset_explicit_for_rebuild(&key)
            .await
            .unwrap()
            .unwrap();
        assert!(files.has_session_reference(&a.id, &session_ref).await);
        references.reconcile_exclusive().await.unwrap();
        assert!(!files.has_session_reference(&a.id, &session_ref).await);
    }

    #[tokio::test]
    async fn cancelled_rebuild_removes_orphaned_file_references() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer::default();
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "bf".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let directory = tempfile::tempdir().unwrap();
        let files = crate::protocol_files::StagedFileStore::persistent(directory.path())
            .await
            .unwrap();
        let staged = stage_file(&files, &principal, "cancelled", b"a").await;
        let mut existing = explicit_conversation(ExplicitSessionState::Uncertain);
        existing
            .explicit
            .as_mut()
            .unwrap()
            .file_mappings
            .insert(staged.id.clone(), "upstream".into());
        cache
            .set_explicit_checked(key.clone(), existing)
            .await
            .unwrap();
        files
            .add_reference(&staged.id, &key.session_ref())
            .await
            .unwrap();
        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let reached_wait = reached.notified();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal);
        state.staged_files = Some(files.clone());
        state.explicit_after_reset = Some((reached.clone(), release));
        let request = session_request(
            &digest,
            vec![Message::new_text(Role::User, "cancel recovery")],
        );
        let request = tokio::spawn(async move { state.try_chat(request).await });
        reached_wait.await;
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while files
                .has_session_reference(&staged.id, &key.session_ref())
                .await
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(cache.get_explicit(&key).await.is_none());
    }

    #[tokio::test]
    async fn cancelled_rebuild_reconciles_after_a_new_operation() {
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let key = ExplicitSessionKey::new(principal.as_str(), "cancel-race");
        let cache = ConversationCache::new();
        let directory = tempfile::tempdir().unwrap();
        let files = crate::protocol_files::StagedFileStore::persistent(directory.path())
            .await
            .unwrap();
        let a = stage_file(&files, &principal, "a", b"a").await;
        let b = stage_file(&files, &principal, "b", b"b").await;
        let session_ref = key.session_ref();
        files.add_reference(&a.id, &session_ref).await.unwrap();
        files.add_reference(&b.id, &session_ref).await.unwrap();
        let operation = cache.lock_explicit_operation(&key).await;
        drop(PreservedFileReferencesGuard::new(
            cache.clone(),
            files.clone(),
            key.clone(),
        ));

        let mut replacement = explicit_conversation(ExplicitSessionState::Committed);
        replacement
            .explicit
            .as_mut()
            .unwrap()
            .file_mappings
            .insert(a.id.clone(), "replacement".into());
        cache
            .set_explicit_checked(key.clone(), replacement)
            .await
            .unwrap();
        tokio::task::yield_now().await;
        assert!(files.has_session_reference(&b.id, &session_ref).await);
        drop(operation);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while files.has_session_reference(&b.id, &session_ref).await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(files.has_session_reference(&a.id, &session_ref).await);
    }

    #[tokio::test]
    async fn incomplete_response_rebuilds_once() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            incomplete_first_completion: true,
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "bb".repeat(32);
        let mut state = ClaudeWebState::new(handle, cache);
        state.principal = Some(principal);

        let response = state
            .try_chat(session_request(
                &digest,
                vec![Message::new_text(Role::User, "hello")],
            ))
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();

        assert_eq!(mock.completions().len(), 2);
    }

    #[tokio::test]
    async fn oversized_rebuild_response_releases_operation_and_file_references() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            response_padding_bytes: MAX_EXPLICIT_RESPONSE_BYTES + 1,
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "c0".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let directory = tempfile::tempdir().unwrap();
        let files = crate::protocol_files::StagedFileStore::persistent(directory.path())
            .await
            .unwrap();
        let staged = stage_file(&files, &principal, "oversized", b"a").await;
        let mut existing = explicit_conversation(ExplicitSessionState::Uncertain);
        existing
            .explicit
            .as_mut()
            .unwrap()
            .file_mappings
            .insert(staged.id.clone(), "upstream".into());
        cache
            .set_explicit_checked(key.clone(), existing)
            .await
            .unwrap();
        files
            .add_reference(&staged.id, &key.session_ref())
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal);
        state.staged_files = Some(files.clone());
        state.stream = true;
        let mut request = session_request(&digest, vec![Message::new_text(Role::User, "large")]);
        request.stream = Some(true);

        let error = state.try_chat(request).await.unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol error")
        };
        assert_eq!(source.status, StatusCode::BAD_GATEWAY);
        assert_eq!(source.code, "explicit_response_too_large");
        assert!(
            !files
                .has_session_reference(&staged.id, &key.session_ref())
                .await
        );
        let operation = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            cache.lock_explicit_operation(&key),
        )
        .await
        .unwrap();
        drop(operation);
        assert_eq!(mock.completions().len(), 1);
    }

    #[tokio::test]
    async fn repeated_expiration_stops_after_one_rebuild() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            always_fail_completion_gone: true,
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "bc".repeat(32);
        let mut state = ClaudeWebState::new(handle, cache);
        state.principal = Some(principal);

        let error = state
            .try_chat(session_request(
                &digest,
                vec![Message::new_text(Role::User, "hello")],
            ))
            .await
            .unwrap_err();

        assert_eq!(
            recoverable_explicit_error(&error),
            Some("conversation_expired")
        );
        assert_eq!(mock.completions().len(), 2);
    }

    #[tokio::test]
    async fn concurrent_identical_session_requests_share_one_completion() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            completion_delay: std::time::Duration::from_millis(50),
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let invoke = |handle, cache, principal, request: CreateMessageParams, stream| async move {
            let mut state = ClaudeWebState::new(handle, cache);
            state.principal = Some(principal);
            state.stream = stream;
            let response = state.try_chat(request).await.unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
        };
        for (index, stream) in [false, true].into_iter().enumerate() {
            let digest = format!("c{index}").repeat(32);
            let mut request =
                session_request(&digest, vec![Message::new_text(Role::User, "concurrent")]);
            request.stream = Some(stream);
            let expected_count = index + 1;
            let first = tokio::spawn(invoke(
                handle.clone(),
                cache.clone(),
                principal.clone(),
                request.clone(),
                stream,
            ));
            while mock.completions().len() < expected_count {
                tokio::task::yield_now().await;
            }
            let second = tokio::spawn(invoke(
                handle.clone(),
                cache.clone(),
                principal.clone(),
                request,
                stream,
            ));
            let (first, second) = tokio::join!(first, second);

            assert_eq!(first.unwrap(), second.unwrap());
        }
        assert_eq!(mock.completions().len(), 2);
    }

    #[tokio::test]
    async fn identical_waiter_after_message_stop_replays_active_generation() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer::default();
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "ce".repeat(32);
        let request = session_request(&digest, vec![Message::new_text(Role::User, "message stop")]);
        let reached = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let reached_wait = reached.notified();
        let first_handle = handle.clone();
        let first_cache = cache.clone();
        let first_principal = principal.clone();
        let first_request = request.clone();
        let first_reached = reached.clone();
        let first_release = release.clone();
        let first = tokio::spawn(async move {
            let mut state = ClaudeWebState::new(first_handle, first_cache);
            state.principal = Some(first_principal);
            state.explicit_before_replay = Some((first_reached, first_release));
            let response = state.try_chat(first_request).await.unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
        });
        reached_wait.await;

        let second = tokio::spawn(async move {
            let mut state = ClaudeWebState::new(handle, cache);
            state.principal = Some(principal);
            let response = state.try_chat(request).await.unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap()
        });
        tokio::task::yield_now().await;
        assert!(!second.is_finished());
        release.notify_one();
        let (first, second) = tokio::join!(first, second);

        assert_eq!(first.unwrap(), second.unwrap());
        assert_eq!(mock.completions().len(), 1);
    }

    #[tokio::test]
    async fn concurrent_different_session_requests_run_in_order() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer {
            completion_delay: std::time::Duration::from_millis(50),
            ..Default::default()
        };
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "da".repeat(32);
        let first_request = session_request(&digest, vec![Message::new_text(Role::User, "first")]);
        let second_request = session_request(
            &digest,
            vec![
                Message::new_text(Role::User, "first"),
                Message::new_text(Role::Assistant, "answer"),
                Message::new_text(Role::User, "second"),
            ],
        );
        let invoke = |handle, cache, principal, request| async move {
            let mut state = ClaudeWebState::new(handle, cache);
            state.principal = Some(principal);
            let response = state.try_chat(request).await.unwrap();
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        };
        let first = tokio::spawn(invoke(
            handle.clone(),
            cache.clone(),
            principal.clone(),
            first_request,
        ));
        while mock.completions().is_empty() {
            tokio::task::yield_now().await;
        }
        let second = tokio::spawn(invoke(handle, cache, principal, second_request));
        let (first, second) = tokio::join!(first, second);

        first.unwrap();
        second.unwrap();
        assert_eq!(mock.completions().len(), 2);
    }

    #[tokio::test]
    async fn try_chat_reuses_conversation_and_stops_before_unpersisted_side_effects() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let (endpoint, requests) = mock_endpoint().await;
        let original = crate::config::CLEWDR_CONFIG.load().as_ref().clone();
        let _restore = ConfigRestore(original.clone());
        crate::config::CLEWDR_CONFIG.rcu(|_| {
            let mut config = original.clone();
            config.rproxy = Some(endpoint.clone());
            config.no_fs = true;
            config.skip_non_pro = false;
            config.skip_normal_pro = false;
            config
        });
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cookie =
            crate::config::CookieStatus::new(&format!("{}-bbbbbbAA", "a".repeat(86)), None)
                .unwrap();
        handle.submit(cookie).await.unwrap();
        tokio::task::yield_now().await;
        let cache = ConversationCache::new();
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let digest = "ab".repeat(32);
        let mut state = ClaudeWebState::new(handle.clone(), cache.clone());
        state.principal = Some(principal.clone());
        let user = |text| Message::new_text(Role::User, text);
        let assistant = || Message::new_text(Role::Assistant, "answer");
        for messages in [
            vec![user("u1")],
            vec![user("u1"), assistant(), user("u2")],
            vec![user("u1"), assistant(), user("changed")],
            vec![user("u1"), assistant(), user("changed")],
        ] {
            send_session(&mut state, &digest, messages).await;
        }
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let cached = cache.get_explicit(&key).await.unwrap();
        let conversation_uuid = cached.conv_uuid.clone();
        let first_assistant_uuid = cached.turns[0].assistant_uuid.clone();

        let completion_bodies = |from| {
            requests.completions()[from..]
                .iter()
                .map(|request| request.body.clone())
                .collect::<Vec<_>>()
        };
        for messages in [
            vec![user("first"), user("second")],
            vec![
                user("first"),
                user("second"),
                assistant(),
                user("first"),
                user("second"),
            ],
        ] {
            send_session(&mut state, &"ce".repeat(32), messages).await;
        }
        let consecutive = completion_bodies(4);
        assert_eq!(consecutive[0]["prompt"], "first\nsecond");
        assert_eq!(consecutive[0]["prompt"], consecutive[1]["prompt"]);

        let rich = || {
            serde_json::from_value::<Message>(json!({"role":"user","content":[
                {"type":"text","text":"question"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,aW1hZ2U="}},
                {"type":"document","source":{"type":"text","data":"notes"},"title":"notes.txt"}
            ]}))
            .unwrap()
        };
        for messages in [vec![rich()], vec![rich(), assistant(), rich()]] {
            send_session(&mut state, &"ef".repeat(32), messages).await;
        }
        let rich = completion_bodies(6);
        assert_eq!(rich[0]["prompt"], rich[1]["prompt"]);
        assert_eq!(rich[0]["attachments"], rich[1]["attachments"]);
        assert_eq!(rich[1]["files"], json!(["uploaded-file"]));
        assert_eq!(rich[1]["attachments"][0]["file_name"], "notes.txt");
        let _initial_request_count = {
            let completions = requests.completions();
            assert_eq!(completions.len(), 8);
            assert!(
                completions[..4]
                    .iter()
                    .all(|request| request.path.contains(&conversation_uuid))
            );
            assert_eq!(
                completions[1].body["parent_message_uuid"],
                first_assistant_uuid
            );
            assert!(
                completions[2..4]
                    .iter()
                    .all(|request| request.body["parent_message_uuid"] == first_assistant_uuid)
            );
            assert_eq!(
                cached.explicit.unwrap().state,
                ExplicitSessionState::Committed
            );
            requests.recorded().len()
        };

        let switched_messages = vec![
            user("u1"),
            assistant(),
            user("changed"),
            assistant(),
            user("next"),
        ];
        let first_switch = state
            .try_chat(session_request_with_model(
                &digest,
                switched_messages.clone(),
                "claude-opus-4-6",
            ))
            .await;
        assert!(
            first_switch.is_ok(),
            "first model switch failed: {first_switch:?}"
        );
        let switched_back_messages = vec![
            user("u1"),
            assistant(),
            user("changed"),
            assistant(),
            user("next"),
            assistant(),
            user("back"),
        ];
        let second_switch = state
            .try_chat(session_request_with_model(
                &digest,
                switched_back_messages,
                "claude-sonnet-4-6",
            ))
            .await;
        assert!(
            second_switch.is_ok(),
            "second model switch failed: {second_switch:?}"
        );
        let switched = cache.get_explicit(&key).await.unwrap();
        assert_eq!(switched.conv_uuid, conversation_uuid);
        assert_eq!(switched.model, "claude-sonnet-4-6");
        assert_eq!(switched.turns[2].model.as_deref(), Some("claude-opus-4-6"));
        assert_eq!(
            switched.turns[3].model.as_deref(),
            Some("claude-sonnet-4-6")
        );
        let request_count = requests.recorded().len();

        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"block").unwrap();
        let cache = ConversationCache::persistent(blocker.join("cache.json")).await;
        let mut state = ClaudeWebState::new(handle.clone(), cache);
        state.principal = Some(crate::protocol::AuthPrincipal::for_authenticated_user());
        let error = state
            .try_chat(session_request(
                &"cd".repeat(32),
                vec![Message::new_text(Role::User, "hello")],
            ))
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol storage error");
        };
        assert_eq!(source.code, "session_storage_unavailable");
        let recorded = requests.recorded();
        assert!(
            recorded[request_count..]
                .iter()
                .any(|request| request.path == "/api/bootstrap")
        );
        assert!(!recorded[request_count..].iter().any(|request| {
            request.path.contains("/upload-file") || request.path.contains("/completion")
        }));

        let staged_dir = tempfile::tempdir().unwrap();
        let files = crate::protocol_files::StagedFileStore::persistent(staged_dir.path())
            .await
            .unwrap();
        let staged = stage_file(&files, &principal, "asset.png", b"image").await;
        let blocks = [
            json!({"type":"image","source":{"type":"file","file_id":staged.id}}),
            json!({"type":"document","source":{"type":"file","file_id":staged.id}}),
            json!({"type":"container_upload","file_id":staged.id}),
        ];
        let no_fs_message = serde_json::from_value(json!({
            "role":"user", "content":[blocks[0].clone()]
        }))
        .unwrap();
        let mut no_fs = ClaudeWebState::new(handle.clone(), ConversationCache::new());
        no_fs.principal = Some(principal.clone());
        let ClewdrError::Protocol { source } = no_fs
            .try_chat(session_request(&"a1".repeat(32), vec![no_fs_message]))
            .await
            .unwrap_err()
        else {
            panic!("expected staged files error");
        };
        assert_eq!(source.code, "staged_files_unavailable");

        let upload_start = requests.uploads.load(Ordering::SeqCst);
        for (index, block) in blocks.into_iter().enumerate() {
            let message: Message = serde_json::from_value(json!({
                "role":"user", "content":[block.clone()]
            }))
            .unwrap();
            let mut staged_state = ClaudeWebState::new(handle.clone(), ConversationCache::new());
            staged_state.principal = Some(principal.clone());
            staged_state.staged_files = Some(files.clone());
            let digest = format!("{:064x}", index + 2);
            send_session(&mut staged_state, &digest, vec![message.clone()]).await;
            if index == 2 {
                send_session(
                    &mut staged_state,
                    &digest,
                    vec![
                        message,
                        assistant(),
                        serde_json::from_value(json!({
                            "role":"user", "content":[block]
                        }))
                        .unwrap(),
                    ],
                )
                .await;
            }
        }
        assert_eq!(requests.uploads.load(Ordering::SeqCst) - upload_start, 3);

        let blocked = tempfile::tempdir().unwrap();
        let parent = blocked.path().join("blocked");
        std::fs::write(&parent, b"block").unwrap();
        let cache = ConversationCache::persistent(parent.join("cache.json")).await;
        let key = ExplicitSessionKey::new(principal.as_str(), "mapping-failure");
        explicit_test_seed(
            &cache,
            key.clone(),
            explicit_conversation(ExplicitSessionState::InFlight),
        )
        .await;
        let limited_dir = tempfile::tempdir().unwrap();
        let limited = crate::protocol_files::StagedFileStore::persistent_with_limits(
            limited_dir.path(),
            1,
            1,
        )
        .await
        .unwrap();
        let first = stage_file(&limited, &principal, "first", b"a").await;
        let mut failed = ClaudeWebState::new(handle.clone(), cache);
        failed.principal = Some(principal.clone());
        failed.staged_files = Some(limited.clone());
        failed.explicit_file_key = Some(key);
        assert!(
            failed
                .upload_files(
                    vec![ImageSource::File { file_id: first.id }],
                    "org",
                    "conversation",
                )
                .await
                .is_err()
        );
        stage_file(&limited, &principal, "second", b"b").await;

        let partial = MockClaudeServer {
            fail_second_upload: true,
            ..Default::default()
        };
        let endpoint = partial.start().await;
        CLEWDR_CONFIG.rcu(|config| {
            let mut config = ClewdrConfig::clone(config);
            config.rproxy = Some(endpoint.clone());
            config
        });
        let cache = ConversationCache::new();
        let digest = "fd".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let mut state = ClaudeWebState::new(handle.clone(), cache.clone());
        state.principal = Some(principal.clone());
        let image = || ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
                file_name: None,
            },
            cache_control: None,
        };
        state
            .try_chat(session_request(
                &digest,
                vec![Message::new_blocks(Role::User, vec![image(), image()])],
            ))
            .await
            .unwrap_err();
        assert_eq!(
            explicit_test_state(&cache, &key).await,
            ExplicitSessionState::Uncertain
        );
        assert!(cache.reset_explicit(&key).await.unwrap());
        cache
            .set_explicit_checked(
                key.clone(),
                explicit_conversation(ExplicitSessionState::InFlight),
            )
            .await
            .unwrap();
        let state = ClaudeWebState::new(handle.clone(), cache.clone());
        state
            .finish_explicit_send(
                &key,
                true,
                Err(ClewdrError::BadRequest {
                    msg: "upload failed",
                }),
            )
            .await
            .unwrap_err();
        assert_eq!(
            cache
                .get_explicit(&key)
                .await
                .unwrap()
                .explicit
                .unwrap()
                .state,
            ExplicitSessionState::Committed
        );

        let incomplete = MockClaudeServer {
            incomplete_completion: true,
            ..Default::default()
        };
        let endpoint = incomplete.start().await;
        CLEWDR_CONFIG.rcu(|config| {
            let mut config = ClewdrConfig::clone(config);
            config.rproxy = Some(endpoint.clone());
            config
        });
        let mut state = ClaudeWebState::new(handle, ConversationCache::new());
        state.stream = true;
        let stream_request = |messages| CreateMessageParams {
            model: "model".into(),
            messages,
            stream: Some(true),
            ..Default::default()
        };
        let first = user("first");
        let second = user("second");
        let third = user("third");
        let response = state
            .try_chat(stream_request(vec![first.clone()]))
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        for messages in [
            vec![first.clone(), second.clone()],
            vec![first, second, third],
        ] {
            drop(state.try_chat(stream_request(messages)).await.unwrap());
        }
        let paths = incomplete
            .completions()
            .into_iter()
            .map(|request| request.path)
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 3);
        assert!(paths.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[tokio::test]
    async fn missing_bound_cookie_rebuilds_with_an_available_cookie() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let mock = MockClaudeServer::default();
        let (_restore, handle, cache, principal) = configured_explicit_state(&mock).await;
        let digest = "ef".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let first = Message::new_text(Role::User, "u1");
        let user_digest = digest_messages(std::slice::from_ref(&first)).unwrap().users[0]
            .1
            .clone();
        let mut conversation = explicit_conversation(ExplicitSessionState::Committed);
        conversation.cookie_id = "missing-cookie-id".into();
        conversation.model = "claude-sonnet-4-6".into();
        let explicit = conversation.explicit.as_mut().unwrap();
        explicit.model_digest = digest_model(&conversation.model);
        explicit
            .turns
            .push(crate::claude_web_state::explicit_session::ExplicitTurn {
                model: None,
                parent_uuid_before: None,
                user_digests: vec![user_digest.clone()],
                assistant_uuid_after: "assistant".into(),
                parent_timeline: Vec::new(),
                request_timeline: vec![format!("user:{user_digest}")],
                assistant_digest_after: Some(
                    crate::claude_web_state::explicit_session::digest_assistant_output("answer"),
                ),
            });
        cache
            .set_explicit_checked(key.clone(), conversation)
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal);
        let response = state
            .try_explicit_chat(params(vec![first]), digest)
            .await
            .unwrap();
        axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            explicit_test_state(&cache, &key).await,
            ExplicitSessionState::Committed
        );
        assert_eq!(mock.completions().len(), 1);
    }

    #[test]
    fn cookie_and_organization_binding_mismatches_expire_session() {
        let mut cached = explicit_conversation(ExplicitSessionState::Committed);
        assert!(explicit_binding_error(&cached, "other", "org").is_some());
        cached.cookie_id = "cookie".into();
        assert!(explicit_binding_error(&cached, "cookie", "other").is_some());
        assert!(explicit_binding_error(&cached, "cookie", "org").is_none());
    }

    #[tokio::test]
    async fn upstream_not_found_or_gone_tombstones_explicit_session() {
        for status in [StatusCode::NOT_FOUND, StatusCode::GONE] {
            let handle = crate::services::cookie_actor::CookieActorHandle::start()
                .await
                .unwrap();
            let cache = ConversationCache::new();
            let key = ExplicitSessionKey::new("principal", status.as_u16().to_string());
            cache
                .set_explicit_checked(
                    key.clone(),
                    explicit_conversation(ExplicitSessionState::InFlight),
                )
                .await
                .unwrap();
            let state = ClaudeWebState::new(handle, cache.clone());
            let error = state
                .finish_explicit_send(
                    &key,
                    true,
                    Err(ClewdrError::ClaudeHttpError {
                        code: status,
                        inner: crate::error::ClaudeErrorBody {
                            message: json!("missing"),
                            r#type: "not_found".into(),
                            code: Some(status.as_u16()),
                        },
                    }),
                )
                .await
                .unwrap_err();
            let ClewdrError::Protocol { source } = error else {
                panic!("expected protocol error");
            };
            assert_eq!(source.status, StatusCode::GONE);
            assert_eq!(
                explicit_test_state(&cache, &key).await,
                ExplicitSessionState::Tombstoned
            );
        }
    }
    #[test]
    fn model_selector_state_body_uses_effort_and_mode_shape() {
        for (effort, thinking, expected_effort, expected_mode) in [
            (OutputEffort::Max, Thinking::adaptive(), "max", "auto"),
            (OutputEffort::Xhigh, Thinking::Disabled, "xhigh", "off"),
        ] {
            let params = CreateMessageParams {
                model: "claude-opus-4-8".into(),
                messages: vec![Message::new_text(Role::User, "hi")],
                output_config: Some(OutputConfig {
                    effort: Some(effort),
                    format: None,
                }),
                thinking: Some(thinking),
                ..Default::default()
            };
            assert_eq!(
                model_selector_state_body(&params)["thinking"],
                json!({"type":"effort_and_mode", "effort":expected_effort, "mode":expected_mode})
            );
        }
    }

    #[test]
    fn create_conversation_params_match_claude_web_first_completion() {
        let params = CreateMessageParams {
            model: "claude-opus-4-8".to_string(),
            messages: vec![Message::new_text(Role::User, "hi")],
            thinking: Some(Thinking::adaptive()),
            ..Default::default()
        };

        assert_eq!(
            serde_json::to_value(create_conversation_params(&params, true, true)).unwrap(),
            json!({
                "name": "",
                "model": "claude-opus-4-8",
                "include_conversation_preferences": true,
                "paprika_mode": "auto",
                "compass_mode": null,
                "tool_search_mode": "auto",
                "is_temporary": true,
                "enabled_imagine": true
            })
        );
    }
}
