use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use colored::Colorize;
use futures::TryFutureExt;
use serde_json::json;
use snafu::ResultExt;
use tracing::{Instrument, debug, error, info, info_span, warn};
use wreq::{Method, Response, header::ACCEPT};

use super::{
    ClaudeWebState, PendingCacheWrite,
    transform::{
        extract_base64_file, extract_document_file_name, extract_document_text, extract_file_id,
    },
};
use crate::{
    claude_web_state::conversation_cache::{CachedConversation, CachedTurn, ExplicitSessionKey},
    claude_web_state::diff::{self, DiffResult, extract_user_hashes, hash_system},
    claude_web_state::explicit_session::{
        ExplicitConversation, ExplicitLifecycle, ExplicitReusePlan, ExplicitSessionState,
        PendingExplicitTurn, digest_messages, digest_model, digest_system, parent_timeline, plan,
    },
    config::CLEWDR_CONFIG,
    error::{CheckClaudeErr, ClewdrError, WreqSnafu},
    protocol::{ProtocolError, parse_session_id},
    types::claude::{ContentBlock, CreateMessageParams, ImageSource, Message, MessageContent},
    types::claude_web::request::{Attachment, CreateConversationParams, TurnMessageUuids},
    utils::{TIME_ZONE, print_out_json},
};

/// Bundled user messages ready to be sent
struct BundledMessages {
    /// Short content goes into prompt
    prompt: String,
    /// Text document content goes into Claude Web attachments
    attachments: Vec<Attachment>,
    /// Extracted images (if any)
    #[allow(dead_code)]
    images: Vec<ImageSource>,
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
    /// Attempts to send a chat message to Claude API with retry mechanism
    ///
    /// This method handles the complete chat flow including:
    /// - Request preparation and logging
    /// - Cookie management for authentication
    /// - Executing the chat request with automatic retries on failure
    /// - Response transformation according to the specified API format
    /// - Error handling and cleanup
    ///
    /// The method implements a sophisticated retry mechanism to handle transient failures,
    /// and manages conversation cleanup to prevent resource leaks. It also includes
    /// performance tracking to measure response times.
    ///
    /// # Arguments
    /// * `p` - The client request body containing messages and configuration
    ///
    /// # Returns
    /// * `Result<axum::response::Response, ClewdrError>` - Formatted response or error
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

            // Create shared stream health flag for monitoring SSE completion
            let can_reuse =
                CLEWDR_CONFIG.load().reuse_conversation && !CLEWDR_CONFIG.load().preserve_chats;
            if can_reuse {
                let flag = Arc::new(AtomicBool::new(false));
                state.stream_health_flag = Some(flag.clone());
                self.stream_health_flag = Some(flag);
            }

            let cookie = state.request_cookie().await?;
            // check if request is successful
            let web_res = async {
                state.bootstrap().await?;
                state.send_chat(p).await
            };
            let transform_res = web_res
                .and_then(async |r| self.transform_response(r).await)
                .instrument(info_span!("claude_web", "cookie" = cookie.cookie.mask()));

            match transform_res.await {
                Ok(b) => {
                    // Commit pending cache write (optimistic)
                    if let Some(pending) = state.pending_cache_write.take() {
                        state.commit_cache_write(pending).await;
                    }

                    if let Err(e) = state.clean_chat().await {
                        warn!("Failed to clean chat: {}", e);
                    }
                    return Ok(b);
                }
                Err(e) => {
                    // Invalidate cache on error
                    state.conv_cache.invalidate(&state.cache_key()).await;
                    state.pending_cache_write = None;

                    // delete chat after an error
                    if let Err(e) = state.clean_chat().await {
                        warn!("Failed to clean chat: {}", e);
                    }
                    error!("{e}");
                    // 429 error
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

    async fn try_explicit_chat(
        &mut self,
        p: CreateMessageParams,
        session_digest: String,
    ) -> Result<axum::response::Response, ClewdrError> {
        let principal = self.principal.clone().ok_or(ClewdrError::InvalidAuth)?;
        let key = ExplicitSessionKey::new(principal.as_str(), &session_digest);
        let operation = self.conv_cache.try_lock_explicit_operation(&key).await?;
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
            self.conv_cache.tombstone_explicit(&key).await;
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
            self.conv_cache.tombstone_explicit(&key).await;
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

        let send_result = self
            .send_explicit_plan(
                &reuse,
                existing.as_ref(),
                parent_uuid.as_deref(),
                replace_from_turn,
                &indices,
                &suffix_hashes,
                &p,
            )
            .await;

        let Some(pending_write) = self.pending_cache_write.take() else {
            return match send_result {
                Err(error) => Err(error),
                Ok(_) => Err(ClewdrError::Whatever {
                    message: "Explicit session response has no pending cache write".to_string(),
                    source: None,
                }),
            };
        };
        self.stage_explicit_write(
            key.clone(),
            pending_write,
            model_digest,
            system_digest,
            parent_uuid,
            user_digests[suffix_start..].to_vec(),
            replace_from_turn,
            selected_parent_timeline,
            digested.timeline,
        )
        .await?;
        let response = self.finish_explicit_send(&key, send_result).await?;
        self.explicit_lifecycle = Some(ExplicitLifecycle::new(
            self.conv_cache.clone(),
            key,
            operation,
        ));
        self.transform_response(response).await
    }

    async fn finish_explicit_send(
        &self,
        key: &ExplicitSessionKey,
        result: Result<Response, ClewdrError>,
    ) -> Result<Response, ClewdrError> {
        match result {
            Ok(response) => Ok(response),
            Err(ClewdrError::ClaudeHttpError { code, .. })
                if code == http::StatusCode::NOT_FOUND || code == http::StatusCode::GONE =>
            {
                self.conv_cache.tombstone_explicit(key).await;
                Err(ProtocolError::new(
                    http::StatusCode::GONE,
                    "conversation_expired",
                    "The upstream conversation no longer exists",
                )
                .into())
            }
            Err(error) => {
                self.conv_cache.mark_explicit_uncertain(key).await;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn send_explicit_plan(
        &mut self,
        reuse: &ExplicitReusePlan,
        existing: Option<&CachedConversation>,
        parent_uuid: Option<&str>,
        replace_from_turn: usize,
        indices: &[usize],
        hashes: &[u64],
        p: &CreateMessageParams,
    ) -> Result<Response, ClewdrError> {
        match reuse {
            ExplicitReusePlan::Create => self.send_full(p.clone(), true).await,
            ExplicitReusePlan::Append { parent_uuid, .. } => {
                self.send_incremental(
                    existing.expect("append requires cache"),
                    parent_uuid,
                    indices,
                    hashes,
                    p,
                )
                .await
            }
            ExplicitReusePlan::Fork { .. } | ExplicitReusePlan::Regenerate { .. } => {
                self.send_incremental_fork(
                    existing.expect("reuse requires cache"),
                    parent_uuid,
                    replace_from_turn,
                    indices,
                    hashes,
                    p,
                )
                .await
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn stage_explicit_write(
        &self,
        key: ExplicitSessionKey,
        write: PendingCacheWrite,
        model_digest: String,
        system_digest: String,
        parent_uuid_before: Option<String>,
        user_digests: Vec<String>,
        replace_from_turn: usize,
        parent_timeline: Vec<String>,
        request_timeline: Vec<String>,
    ) -> Result<(), ProtocolError> {
        let assistant_uuid_after = match write {
            PendingCacheWrite::Init { mut conv, .. } => {
                let assistant_uuid = conv
                    .turns
                    .last()
                    .expect("initial cache write has a turn")
                    .assistant_uuid
                    .clone();
                conv.turns.clear();
                conv.explicit = Some(ExplicitConversation {
                    state: ExplicitSessionState::InFlight,
                    model_digest,
                    system_digest,
                    turns: Vec::new(),
                    pending: None,
                });
                self.conv_cache.set_explicit(key.clone(), *conv).await;
                assistant_uuid
            }
            PendingCacheWrite::AppendTurn { turn, .. }
            | PendingCacheWrite::ForkAndAppend { turn, .. } => turn.assistant_uuid,
        };
        self.conv_cache
            .stage_explicit_turn(
                &key,
                PendingExplicitTurn {
                    parent_uuid_before,
                    user_digests,
                    assistant_uuid_after,
                    replace_from_turn,
                    parent_timeline,
                    request_timeline,
                },
            )
            .await?;
        Ok(())
    }

    /// Main entry point — tries cache reuse, falls back to full paste
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
                        // invalidate on failure
                        self.conv_cache.invalidate(&self.cache_key_for(&p)).await;
                        // fall through to full path
                    }
                }
            }
            // cache miss or reuse not possible → full path
            // but this time we'll also write cache on success
        }

        self.send_full(p, can_reuse).await
    }

    /// Attempt to reuse a cached conversation
    /// Returns None if no cache or not reusable
    /// Returns Some(Ok(response)) on success
    /// Returns Some(Err(e)) if reuse was attempted but failed
    async fn try_reuse_conversation(
        &mut self,
        p: &CreateMessageParams,
    ) -> Option<Result<Response, ClewdrError>> {
        let key = self.cache_key_for(p);
        let cached = self.conv_cache.get(&key).await?;

        // Check stream health from previous request
        if !self.conv_cache.is_last_stream_healthy(&key).await {
            info!("[CACHE] last stream was unhealthy, invalidating");
            self.conv_cache.invalidate(&key).await;
            return None;
        }

        // Validate: cookie must match
        if cached.cookie_id != self.cookie_id() {
            info!("[CACHE] cookie mismatch, invalidating");
            self.conv_cache.invalidate(&key).await;
            return None;
        }
        // Validate: model must match
        if cached.model != p.model {
            info!("[CACHE] model changed: {} → {}", cached.model, p.model);
            self.conv_cache.invalidate(&key).await;
            return None;
        }
        // Validate: is_pro must match
        if cached.is_pro != self.is_pro() {
            info!("[CACHE] pro status changed");
            self.conv_cache.invalidate(&key).await;
            return None;
        }

        // Extract user message hashes from new request
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

    /// Full paste path (existing logic + cache write on success)
    async fn send_full(
        &mut self,
        p: CreateMessageParams,
        write_cache: bool,
    ) -> Result<Response, ClewdrError> {
        let org_uuid = self
            .org_uuid
            .to_owned()
            .ok_or(ClewdrError::UnexpectedNone {
                msg: "Organization UUID is not set",
            })?;

        self.sync_model_selector_state(&p).await?;

        // Claude Web generates the UUID client-side and creates the conversation
        // as part of the first completion request.
        let new_uuid = uuid::Uuid::new_v4().to_string();
        let is_temporary = !CLEWDR_CONFIG.load().preserve_chats;
        self.conv_uuid = Some(new_uuid.clone());
        self.last_params = Some(p.clone());
        debug!("Generated conversation UUID: {}", new_uuid);

        // === Transform and send ===
        let mut body = self
            .transform_request(p.clone())
            .ok_or(ClewdrError::BadRequest {
                msg: "Request body is empty",
            })?;
        body.create_conversation_params =
            Some(create_conversation_params(&p, is_temporary, self.is_pro()));

        // Generate turn_message_uuids
        let human_uuid = uuid::Uuid::new_v4().to_string();
        let assistant_uuid = uuid::Uuid::new_v4().to_string();
        body.turn_message_uuids = Some(TurnMessageUuids {
            human_message_uuid: human_uuid.clone(),
            assistant_message_uuid: assistant_uuid.clone(),
        });

        let images = body.images.drain(..).collect::<Vec<_>>();

        let files = self.upload_files(images, &org_uuid, &new_uuid).await?;
        body.files = files;

        if write_cache {
            let user_hashes = extract_user_hashes(&p.messages)
                .iter()
                .map(|(_, h)| *h)
                .collect();
            let stream_flag = self
                .stream_health_flag
                .clone()
                .unwrap_or_else(|| Arc::new(AtomicBool::new(true)));
            self.pending_cache_write = Some(PendingCacheWrite::Init {
                key: self.cache_key_for(&p),
                conv: Box::new(CachedConversation {
                    conv_uuid: new_uuid.clone(),
                    org_uuid: org_uuid.clone(),
                    cookie_id: self.cookie_id(),
                    model: p.model.clone(),
                    is_pro: self.is_pro(),
                    system_hash: hash_system(&p.system),
                    turns: vec![CachedTurn {
                        user_hashes,
                        assistant_uuid: assistant_uuid.clone(),
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                    last_stream_healthy: stream_flag,
                    explicit: None,
                }),
            });
        }

        // send the request
        print_out_json(&body, "claude_web_clewdr_req.json");
        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{org_uuid}/chat_conversations/{new_uuid}/completion"
            ))
            .expect("Url parse error");

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

    /// Incremental send — append new messages to existing conversation
    async fn send_incremental(
        &mut self,
        cached: &CachedConversation,
        parent_uuid: &str,
        new_user_indices: &[usize],
        new_user_hashes: &[u64],
        p: &CreateMessageParams,
    ) -> Result<Response, ClewdrError> {
        self.conv_uuid = Some(cached.conv_uuid.clone());
        self.last_params = Some(p.clone());

        self.sync_model_selector_state(p).await?;

        // Update paprika_mode if needed
        let need_thinking = p
            .web_thinking_mode()
            .is_some_and(|mode| mode == crate::types::claude::ThinkingMode::Auto)
            && self.is_pro();
        self.update_paprika(&cached.conv_uuid, need_thinking).await;

        // Extract new user messages from original messages array
        let new_user_msgs: Vec<&Message> = new_user_indices
            .iter()
            .map(|&idx| &p.messages[idx])
            .collect();

        // Bundle user messages into prompt + optional attachment
        let bundled = self.bundle_user_messages(&new_user_msgs);

        // Generate turn UUIDs
        let human_uuid = uuid::Uuid::new_v4().to_string();
        let assistant_uuid = uuid::Uuid::new_v4().to_string();

        let body = self
            .build_incremental_body(
                &bundled,
                &cached.org_uuid,
                &cached.conv_uuid,
                Some(parent_uuid),
                &human_uuid,
                &assistant_uuid,
                p,
            )
            .await?;

        self.pending_cache_write = Some(PendingCacheWrite::AppendTurn {
            key: self.cache_key_for(p),
            turn: CachedTurn {
                user_hashes: new_user_hashes.to_vec(),
                assistant_uuid: assistant_uuid.clone(),
            },
        });

        print_out_json(&body, "claude_web_incremental_req.json");

        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}/completion",
                cached.org_uuid, cached.conv_uuid
            ))
            .expect("Url parse error");

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

    /// Incremental send with fork — edit scenario
    async fn send_incremental_fork(
        &mut self,
        cached: &CachedConversation,
        parent_uuid: Option<&str>,
        fork_turn_index: usize,
        remaining_user_indices: &[usize],
        remaining_user_hashes: &[u64],
        p: &CreateMessageParams,
    ) -> Result<Response, ClewdrError> {
        self.conv_uuid = Some(cached.conv_uuid.clone());
        self.last_params = Some(p.clone());

        self.sync_model_selector_state(p).await?;

        // Update paprika_mode if needed
        let need_thinking = p
            .web_thinking_mode()
            .is_some_and(|mode| mode == crate::types::claude::ThinkingMode::Auto)
            && self.is_pro();
        self.update_paprika(&cached.conv_uuid, need_thinking).await;

        // Extract remaining user messages
        let remaining_user_msgs: Vec<&Message> = remaining_user_indices
            .iter()
            .map(|&idx| &p.messages[idx])
            .collect();

        // Bundle all remaining user messages
        let bundled = self.bundle_user_messages(&remaining_user_msgs);

        let human_uuid = uuid::Uuid::new_v4().to_string();
        let assistant_uuid = uuid::Uuid::new_v4().to_string();

        let body = self
            .build_incremental_body(
                &bundled,
                &cached.org_uuid,
                &cached.conv_uuid,
                parent_uuid,
                &human_uuid,
                &assistant_uuid,
                p,
            )
            .await?;

        self.pending_cache_write = Some(PendingCacheWrite::ForkAndAppend {
            key: self.cache_key_for(p),
            fork_turn_index,
            turn: CachedTurn {
                user_hashes: remaining_user_hashes.to_vec(),
                assistant_uuid: assistant_uuid.clone(),
            },
        });

        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{}/chat_conversations/{}/completion",
                cached.org_uuid, cached.conv_uuid
            ))
            .expect("Url parse error");

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

    /// PUT paprika_mode setting on existing conversation
    async fn update_paprika(&self, conv_uuid: &str, need_thinking: bool) {
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

    /// Build the completion request body for incremental sends
    async fn build_incremental_body(
        &self,
        bundled: &BundledMessages,
        org_uuid: &str,
        conversation_uuid: &str,
        parent_uuid: Option<&str>,
        human_uuid: &str,
        assistant_uuid: &str,
        p: &CreateMessageParams,
    ) -> Result<serde_json::Value, ClewdrError> {
        let files = self
            .upload_files(bundled.images.clone(), org_uuid, conversation_uuid)
            .await?;
        let mut body = json!({
            "prompt": bundled.prompt,
            "timezone": TIME_ZONE.to_string(),
            "turn_message_uuids": {
                "human_message_uuid": human_uuid,
                "assistant_message_uuid": assistant_uuid,
            },
            "attachments": bundled.attachments,
            "files": files,
            "rendering_mode": if p.stream.unwrap_or_default() { "messages" } else { "raw" },
        });
        if let Some(parent_uuid) = parent_uuid {
            body["parent_message_uuid"] = json!(parent_uuid);
        }
        // Model (only for pro)
        if self.is_pro() {
            body["model"] = json!(p.model);
        }
        if let Some(effort) = p.web_thinking_effort() {
            body["effort"] = json!(effort);
        }
        if let Some(mode) = p.web_thinking_mode() {
            body["thinking_mode"] = json!(mode);
        }
        // Tools (same as full request)
        let mut tools = vec![];
        if CLEWDR_CONFIG.load().web_search {
            tools.push(json!({"type": "web_search_v0", "name": "web_search"}));
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        Ok(body)
    }

    /// Merge user messages into prompt or attachment based on length
    fn bundle_user_messages(&self, user_msgs: &[&Message]) -> BundledMessages {
        let mut texts: Vec<String> = vec![];
        let mut attachments: Vec<Attachment> = vec![];
        let mut images: Vec<ImageSource> = vec![];

        for msg in user_msgs {
            match &msg.content {
                MessageContent::Text { content } => {
                    texts.push(content.trim().to_string());
                }
                MessageContent::Blocks { content } => {
                    for block in content {
                        match block {
                            ContentBlock::Text { text, .. } => {
                                texts.push(text.trim().to_string());
                            }
                            ContentBlock::Image { source, .. } => {
                                images.push(source.clone());
                            }
                            ContentBlock::Document { source, title, .. } => {
                                let file_name =
                                    extract_document_file_name(source, title.as_deref());
                                if let Some(text) = extract_document_text(source) {
                                    attachments.push(match file_name {
                                        Some(file_name) => {
                                            Attachment::new_with_file_name(text, file_name)
                                        }
                                        None => Attachment::new(text),
                                    });
                                } else if let Some(file_id) = extract_file_id(source) {
                                    images.push(ImageSource::File { file_id });
                                } else if let Some((media_type, data)) = extract_base64_file(source)
                                {
                                    images.push(ImageSource::Base64 {
                                        media_type,
                                        data,
                                        file_name,
                                    });
                                }
                            }
                            ContentBlock::ContainerUpload { file_id, .. } => {
                                images.push(ImageSource::File {
                                    file_id: file_id.clone(),
                                });
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let combined = texts.join("\n\n");

        // Threshold: if combined text is under ~4000 chars, use prompt directly
        // Otherwise put it in an attachment
        const PROMPT_THRESHOLD: usize = 4000;

        if combined.len() <= PROMPT_THRESHOLD {
            let mut prompt = combined;
            if prompt.is_empty() && (!attachments.is_empty() || !images.is_empty()) {
                prompt = "Please answer using the attached content.".to_string();
            }
            BundledMessages {
                prompt,
                attachments,
                images,
            }
        } else {
            attachments.push(Attachment::new(combined));
            let mut p_str = CLEWDR_CONFIG.load().custom_prompt.clone();
            if p_str.is_empty() {
                p_str = "Please answer using the attached content.".to_string();
            }
            BundledMessages {
                prompt: p_str,
                attachments,
                images,
            }
        }
    }

    /// Execute a pending cache write
    async fn commit_cache_write(&self, pending: PendingCacheWrite) {
        match pending {
            PendingCacheWrite::Init { key, conv } => {
                info!("[CACHE] initialized for conv {}", conv.conv_uuid);
                self.conv_cache.set(key, *conv).await;
            }
            PendingCacheWrite::AppendTurn { key, turn } => {
                info!("[CACHE] appended turn (assistant={})", turn.assistant_uuid);
                self.conv_cache.append_turn(&key, turn).await;
                // Update stream health flag for the new request
                if let Some(flag) = self.stream_health_flag.as_ref() {
                    self.conv_cache
                        .update_stream_health(&key, flag.clone())
                        .await;
                }
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
                // Update stream health flag for the new request
                if let Some(flag) = self.stream_health_flag.as_ref() {
                    self.conv_cache
                        .update_stream_health(&key, flag.clone())
                        .await;
                }
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
    use std::sync::{Arc, Mutex, atomic::AtomicBool};

    use axum::{
        Router,
        body::{Body, to_bytes},
        extract::{Request, State},
        http::{StatusCode, header::CONTENT_TYPE},
        response::Response as AxumResponse,
    };
    use serde_json::json;

    use crate::claude_web_state::conversation_cache::ConversationCache;

    use super::*;
    use crate::types::claude::{OutputConfig, OutputEffort, Role, Thinking, ThinkingMode};

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        path: String,
        body: serde_json::Value,
    }

    async fn record_request(
        State(requests): State<Arc<Mutex<Vec<RecordedRequest>>>>,
        request: Request,
    ) -> AxumResponse {
        let path = request.uri().path().to_owned();
        let bytes = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        requests
            .lock()
            .unwrap()
            .push(RecordedRequest { path, body });
        AxumResponse::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from("data: {\"type\":\"message_stop\"}\n\n"))
            .unwrap()
    }

    async fn mock_endpoint() -> (url::Url, Arc<Mutex<Vec<RecordedRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .fallback(record_request)
            .with_state(requests.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            url::Url::parse(&format!("http://{address}/")).unwrap(),
            requests,
        )
    }

    fn params(messages: Vec<Message>) -> CreateMessageParams {
        CreateMessageParams {
            model: "claude-sonnet-4-6".into(),
            messages,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn explicit_plans_reuse_existing_send_transports_and_parent_uuids() {
        let (endpoint, requests) = mock_endpoint().await;
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, ConversationCache::new());
        state.endpoint = endpoint;
        state.org_uuid = Some("org".into());

        let create_params = params(vec![Message::new_text(Role::User, "u1")]);
        state
            .send_explicit_plan(
                &ExplicitReusePlan::Create,
                None,
                None,
                0,
                &[0],
                &[1],
                &create_params,
            )
            .await
            .unwrap();
        let PendingCacheWrite::Init {
            conv: mut cached, ..
        } = state.pending_cache_write.take().unwrap()
        else {
            panic!("create must use send_full");
        };
        let conversation_uuid = cached.conv_uuid.clone();
        let first_assistant = cached.turns[0].assistant_uuid.clone();

        let append_params = params(vec![
            Message::new_text(Role::User, "u1"),
            Message::new_text(Role::Assistant, "a1"),
            Message::new_text(Role::User, "u2"),
        ]);
        state
            .send_explicit_plan(
                &ExplicitReusePlan::Append {
                    parent_uuid: first_assistant.clone(),
                    suffix_start: 1,
                },
                Some(&cached),
                Some(&first_assistant),
                1,
                &[2],
                &[2],
                &append_params,
            )
            .await
            .unwrap();
        let PendingCacheWrite::AppendTurn { turn, .. } = state.pending_cache_write.take().unwrap()
        else {
            panic!("append must use send_incremental");
        };
        let second_assistant = turn.assistant_uuid.clone();
        cached.turns.push(turn);

        let fork_params = params(vec![
            Message::new_text(Role::User, "u1"),
            Message::new_text(Role::Assistant, "a1"),
            Message::new_text(Role::User, "forked"),
        ]);
        state
            .send_explicit_plan(
                &ExplicitReusePlan::Fork {
                    parent_uuid: Some(first_assistant.clone()),
                    suffix_start: 1,
                    replace_from_turn: 1,
                },
                Some(&cached),
                Some(&first_assistant),
                1,
                &[2],
                &[3],
                &fork_params,
            )
            .await
            .unwrap();
        assert!(matches!(
            state.pending_cache_write.take(),
            Some(PendingCacheWrite::ForkAndAppend {
                fork_turn_index: 1,
                ..
            })
        ));

        state
            .send_explicit_plan(
                &ExplicitReusePlan::Regenerate {
                    parent_uuid: Some(first_assistant.clone()),
                    suffix_start: 1,
                    replace_from_turn: 1,
                },
                Some(&cached),
                Some(&first_assistant),
                1,
                &[2],
                &[4],
                &append_params,
            )
            .await
            .unwrap();

        let requests = requests.lock().unwrap();
        let completions = requests
            .iter()
            .filter(|request| request.path.contains("/completion"))
            .collect::<Vec<_>>();
        assert_eq!(completions.len(), 4);
        assert!(
            completions
                .iter()
                .all(|request| request.path.contains(&conversation_uuid))
        );
        assert_eq!(completions[1].body["parent_message_uuid"], first_assistant);
        assert_eq!(completions[2].body["parent_message_uuid"], first_assistant);
        assert_eq!(completions[3].body["parent_message_uuid"], first_assistant);
        assert_ne!(second_assistant, first_assistant);
    }

    #[tokio::test]
    async fn no_fs_text_only_explicit_session_commits_in_memory() {
        let (endpoint, _) = mock_endpoint().await;
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cache = ConversationCache::new();
        let key = ExplicitSessionKey::new("principal", "aa".repeat(32));
        let operation = cache.try_lock_explicit_operation(&key).await.unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.endpoint = endpoint;
        state.org_uuid = Some("org".into());
        let request = params(vec![Message::new_text(Role::User, "hello")]);
        let digested = digest_messages(&request.messages).unwrap();
        let upstream = state
            .send_explicit_plan(
                &ExplicitReusePlan::Create,
                None,
                None,
                0,
                &[0],
                &[1],
                &request,
            )
            .await
            .unwrap();
        let pending = state.pending_cache_write.take().unwrap();
        state
            .stage_explicit_write(
                key.clone(),
                pending,
                digest_model(&request.model),
                digest_system(&request.system),
                None,
                digested
                    .users
                    .into_iter()
                    .map(|(_, digest)| digest)
                    .collect(),
                0,
                Vec::new(),
                digested.timeline,
            )
            .await
            .unwrap();
        state.explicit_lifecycle = Some(ExplicitLifecycle::new(
            cache.clone(),
            key.clone(),
            operation,
        ));
        state.transform_response(upstream).await.unwrap();
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
    }

    #[tokio::test]
    async fn missing_bound_cookie_tombstones_session_before_upstream() {
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cache = ConversationCache::new();
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let digest = "ef".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let first = Message::new_text(Role::User, "u1");
        let first_digest = digest_messages(std::slice::from_ref(&first)).unwrap().users[0]
            .1
            .clone();
        cache
            .set_explicit(
                key.clone(),
                CachedConversation {
                    conv_uuid: "conversation".into(),
                    org_uuid: "org".into(),
                    cookie_id: "missing-cookie-id".into(),
                    model: "claude-sonnet-4-6".into(),
                    is_pro: false,
                    system_hash: 0,
                    turns: vec![CachedTurn {
                        user_hashes: vec![1],
                        assistant_uuid: "assistant".into(),
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                    last_stream_healthy: Arc::new(AtomicBool::new(true)),
                    explicit: Some(ExplicitConversation {
                        state: ExplicitSessionState::Committed,
                        model_digest: digest_model("claude-sonnet-4-6"),
                        system_digest: digest_system(&None),
                        turns: vec![crate::claude_web_state::explicit_session::ExplicitTurn {
                            parent_uuid_before: None,
                            user_digests: vec![first_digest.clone()],
                            assistant_uuid_after: "assistant".into(),
                            parent_timeline: Vec::new(),
                            request_timeline: vec![format!("user:{first_digest}")],
                            assistant_digest_after: Some(
                                crate::claude_web_state::explicit_session::digest_assistant_output(
                                    "generated",
                                ),
                            ),
                        }],
                        pending: None,
                    }),
                },
            )
            .await;
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.principal = Some(principal);
        let error = state
            .try_explicit_chat(
                params(vec![
                    first,
                    Message::new_text(Role::Assistant, "generated"),
                    Message::new_text(Role::User, "u2"),
                ]),
                digest,
            )
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol error");
        };
        assert_eq!(source.status, StatusCode::GONE);
        assert_eq!(source.code, "conversation_expired");
        assert_eq!(
            cache
                .get_explicit(&key)
                .await
                .unwrap()
                .explicit
                .unwrap()
                .state,
            ExplicitSessionState::Tombstoned
        );
    }

    #[test]
    fn cookie_and_organization_binding_mismatches_expire_session() {
        let mut cached = CachedConversation {
            conv_uuid: "conversation".into(),
            org_uuid: "org".into(),
            cookie_id: "cookie".into(),
            model: "model".into(),
            is_pro: false,
            system_hash: 0,
            turns: Vec::new(),
            created_at: chrono::Utc::now(),
            last_used: chrono::Utc::now(),
            valid: true,
            last_stream_healthy: Arc::new(AtomicBool::new(true)),
            explicit: None,
        };
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
                .set_explicit(
                    key.clone(),
                    CachedConversation {
                        conv_uuid: "conversation".into(),
                        org_uuid: "org".into(),
                        cookie_id: "cookie".into(),
                        model: "model".into(),
                        is_pro: false,
                        system_hash: 0,
                        turns: Vec::new(),
                        created_at: chrono::Utc::now(),
                        last_used: chrono::Utc::now(),
                        valid: true,
                        last_stream_healthy: Arc::new(AtomicBool::new(true)),
                        explicit: Some(ExplicitConversation {
                            state: ExplicitSessionState::InFlight,
                            model_digest: "model".into(),
                            system_digest: "system".into(),
                            turns: Vec::new(),
                            pending: None,
                        }),
                    },
                )
                .await;
            let state = ClaudeWebState::new(handle, cache.clone());
            let error = state
                .finish_explicit_send(
                    &key,
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
                cache
                    .get_explicit(&key)
                    .await
                    .unwrap()
                    .explicit
                    .unwrap()
                    .state,
                ExplicitSessionState::Tombstoned
            );
        }
    }

    #[test]
    fn model_selector_state_body_uses_effort_and_mode_shape() {
        let params = CreateMessageParams {
            model: "claude-opus-4-8".to_string(),
            messages: vec![Message::new_text(Role::User, "hi")],
            output_config: Some(OutputConfig {
                effort: Some(OutputEffort::Max),
                format: None,
            }),
            thinking: Some(Thinking::adaptive()),
            ..Default::default()
        };

        assert_eq!(
            model_selector_state_body(&params),
            json!({
                "model": "claude-opus-4-8",
                "thinking": {
                    "type": "effort_and_mode",
                    "effort": "max",
                    "mode": "auto"
                }
            })
        );
    }

    #[test]
    fn model_selector_state_body_keeps_off_mode() {
        let params = CreateMessageParams {
            model: "claude-opus-4-8".to_string(),
            messages: vec![Message::new_text(Role::User, "hi")],
            output_config: Some(OutputConfig {
                effort: Some(OutputEffort::Xhigh),
                format: None,
            }),
            thinking: Some(Thinking::Disabled),
            ..Default::default()
        };

        assert_eq!(
            model_selector_state_body(&params)["thinking"],
            json!({
                "type": "effort_and_mode",
                "effort": "xhigh",
                "mode": "off"
            })
        );
        assert_eq!(params.web_thinking_mode(), Some(ThinkingMode::Off));
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
