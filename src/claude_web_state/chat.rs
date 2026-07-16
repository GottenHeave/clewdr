use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use colored::Colorize;
use futures::TryFutureExt;
use serde_json::json;
use snafu::ResultExt;
use tracing::{Instrument, debug, error, info, info_span, warn};
use wreq::{Method, Response, header::ACCEPT};

use super::{ClaudeWebState, PendingCacheWrite};
use crate::{
    claude_web_state::conversation_cache::{CachedConversation, CachedTurn, ExplicitSessionKey},
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

#[derive(Clone)]
struct PreparedExplicitTurn {
    conversation_uuid: String,
    human_uuid: String,
    assistant_uuid: String,
    cache_write: PendingCacheWrite,
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
        self.explicit_file_key = Some(key.clone());
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

        let prepared = self.prepare_explicit_turn(
            &reuse,
            existing.as_ref(),
            replace_from_turn,
            &suffix_hashes,
            &p,
            &organization_uuid,
        );
        self.stage_explicit_write(
            key.clone(),
            prepared.cache_write.clone(),
            model_digest,
            system_digest,
            parent_uuid.clone(),
            user_digests[suffix_start..].to_vec(),
            replace_from_turn,
            selected_parent_timeline,
            digested.timeline,
        )
        .await?;

        let send_result = self
            .send_explicit_plan_prepared(
                &reuse,
                existing.as_ref(),
                parent_uuid.as_deref(),
                replace_from_turn,
                &indices,
                &suffix_hashes,
                &p,
                &prepared,
            )
            .await;
        self.pending_cache_write.take();
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
                self.conv_cache.tombstone_explicit(key).await?;
                Err(ProtocolError::new(
                    http::StatusCode::GONE,
                    "conversation_expired",
                    "The upstream conversation no longer exists",
                )
                .into())
            }
            Err(error) => {
                self.conv_cache.mark_explicit_uncertain(key).await?;
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn send_explicit_plan_prepared(
        &mut self,
        reuse: &ExplicitReusePlan,
        existing: Option<&CachedConversation>,
        parent_uuid: Option<&str>,
        replace_from_turn: usize,
        indices: &[usize],
        hashes: &[u64],
        p: &CreateMessageParams,
        prepared: &PreparedExplicitTurn,
    ) -> Result<Response, ClewdrError> {
        match reuse {
            ExplicitReusePlan::Create => self.send_full(p.clone(), true, Some(prepared)).await,
            ExplicitReusePlan::Append { parent_uuid, .. } => {
                self.send_incremental(
                    existing.expect("append requires cache"),
                    parent_uuid,
                    indices,
                    hashes,
                    p,
                    Some(prepared),
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
                    Some(prepared),
                )
                .await
            }
        }
    }

    #[cfg(test)]
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
        let organization_uuid = existing
            .map(|conversation| conversation.org_uuid.as_str())
            .or(self.org_uuid.as_deref())
            .expect("explicit send requires an organization");
        let prepared = self.prepare_explicit_turn(
            reuse,
            existing,
            replace_from_turn,
            hashes,
            p,
            organization_uuid,
        );
        self.send_explicit_plan_prepared(
            reuse,
            existing,
            parent_uuid,
            replace_from_turn,
            indices,
            hashes,
            p,
            &prepared,
        )
        .await
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
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                    last_stream_healthy: self
                        .stream_health_flag
                        .clone()
                        .unwrap_or_else(|| Arc::new(AtomicBool::new(true))),
                    explicit: None,
                }),
            },
            ExplicitReusePlan::Append { .. } => PendingCacheWrite::AppendTurn {
                key: self.cache_key_for(p),
                turn: CachedTurn {
                    user_hashes: user_hashes.to_vec(),
                    assistant_uuid: assistant_uuid.clone(),
                },
            },
            ExplicitReusePlan::Fork { .. } | ExplicitReusePlan::Regenerate { .. } => {
                PendingCacheWrite::ForkAndAppend {
                    key: self.cache_key_for(p),
                    fork_turn_index: replace_from_turn,
                    turn: CachedTurn {
                        user_hashes: user_hashes.to_vec(),
                        assistant_uuid: assistant_uuid.clone(),
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

    #[allow(clippy::too_many_arguments)]
    async fn stage_explicit_write(
        &mut self,
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
        let (assistant_uuid_after, initial) = match write {
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
            parent_uuid_before,
            user_digests,
            assistant_uuid_after,
            replace_from_turn,
            parent_timeline,
            request_timeline,
        };
        if let Some(mut conversation) = initial {
            conversation.turns.clear();
            conversation.explicit = Some(ExplicitConversation {
                state: ExplicitSessionState::InFlight,
                model_digest,
                system_digest,
                turns: Vec::new(),
                pending: Some(pending),
                file_mappings: Default::default(),
            });
            self.conv_cache
                .set_explicit_checked(key, *conversation)
                .await?;
        } else {
            self.conv_cache.stage_explicit_turn(&key, pending).await?;
        }
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

        self.send_full(p, can_reuse, None).await
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

    /// Full paste path (existing logic + cache write on success)
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

        // Claude Web generates the UUID client-side and creates the conversation
        // as part of the first completion request.
        let new_uuid = prepared.conversation_uuid.clone();
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
        let human_uuid = prepared.human_uuid.clone();
        let assistant_uuid = prepared.assistant_uuid.clone();
        body.turn_message_uuids = Some(TurnMessageUuids {
            human_message_uuid: human_uuid.clone(),
            assistant_message_uuid: assistant_uuid.clone(),
        });

        if write_cache {
            self.pending_cache_write = Some(prepared.cache_write.clone());
        }

        let images = body.images.drain(..).collect::<Vec<_>>();
        let files = self.upload_files(images, &org_uuid, &new_uuid).await?;
        body.files = files;

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
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        Ok(body)
    }

    /// Merge user messages into prompt or attachment based on length
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

        let combined = texts.join(message_separator);

        // Threshold: if combined text is under ~4000 chars, use prompt directly
        // Otherwise put it in an attachment
        const PROMPT_THRESHOLD: usize = 4000;

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
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

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
    use crate::types::claude::{
        ContentBlock, ImageSource, Metadata, OutputConfig, OutputEffort, Role, Thinking,
        ThinkingMode,
    };

    static CONFIG_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn serve(app: Router) -> url::Url {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url::Url::parse(&format!("http://{address}/")).unwrap()
    }

    async fn test_state(endpoint: url::Url, cache: ConversationCache) -> ClaudeWebState {
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let mut state = ClaudeWebState::new(handle, cache);
        state.endpoint = endpoint;
        state.org_uuid = Some("org".into());
        state
    }

    fn cached_conversation() -> CachedConversation {
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
            explicit: None,
        }
    }

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
        let is_upload = path.contains("/upload-file");
        let bytes = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        requests
            .lock()
            .unwrap()
            .push(RecordedRequest { path, body });
        if is_upload {
            return AxumResponse::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{\"file_uuid\":\"uploaded-file\"}"))
                .unwrap();
        }
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
        (serve(app).await, requests)
    }

    #[derive(Clone)]
    struct FailingUploadState {
        uploads: Arc<AtomicUsize>,
    }

    async fn fail_second_upload(
        State(state): State<FailingUploadState>,
        request: Request,
    ) -> AxumResponse {
        if request.uri().path().contains("/upload-file") {
            let upload = state.uploads.fetch_add(1, Ordering::SeqCst);
            if upload == 1 {
                return AxumResponse::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from("upload failed"))
                    .unwrap();
            }
            return AxumResponse::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{\"file_uuid\":\"uploaded-file\"}"))
                .unwrap();
        }
        AxumResponse::builder()
            .status(StatusCode::OK)
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from("data: {\"type\":\"message_stop\"}\n\n"))
            .unwrap()
    }

    async fn partial_upload_failure_endpoint() -> url::Url {
        let app = Router::new()
            .fallback(fail_second_upload)
            .with_state(FailingUploadState {
                uploads: Arc::new(AtomicUsize::new(0)),
            });
        serve(app).await
    }

    #[derive(Clone)]
    struct BlackBoxState {
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        completions: Arc<AtomicUsize>,
    }

    async fn black_box_upstream(
        State(state): State<BlackBoxState>,
        request: Request,
    ) -> AxumResponse {
        let path = request.uri().path().to_owned();
        let bytes = to_bytes(request.into_body(), usize::MAX).await.unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        state.requests.lock().unwrap().push(RecordedRequest {
            path: path.clone(),
            body,
        });
        let json_response = |value: serde_json::Value| {
            AxumResponse::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&value).unwrap()))
                .unwrap()
        };
        if path == "/api/bootstrap" {
            return json_response(json!({
                "account": {
                    "email_address": "test@example.com",
                    "memberships": [{"organization":{"capabilities":["chat"]}}]
                }
            }));
        }
        if path == "/api/organizations" {
            return json_response(json!([{
                "uuid":"org", "capabilities":["chat"], "active_flags":[]
            }]));
        }
        if path.contains("/completion") {
            let index = state.completions.fetch_add(1, Ordering::SeqCst);
            let answer = if index == 0 { "a1" } else { "a2" };
            return AxumResponse::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(format!(
                    "data: {{\"completion\":\"{answer}\"}}\n\ndata: {{\"type\":\"message_stop\"}}\n\n"
                )))
                .unwrap();
        }
        json_response(json!({}))
    }

    async fn black_box_endpoint() -> (url::Url, Arc<Mutex<Vec<RecordedRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .fallback(black_box_upstream)
            .with_state(BlackBoxState {
                requests: requests.clone(),
                completions: Arc::new(AtomicUsize::new(0)),
            });
        (serve(app).await, requests)
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

    async fn create_append_requests(
        create: Vec<Message>,
        append: Vec<Message>,
        create_indices: &[usize],
        append_indices: &[usize],
    ) -> Vec<RecordedRequest> {
        let (endpoint, requests) = mock_endpoint().await;
        let mut state = test_state(endpoint, ConversationCache::new()).await;
        let create_hashes = vec![1; create_indices.len()];
        state
            .send_explicit_plan(
                &ExplicitReusePlan::Create,
                None,
                None,
                0,
                create_indices,
                &create_hashes,
                &params(create),
            )
            .await
            .unwrap();
        let PendingCacheWrite::Init { conv: cached, .. } =
            state.pending_cache_write.take().unwrap()
        else {
            panic!("create must initialize the cache");
        };
        let parent = cached.turns[0].assistant_uuid.clone();
        let append_hashes = vec![2; append_indices.len()];
        state
            .send_explicit_plan(
                &ExplicitReusePlan::Append {
                    parent_uuid: parent.clone(),
                    suffix_start: 1,
                },
                Some(&cached),
                Some(&parent),
                1,
                append_indices,
                &append_hashes,
                &params(append),
            )
            .await
            .unwrap();
        requests.lock().unwrap().clone()
    }

    fn file_session() -> CachedConversation {
        let mut conversation = cached_conversation();
        conversation.explicit = Some(ExplicitConversation {
            state: ExplicitSessionState::InFlight,
            model_digest: "model".into(),
            system_digest: "system".into(),
            turns: Vec::new(),
            pending: None,
            file_mappings: Default::default(),
        });
        conversation
    }

    fn file_source(id: impl Into<String>) -> Vec<ImageSource> {
        vec![ImageSource::File { file_id: id.into() }]
    }

    async fn upload_test_file(
        state: &mut ClaudeWebState,
        source: Vec<ImageSource>,
    ) -> Result<Vec<String>, ClewdrError> {
        state.upload_files(source, "org", "conversation").await
    }

    #[tokio::test]
    async fn staged_upload_mapping_is_reused_only_within_its_session() {
        let (endpoint, requests) = mock_endpoint().await;
        let cache = ConversationCache::new();
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
        let first = ExplicitSessionKey::new(principal.as_str(), "first");
        let second = ExplicitSessionKey::new(principal.as_str(), "second");
        for key in [&first, &second] {
            cache
                .set_explicit_checked(key.clone(), file_session())
                .await
                .unwrap();
        }
        let temp = tempfile::tempdir().unwrap();
        let files = crate::protocol_files::StagedFileStore::persistent(temp.path())
            .await
            .unwrap();
        let staged = files
            .stage_stream(
                &principal,
                "report.txt",
                "text/plain",
                futures::stream::iter([Ok::<_, std::io::Error>(bytes::Bytes::from_static(
                    b"report",
                ))]),
            )
            .await
            .unwrap();
        let mut state = test_state(endpoint, cache).await;
        state.principal = Some(principal);
        state.staged_files = Some(files);
        let source = file_source(staged.id);
        state.explicit_file_key = Some(first);
        assert_eq!(
            upload_test_file(&mut state, source.clone()).await.unwrap(),
            vec!["uploaded-file"]
        );
        upload_test_file(&mut state, source.clone()).await.unwrap();
        state.explicit_file_key = Some(second);
        upload_test_file(&mut state, source).await.unwrap();
        assert_eq!(
            requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.path.contains("/upload-file"))
                .count(),
            2
        );
        state.staged_files = None;
        state.explicit_file_key = Some(ExplicitSessionKey::new("principal", "no-fs"));
        assert_eq!(
            upload_test_file(&mut state, file_source("upstream-file"))
                .await
                .unwrap(),
            vec!["upstream-file"]
        );
        let error = upload_test_file(&mut state, file_source("file_clewdr_v1_missing"))
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol error");
        };
        assert_eq!(source.code, "staged_files_unavailable");
    }

    async fn send_and_stage_create(
        state: &mut ClaudeWebState,
        key: &ExplicitSessionKey,
        request: &CreateMessageParams,
    ) -> Result<Response, ClewdrError> {
        let digested = digest_messages(&request.messages).unwrap();
        let send = state
            .send_explicit_plan(
                &ExplicitReusePlan::Create,
                None,
                None,
                0,
                &[0],
                &[1],
                request,
            )
            .await;
        let cache_write = state.pending_cache_write.take().unwrap();
        state
            .stage_explicit_write(
                key.clone(),
                cache_write,
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
        send
    }

    #[tokio::test]
    async fn fork_and_regenerate_reuse_incremental_transport_and_parent_uuid() {
        let (endpoint, requests) = mock_endpoint().await;
        let mut state = test_state(endpoint, ConversationCache::new()).await;
        let mut cached = cached_conversation();
        cached.turns.push(CachedTurn {
            user_hashes: vec![1],
            assistant_uuid: "parent".into(),
        });
        let request = params(vec![
            Message::new_text(Role::User, "u1"),
            Message::new_text(Role::Assistant, "a1"),
            Message::new_text(Role::User, "replacement"),
        ]);
        for plan in [
            ExplicitReusePlan::Fork {
                parent_uuid: Some("parent".into()),
                suffix_start: 1,
                replace_from_turn: 1,
            },
            ExplicitReusePlan::Regenerate {
                parent_uuid: Some("parent".into()),
                suffix_start: 1,
                replace_from_turn: 1,
            },
        ] {
            state
                .send_explicit_plan(
                    &plan,
                    Some(&cached),
                    Some("parent"),
                    1,
                    &[2],
                    &[2],
                    &request,
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
        }
        let requests = requests.lock().unwrap();
        let completions = requests
            .iter()
            .filter(|request| request.path.contains("/completion"))
            .collect::<Vec<_>>();
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().all(|request| {
            request.path.contains("/conversation/completion")
                && request.body["parent_message_uuid"] == "parent"
        }));
    }

    #[tokio::test]
    async fn try_chat_reuses_conversation_and_stops_before_unpersisted_side_effects() {
        let _config_guard = CONFIG_TEST_LOCK.lock().await;
        let (endpoint, requests) = black_box_endpoint().await;
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
        let session_id = format!("cherry_topic_v1_{}", "ab".repeat(32));
        let request = |messages| CreateMessageParams {
            model: "claude-sonnet-4-6".into(),
            messages,
            metadata: Some(Metadata {
                fields: [("user_id".into(), session_id.clone())]
                    .into_iter()
                    .collect(),
            }),
            ..Default::default()
        };

        let mut first = ClaudeWebState::new(handle.clone(), cache.clone());
        first.principal = Some(principal.clone());
        first
            .try_chat(request(vec![Message::new_text(Role::User, "u1")]))
            .await
            .unwrap();
        let key = ExplicitSessionKey::new(principal.as_str(), "ab".repeat(32));
        let cached = cache.get_explicit(&key).await.unwrap();
        let conversation_uuid = cached.conv_uuid.clone();
        let first_assistant_uuid = cached.turns[0].assistant_uuid.clone();

        let mut second = ClaudeWebState::new(handle.clone(), cache.clone());
        second.principal = Some(principal);
        second
            .try_chat(request(vec![
                Message::new_text(Role::User, "u1"),
                Message::new_text(Role::Assistant, "a1"),
                Message::new_text(Role::User, "u2"),
            ]))
            .await
            .unwrap();
        let request_count = {
            let requests = requests.lock().unwrap();
            let completions = requests
                .iter()
                .filter(|request| request.path.contains("/completion"))
                .collect::<Vec<_>>();
            assert_eq!(completions.len(), 2);
            assert!(
                completions
                    .iter()
                    .all(|request| request.path.contains(&conversation_uuid))
            );
            assert_eq!(
                completions[1].body["parent_message_uuid"],
                first_assistant_uuid
            );
            requests.len()
        };

        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"block").unwrap();
        let cache = ConversationCache::persistent(blocker.join("cache.json")).await;
        let mut state = ClaudeWebState::new(handle.clone(), cache);
        state.principal = Some(crate::protocol::AuthPrincipal::for_authenticated_user());
        let error = state
            .try_chat(CreateMessageParams {
                model: "claude-sonnet-4-6".into(),
                messages: vec![Message::new_text(Role::User, "hello")],
                metadata: Some(Metadata {
                    fields: [(
                        "user_id".into(),
                        format!("cherry_topic_v1_{}", "cd".repeat(32)),
                    )]
                    .into_iter()
                    .collect(),
                }),
                ..Default::default()
            })
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol storage error");
        };
        assert_eq!(source.code, "session_storage_unavailable");
        let requests = requests.lock().unwrap();
        assert!(
            requests[request_count..]
                .iter()
                .any(|request| request.path == "/api/bootstrap")
        );
        assert!(!requests[request_count..].iter().any(|request| {
            request.path.contains("/upload-file") || request.path.contains("/completion")
        }));
    }

    #[tokio::test]
    async fn consecutive_user_messages_match_create_and_incremental_prompts() {
        let requests = create_append_requests(
            vec![
                Message::new_text(Role::User, "first"),
                Message::new_text(Role::User, "second"),
            ],
            vec![
                Message::new_text(Role::User, "history"),
                Message::new_text(Role::Assistant, "answer"),
                Message::new_text(Role::User, "first"),
                Message::new_text(Role::User, "second"),
            ],
            &[0, 1],
            &[2, 3],
        )
        .await;
        let completions = requests
            .iter()
            .filter(|request| request.path.contains("/completion"))
            .collect::<Vec<_>>();
        assert_eq!(completions[0].body["prompt"], "first\nsecond");
        assert_eq!(completions[0].body["prompt"], completions[1].body["prompt"]);
    }

    #[tokio::test]
    async fn create_and_incremental_forward_equivalent_rich_content() {
        let rich_message = || {
            serde_json::from_value::<Message>(json!({
                "role":"user",
                "content":[
                    {"type":"text", "text":"question"},
                    {"type":"image_url", "image_url":{"url":"data:image/png;base64,aW1hZ2U="}},
                    {"type":"document", "source":{"type":"text", "data":"notes"}, "title":"notes.txt"}
                ]
            }))
            .unwrap()
        };
        let requests = create_append_requests(
            vec![rich_message()],
            vec![
                rich_message(),
                Message::new_text(Role::Assistant, "answer"),
                rich_message(),
            ],
            &[0],
            &[2],
        )
        .await;
        let completions = requests
            .iter()
            .filter(|request| request.path.contains("/completion"))
            .collect::<Vec<_>>();
        assert_eq!(completions.len(), 2);
        assert_eq!(completions[0].body["prompt"], completions[1].body["prompt"]);
        assert_eq!(
            completions[0].body["attachments"],
            completions[1].body["attachments"]
        );
        assert_eq!(completions[0].body["files"], completions[1].body["files"]);
        assert_eq!(completions[1].body["files"], json!(["uploaded-file"]));
        assert_eq!(
            completions[1].body["attachments"][0]["file_name"],
            "notes.txt"
        );
        assert_eq!(
            completions[1].body["attachments"][0]["extracted_content"],
            "notes"
        );
    }

    #[tokio::test]
    async fn partial_initial_upload_failure_requires_reset() {
        let endpoint = partial_upload_failure_endpoint().await;
        let cache = ConversationCache::new();
        let key = ExplicitSessionKey::new("principal", "partial-upload");
        let operation = cache.try_lock_explicit_operation(&key).await.unwrap();
        let mut state = test_state(endpoint, cache.clone()).await;
        let image = || ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: "image/png".into(),
                data: "aW1hZ2U=".into(),
                file_name: None,
            },
            cache_control: None,
        };
        let request = params(vec![Message::new_blocks(
            Role::User,
            vec![image(), image()],
        )]);
        let send = send_and_stage_create(&mut state, &key, &request).await;
        assert!(state.finish_explicit_send(&key, send).await.is_err());
        drop(operation);
        let explicit = cache.get_explicit(&key).await.unwrap().explicit.unwrap();
        assert_eq!(explicit.state, ExplicitSessionState::Uncertain);
        assert_eq!(
            plan(
                Some(&explicit),
                &["user".into()],
                &["user:user".into()],
                "model",
                "system"
            )
            .unwrap_err()
            .code,
            "conversation_state_uncertain"
        );
        assert!(cache.reset_explicit(&key).await.unwrap());
        assert_eq!(
            plan(
                None,
                &["user".into()],
                &["user:user".into()],
                "model",
                "system"
            )
            .unwrap(),
            ExplicitReusePlan::Create
        );
    }

    #[tokio::test]
    async fn no_fs_text_only_explicit_session_commits_in_memory() {
        let (endpoint, _) = mock_endpoint().await;
        let cache = ConversationCache::new();
        let key = ExplicitSessionKey::new("principal", "aa".repeat(32));
        let operation = cache.try_lock_explicit_operation(&key).await.unwrap();
        let mut state = test_state(endpoint, cache.clone()).await;
        let request = params(vec![Message::new_text(Role::User, "hello")]);
        let upstream = send_and_stage_create(&mut state, &key, &request)
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
        let mut conversation = cached_conversation();
        conversation.cookie_id = "missing-cookie-id".into();
        conversation.model = "claude-sonnet-4-6".into();
        conversation.turns.push(CachedTurn {
            user_hashes: vec![1],
            assistant_uuid: "assistant".into(),
        });
        conversation.explicit = Some(ExplicitConversation {
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
                    crate::claude_web_state::explicit_session::digest_assistant_output("generated"),
                ),
            }],
            pending: None,
            file_mappings: Default::default(),
        });
        cache.set_explicit(key.clone(), conversation).await;
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
        let mut cached = cached_conversation();
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
            let mut conversation = cached_conversation();
            conversation.explicit = Some(ExplicitConversation {
                state: ExplicitSessionState::InFlight,
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: Vec::new(),
                pending: None,
                file_mappings: Default::default(),
            });
            cache.set_explicit(key.clone(), conversation).await;
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
