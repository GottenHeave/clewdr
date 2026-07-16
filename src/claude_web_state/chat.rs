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
    claude_web_state::conversation_cache::{CachedConversation, CachedTurn},
    claude_web_state::diff::{self, DiffResult, extract_user_hashes, hash_system},
    config::CLEWDR_CONFIG,
    error::{CheckClaudeErr, ClewdrError, WreqSnafu},
    protocol::{
        ProtocolError, parse_session_id,
        sessions::{
            PendingTurn, ReusePlan, SessionLifecycle, digest_message_timeline, digest_model,
            digest_system, digest_user_messages, selected_parent_message_timeline,
        },
    },
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
        if let Some(session_digest) = parse_session_id(session_id)? {
            return self.try_protocol_chat(p, session_digest).await;
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

    async fn try_protocol_chat(
        &mut self,
        p: CreateMessageParams,
        session_digest: String,
    ) -> Result<axum::response::Response, ClewdrError> {
        let principal = self.principal.clone().ok_or(ClewdrError::InvalidAuth)?;
        let sessions = self
            .protocol_sessions
            .clone()
            .expect("protocol session store is configured");
        let operation = sessions.try_begin(&principal, &session_digest).await?;
        let user_entries = digest_user_messages(&p.messages);
        if user_entries.is_empty() {
            return Err(ProtocolError::new(
                http::StatusCode::BAD_REQUEST,
                "conversation_reuse_failed",
                "A protocol request must contain user content",
            )
            .into());
        }
        let user_digests = user_entries
            .iter()
            .map(|(_, digest)| digest.clone())
            .collect::<Vec<_>>();
        let message_timeline = digest_message_timeline(&p.messages);
        let model_digest = digest_model(&p.model);
        let system_digest = digest_system(&p.system);
        let plan = sessions
            .plan(
                &operation,
                &user_digests,
                &message_timeline,
                &model_digest,
                &system_digest,
            )
            .await?;
        let existing = sessions.get(&operation).await;
        let parent_message_timeline = match &plan {
            ReusePlan::Create => Vec::new(),
            _ => selected_parent_message_timeline(
                &existing
                    .as_ref()
                    .expect("reuse plan requires an existing session")
                    .turns,
                &plan,
            )?,
        };

        let cookie_result = self
            .request_session_cookie(
                &session_digest,
                existing.as_ref().map(|session| session.cookie_id.as_str()),
            )
            .await;
        let _cookie = match cookie_result {
            Err(ClewdrError::NoCookieAvailable) if existing.is_some() => {
                sessions.tombstone(&operation).await?;
                return Err(ProtocolError::new(
                    http::StatusCode::GONE,
                    "conversation_expired",
                    "The session Cookie is no longer available",
                )
                .into());
            }
            result => result?,
        };
        self.bootstrap().await?;
        let organization_uuid = self.org_uuid.clone().ok_or(ClewdrError::UnexpectedNone {
            msg: "Organization UUID is not set",
        })?;
        if let Some(existing) = &existing
            && (existing.cookie_id != self.cookie_id()
                || existing.organization_uuid != organization_uuid)
        {
            sessions.tombstone(&operation).await?;
            return Err(ProtocolError::new(
                http::StatusCode::GONE,
                "conversation_expired",
                "The persisted Cookie or organization is no longer available",
            )
            .into());
        }

        let human_uuid = uuid::Uuid::new_v4().to_string();
        let assistant_uuid = uuid::Uuid::new_v4().to_string();
        let (conversation_uuid, parent_uuid, suffix_start, replace_from_turn, is_new) = match plan {
            ReusePlan::Create => (uuid::Uuid::new_v4().to_string(), None, 0, 0, true),
            ReusePlan::Append {
                parent_uuid,
                suffix_start,
            } => (
                existing.as_ref().unwrap().conversation_uuid.clone(),
                Some(parent_uuid),
                suffix_start,
                existing.as_ref().unwrap().turns.len(),
                false,
            ),
            ReusePlan::Fork {
                parent_uuid,
                suffix_start,
                replace_from_turn,
            }
            | ReusePlan::Regenerate {
                parent_uuid,
                suffix_start,
                replace_from_turn,
            } => (
                existing.as_ref().unwrap().conversation_uuid.clone(),
                parent_uuid,
                suffix_start,
                replace_from_turn,
                false,
            ),
        };
        self.conv_uuid = Some(conversation_uuid.clone());
        self.last_params = Some(p.clone());
        self.sync_model_selector_state(&p).await?;
        let pending = PendingTurn {
            parent_uuid_before: parent_uuid.clone(),
            user_digests: user_digests[suffix_start..].to_vec(),
            assistant_uuid_after: assistant_uuid.clone(),
            replace_from_turn,
            parent_message_timeline: Some(parent_message_timeline),
            request_message_timeline: Some(message_timeline),
        };

        let body = if is_new {
            sessions
                .create_provisional(
                    &operation,
                    &principal,
                    &session_digest,
                    self.cookie_id(),
                    organization_uuid.clone(),
                    conversation_uuid.clone(),
                    model_digest,
                    system_digest,
                    pending,
                )
                .await?;
            let mut body = self
                .transform_request(p.clone())
                .ok_or(ClewdrError::BadRequest {
                    msg: "Request body is empty",
                })?;
            body.create_conversation_params = Some(create_conversation_params(
                &p,
                !CLEWDR_CONFIG.load().preserve_chats,
                self.is_pro(),
            ));
            body.turn_message_uuids = Some(TurnMessageUuids {
                human_message_uuid: human_uuid,
                assistant_message_uuid: assistant_uuid,
            });
            let images = body.images.drain(..).collect::<Vec<_>>();
            body.files = match self
                .upload_protocol_files(images, &organization_uuid, &conversation_uuid, &operation)
                .await
            {
                Ok(files) => files,
                Err(error) => {
                    sessions.mark_uncertain(&operation).await?;
                    return Err(error);
                }
            };
            serde_json::to_value(body)?
        } else {
            sessions.start_existing(&operation, pending).await?;
            let user_messages = user_entries[suffix_start..]
                .iter()
                .map(|(message_index, _)| &p.messages[*message_index])
                .collect::<Vec<_>>();
            let bundled = self.bundle_user_messages(&user_messages);
            let files = match self
                .upload_protocol_files(
                    bundled.images.clone(),
                    &organization_uuid,
                    &conversation_uuid,
                    &operation,
                )
                .await
            {
                Ok(files) => files,
                Err(error) => {
                    sessions
                        .restore_committed_before_completion(&operation)
                        .await?;
                    return Err(error);
                }
            };
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
            if self.is_pro() {
                body["model"] = json!(p.model);
            }
            if let Some(effort) = p.web_thinking_effort() {
                body["effort"] = json!(effort);
            }
            if let Some(mode) = p.web_thinking_mode() {
                body["thinking_mode"] = json!(mode);
            }
            body
        };
        print_out_json(&body, "claude_web_protocol_req.json");
        let endpoint = self
            .endpoint
            .join(&format!(
                "api/organizations/{organization_uuid}/chat_conversations/{conversation_uuid}/completion"
            ))
            .expect("URL path components are generated UUIDs");
        let response = self
            .build_request(Method::POST, endpoint)
            .json(&body)
            .header(ACCEPT, "text/event-stream")
            .send()
            .await
            .context(WreqSnafu {
                msg: "Failed to send protocol chat request",
            });
        let response = match response {
            Ok(response) => match response.check_claude().await {
                Ok(response) => response,
                Err(ClewdrError::ClaudeHttpError { code, .. })
                    if code == http::StatusCode::NOT_FOUND || code == http::StatusCode::GONE =>
                {
                    sessions.tombstone(&operation).await?;
                    return Err(ProtocolError::new(
                        http::StatusCode::GONE,
                        "conversation_expired",
                        "The upstream conversation no longer exists",
                    )
                    .into());
                }
                Err(error) => {
                    sessions.mark_uncertain(&operation).await?;
                    return Err(error);
                }
            },
            Err(error) => {
                sessions.mark_uncertain(&operation).await?;
                return Err(error);
            }
        };
        self.protocol_lifecycle = Some(SessionLifecycle::new(sessions, operation));
        self.transform_response(response).await
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
                        &parent_uuid,
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

        // === Prepare cache write ===
        if write_cache {
            let user_hashes = extract_user_hashes(&p.messages)
                .iter()
                .map(|(_, h)| *h)
                .collect();
            let sys_hash = hash_system(&p.system);
            let stream_flag = self
                .stream_health_flag
                .clone()
                .unwrap_or_else(|| Arc::new(AtomicBool::new(true)));

            self.pending_cache_write = Some(PendingCacheWrite::Init {
                key: self.cache_key_for(&p),
                conv: CachedConversation {
                    conv_uuid: new_uuid.clone(),
                    org_uuid: org_uuid.clone(),
                    cookie_id: self.cookie_id(),
                    model: p.model.clone(),
                    is_pro: self.is_pro(),
                    system_hash: sys_hash,
                    turns: vec![CachedTurn {
                        user_hashes,
                        assistant_uuid,
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                    last_stream_healthy: stream_flag,
                },
            });
        }

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
                parent_uuid,
                &human_uuid,
                &assistant_uuid,
                p,
            )
            .await?;

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

        // Prepare optimistic cache write
        self.pending_cache_write = Some(PendingCacheWrite::AppendTurn {
            key: self.cache_key_for(p),
            turn: CachedTurn {
                user_hashes: new_user_hashes.to_vec(),
                assistant_uuid,
            },
        });

        Ok(response)
    }

    /// Incremental send with fork — edit scenario
    async fn send_incremental_fork(
        &mut self,
        cached: &CachedConversation,
        parent_uuid: &str,
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

        // Prepare fork cache write
        self.pending_cache_write = Some(PendingCacheWrite::ForkAndAppend {
            key: self.cache_key_for(p),
            fork_turn_index,
            turn: CachedTurn {
                user_hashes: remaining_user_hashes.to_vec(),
                assistant_uuid,
            },
        });

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
        parent_uuid: &str,
        human_uuid: &str,
        assistant_uuid: &str,
        p: &CreateMessageParams,
    ) -> Result<serde_json::Value, ClewdrError> {
        let files = self
            .upload_files(bundled.images.clone(), org_uuid, conversation_uuid)
            .await?;
        let mut body = json!({
            "prompt": bundled.prompt,
            "parent_message_uuid": parent_uuid,
            "timezone": TIME_ZONE.to_string(),
            "turn_message_uuids": {
                "human_message_uuid": human_uuid,
                "assistant_message_uuid": assistant_uuid,
            },
            "attachments": bundled.attachments,
            "files": files,
            "rendering_mode": if p.stream.unwrap_or_default() { "messages" } else { "raw" },
        });
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
                self.conv_cache.set(key, conv).await;
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        claude_web_state::conversation_cache::ConversationCache,
        protocol::{AuthPrincipal, sessions::ProtocolSessionStore},
        services::cookie_actor::CookieActorHandle,
        types::claude::{OutputConfig, OutputEffort, Role, Thinking, ThinkingMode},
    };

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

    #[tokio::test]
    async fn regenerate_with_assistant_prefill_fails_before_upstream_request() {
        let params = CreateMessageParams {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![
                Message::new_text(Role::User, "u1"),
                Message::new_text(Role::Assistant, "prefill"),
            ],
            ..Default::default()
        };
        let principal = AuthPrincipal::for_authenticated_user();
        let session_digest = "fa".repeat(32);
        let sessions = ProtocolSessionStore::memory();
        let operation = sessions
            .try_begin(&principal, &session_digest)
            .await
            .unwrap();
        let user_digests = digest_user_messages(&params.messages)
            .into_iter()
            .map(|(_, digest)| digest)
            .collect::<Vec<_>>();
        sessions
            .create_provisional(
                &operation,
                &principal,
                &session_digest,
                "cookie".into(),
                "org".into(),
                "conv".into(),
                digest_model(&params.model),
                digest_system(&params.system),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests,
                    assistant_uuid_after: "assistant".into(),
                    replace_from_turn: 0,
                    parent_message_timeline: Some(vec![]),
                    request_message_timeline: Some(digest_message_timeline(&params.messages)),
                },
            )
            .await
            .unwrap();
        sessions
            .commit(
                &operation,
                Some(crate::protocol::sessions::digest_assistant_output(
                    "generated",
                )),
            )
            .await
            .unwrap();
        drop(operation);

        let handle = CookieActorHandle::start().await.unwrap();
        let mut state = ClaudeWebState::new(handle, ConversationCache::new());
        state.principal = Some(principal);
        state.protocol_sessions = Some(sessions);
        let error = state
            .try_protocol_chat(params, session_digest)
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol validation error");
        };
        assert_eq!(source.status, http::StatusCode::CONFLICT);
        assert_eq!(source.code, "conversation_reuse_failed");
    }
}
