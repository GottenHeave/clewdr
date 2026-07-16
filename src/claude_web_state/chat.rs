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
        for i in 0..CLEWDR_CONFIG.load().max_retries + 1 {
            if i > 0 {
                info!("[RETRY] attempt: {}", i.to_string().green());
            }
            let mut state = self.to_owned();
            let p = p.to_owned();

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

#[cfg(test)]
mod tests {
    use serde_json::json;
    use url::Url;

    use super::*;
    use crate::claude_web_state::ConversationCache;
    use crate::config::CookieStatus;
    use crate::services::cookie_actor::CookieActorHandle;
    use crate::types::claude::{OutputConfig, OutputEffort, Role, Thinking, ThinkingMode};

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
    async fn incomplete_downstream_stream_does_not_invalidate_reusable_conversation() {
        let cache = ConversationCache::new();
        let handle = CookieActorHandle::start().await.unwrap();
        let mut state = ClaudeWebState::new(handle, cache.clone());
        state.cookie = Some(CookieStatus::default());
        state.org_uuid = Some("org".to_string());
        state.endpoint = Url::parse("http://127.0.0.1:9/").unwrap();

        let first_message = Message::new_text(Role::User, "first");
        let params = CreateMessageParams {
            model: "model".to_string(),
            messages: vec![
                first_message.clone(),
                Message::new_text(Role::User, "second"),
            ],
            ..Default::default()
        };
        let key = state.cache_key_for(&params);
        // Cache reuse must not depend on whether the downstream response body was consumed.
        cache
            .set(
                key.clone(),
                CachedConversation {
                    conv_uuid: "conversation".to_string(),
                    org_uuid: "org".to_string(),
                    cookie_id: state.cookie_id(),
                    model: params.model.clone(),
                    is_pro: false,
                    system_hash: hash_system(&params.system),
                    turns: vec![CachedTurn {
                        user_hashes: vec![diff::hash_user_message(&first_message)],
                        assistant_uuid: "assistant".to_string(),
                    }],
                    created_at: chrono::Utc::now(),
                    last_used: chrono::Utc::now(),
                    valid: true,
                },
            )
            .await;

        let reuse_attempt = state.try_reuse_conversation(&params).await;

        assert!(
            reuse_attempt.is_some(),
            "an incomplete downstream stream must not turn a reusable cache entry into a miss"
        );
        assert!(cache.get(&key).await.is_some());
    }
}
