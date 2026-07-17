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
        self.explicit_file_key = Some(key.clone());
        self.explicit_completion_started = false;
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
            let stop = if mock.incomplete_completion {
                ""
            } else {
                "data: {\"type\":\"message_stop\"}\n\n"
            };
            (
                "text/event-stream",
                StatusCode::OK,
                format!("data: {{\"completion\":\"answer\"}}\n\n{stop}"),
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
    async fn missing_bound_cookie_tombstones_session_before_upstream() {
        let handle = crate::services::cookie_actor::CookieActorHandle::start()
            .await
            .unwrap();
        let cache = ConversationCache::new();
        let principal = crate::protocol::AuthPrincipal::for_authenticated_user();
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
        let error = state
            .try_explicit_chat(params(vec![first]), digest)
            .await
            .unwrap_err();
        let ClewdrError::Protocol { source } = error else {
            panic!("expected protocol error");
        };
        assert_eq!(source.status, StatusCode::GONE);
        assert_eq!(source.code, "conversation_expired");
        assert_eq!(
            explicit_test_state(&cache, &key).await,
            ExplicitSessionState::Tombstoned
        );
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
