use std::{sync::Arc, time::Instant};

use axum::response::Response;
use colored::Colorize;
use tracing::info;

use super::LLMProvider;
use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::ClaudeWebState,
    claude_web_state::conversation_cache::ConversationCache,
    error::ClewdrError,
    middleware::claude::{ClaudeApiFormat, ClaudeContext},
    protocol::{AuthPrincipal, files::StagedFileStore, sessions::ProtocolSessionStore},
    services::cookie_actor::CookieActorHandle,
    types::claude::CreateMessageParams,
    utils::{enabled, print_out_json},
};

#[derive(Clone, Copy)]
pub enum ClaudeOperation {
    Messages,
    CountTokens,
}

#[derive(Clone)]
pub struct ClaudeInvocation {
    pub params: CreateMessageParams,
    pub context: ClaudeContext,
    pub operation: ClaudeOperation,
    pub principal: Option<AuthPrincipal>,
}

impl ClaudeInvocation {
    pub fn messages(params: CreateMessageParams, context: ClaudeContext) -> Self {
        Self {
            params,
            context,
            operation: ClaudeOperation::Messages,
            principal: None,
        }
    }

    pub fn count_tokens(params: CreateMessageParams, context: ClaudeContext) -> Self {
        Self {
            params,
            context,
            operation: ClaudeOperation::CountTokens,
            principal: None,
        }
    }

    pub fn protocol_messages(
        params: CreateMessageParams,
        context: ClaudeContext,
        principal: AuthPrincipal,
    ) -> Self {
        Self {
            params,
            context,
            operation: ClaudeOperation::Messages,
            principal: Some(principal),
        }
    }
}

pub struct ClaudeProviderResponse {
    pub context: ClaudeContext,
    pub response: Response,
}

struct ClaudeSharedState {
    cookie_actor_handle: CookieActorHandle,
    conv_cache: ConversationCache,
    staged_files: Option<Arc<StagedFileStore>>,
    protocol_sessions: Arc<ProtocolSessionStore>,
}

impl ClaudeSharedState {
    fn new(
        cookie_actor_handle: CookieActorHandle,
        conv_cache: ConversationCache,
        staged_files: Option<Arc<StagedFileStore>>,
        protocol_sessions: Arc<ProtocolSessionStore>,
    ) -> Self {
        Self {
            cookie_actor_handle,
            conv_cache,
            staged_files,
            protocol_sessions,
        }
    }
}

#[derive(Clone)]
pub struct ClaudeProviders {
    web: Arc<ClaudeWebProvider>,
    code: Arc<ClaudeCodeProvider>,
}

impl ClaudeProviders {
    pub fn new(
        cookie_actor_handle: CookieActorHandle,
        conv_cache: ConversationCache,
        staged_files: Option<Arc<StagedFileStore>>,
        protocol_sessions: Arc<ProtocolSessionStore>,
    ) -> Self {
        let shared = Arc::new(ClaudeSharedState::new(
            cookie_actor_handle,
            conv_cache,
            staged_files,
            protocol_sessions,
        ));
        let web = Arc::new(ClaudeWebProvider::new(shared.clone()));
        let code = Arc::new(ClaudeCodeProvider::new(shared.clone()));
        Self { web, code }
    }

    pub fn web(&self) -> Arc<ClaudeWebProvider> {
        self.web.clone()
    }

    pub fn code(&self) -> Arc<ClaudeCodeProvider> {
        self.code.clone()
    }
}

#[derive(Clone)]
pub struct ClaudeWebProvider {
    shared: Arc<ClaudeSharedState>,
}

impl ClaudeWebProvider {
    fn new(shared: Arc<ClaudeSharedState>) -> Self {
        Self { shared }
    }
}

#[async_trait::async_trait]
impl LLMProvider for ClaudeWebProvider {
    type Request = ClaudeInvocation;
    type Output = ClaudeProviderResponse;

    async fn invoke(&self, request: Self::Request) -> Result<Self::Output, ClewdrError> {
        let mut state = ClaudeWebState::new(
            self.shared.cookie_actor_handle.clone(),
            self.shared.conv_cache.clone(),
        );
        state.staged_files = self.shared.staged_files.clone();
        state.protocol_sessions = Some(self.shared.protocol_sessions.clone());
        let stream = request.context.is_stream();
        state.api_format = request.context.api_format();
        state.stream = stream;
        state.usage = request.context.usage().to_owned();
        let ClaudeInvocation {
            params,
            context,
            operation,
            principal,
        } = request;
        state.principal = principal;
        if !matches!(operation, ClaudeOperation::Messages) {
            return Err(ClewdrError::BadRequest {
                msg: "Unsupported operation for Claude Web",
            });
        }
        let format_display = match context.api_format() {
            ClaudeApiFormat::Claude => ClaudeApiFormat::Claude.to_string().green(),
            ClaudeApiFormat::OpenAI => ClaudeApiFormat::OpenAI.to_string().yellow(),
        };
        info!(
            "[REQ] stream: {}, msgs: {}, model: {}, think: {}, format: {}",
            enabled(stream),
            params.messages.len().to_string().green(),
            params.model.green(),
            enabled(params.thinking.is_some()),
            format_display
        );
        print_out_json(&params, "claude_web_client_req.json");
        let stopwatch = Instant::now();
        let response = state.try_chat(params).await?;
        let elapsed = stopwatch.elapsed();
        info!(
            "[FIN] elapsed: {}s",
            format!("{}", elapsed.as_secs_f32()).green()
        );
        Ok(ClaudeProviderResponse { context, response })
    }
}

#[derive(Clone)]
pub struct ClaudeCodeProvider {
    shared: Arc<ClaudeSharedState>,
}

impl ClaudeCodeProvider {
    fn new(shared: Arc<ClaudeSharedState>) -> Self {
        Self { shared }
    }
}

#[async_trait::async_trait]
impl LLMProvider for ClaudeCodeProvider {
    type Request = ClaudeInvocation;
    type Output = ClaudeProviderResponse;

    async fn invoke(&self, request: Self::Request) -> Result<Self::Output, ClewdrError> {
        let mut state = ClaudeCodeState::new(self.shared.cookie_actor_handle.clone());
        state.api_format = request.context.api_format();
        state.stream = request.context.is_stream();
        state.system_prompt_hash = request.context.system_prompt_hash();
        state.anthropic_beta_header = request.context.anthropic_beta().map(str::to_string);
        state.usage = request.context.usage().to_owned();
        let ClaudeInvocation {
            params,
            context,
            operation,
            principal: _,
        } = request;
        match operation {
            ClaudeOperation::Messages => {
                let format_display = match context.api_format() {
                    ClaudeApiFormat::Claude => ClaudeApiFormat::Claude.to_string().green(),
                    ClaudeApiFormat::OpenAI => ClaudeApiFormat::OpenAI.to_string().yellow(),
                };
                info!(
                    "[REQ] stream: {}, msgs: {}, model: {}, format: {}",
                    enabled(state.stream),
                    params.messages.len().to_string().green(),
                    params.model.green(),
                    format_display
                );
                print_out_json(&params, "claude_code_client_req.json");
                let stopwatch = Instant::now();
                let response = state.try_chat(params).await?;
                let elapsed = stopwatch.elapsed();
                info!(
                    "[FIN] elapsed: {}s",
                    format!("{}", elapsed.as_secs_f32()).green()
                );
                Ok(ClaudeProviderResponse { context, response })
            }
            ClaudeOperation::CountTokens => {
                info!(
                    "[TOKENS] msgs: {}, model: {}",
                    params.messages.len().to_string().green(),
                    params.model.green()
                );
                let stopwatch = Instant::now();
                let response = state.try_count_tokens(params, context.is_web()).await?;
                let elapsed = stopwatch.elapsed();
                info!(
                    "[TOKENS] elapsed: {}s",
                    format!("{}", elapsed.as_secs_f32()).green()
                );
                Ok(ClaudeProviderResponse { context, response })
            }
        }
    }
}

pub fn build_providers(
    cookie_actor_handle: CookieActorHandle,
    conv_cache: ConversationCache,
    staged_files: Option<Arc<StagedFileStore>>,
    protocol_sessions: Arc<ProtocolSessionStore>,
) -> ClaudeProviders {
    ClaudeProviders::new(
        cookie_actor_handle,
        conv_cache,
        staged_files,
        protocol_sessions,
    )
}
