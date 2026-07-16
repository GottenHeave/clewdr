use std::sync::Arc;

use axum::{Extension, extract::State, response::Response};

use crate::{
    error::ClewdrError,
    middleware::claude::{ClaudeContext, ClaudeWebPreprocess},
    protocol::AuthPrincipal,
    providers::{
        LLMProvider,
        claude::{ClaudeInvocation, ClaudeProviderResponse, ClaudeWebProvider},
    },
};
/// Axum handler for the API messages
/// Main API endpoint for handling message requests to Claude
/// Processes messages, handles retries, and returns responses in stream or non-stream mode
///
/// # Arguments
/// * `XApiKey(_)` - API key authentication
/// * `state` - Application state containing client information
/// * `p` - Request body containing messages and configuration
///
/// # Returns
/// * `Response` - Stream or JSON response from Claude
pub async fn api_claude_web(
    State(provider): State<Arc<ClaudeWebProvider>>,
    Extension(principal): Extension<AuthPrincipal>,
    ClaudeWebPreprocess(params, context): ClaudeWebPreprocess,
) -> Result<(Extension<ClaudeContext>, Response), ClewdrError> {
    let ClaudeProviderResponse { context, response } = provider
        .invoke(ClaudeInvocation::protocol_messages(
            params,
            context.clone(),
            principal,
        ))
        .await?;
    Ok((Extension(context), response))
}
