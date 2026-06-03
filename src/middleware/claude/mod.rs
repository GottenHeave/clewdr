mod claude2oai;
mod request;
mod response;
mod stop_sequences;

pub(crate) use claude2oai::*;
pub use request::*;
pub use response::*;
use serde_json::Value;
pub use stop_sequences::*;
use strum::Display;

use crate::types::claude::Usage;

/// Represents the format of the API response
///
/// This enum defines the available API response formats that Clewdr can use
/// when communicating with clients. It supports both Claude's native format
/// and an OpenAI-compatible format for broader compatibility with existing tools.
#[derive(Display, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaudeApiFormat {
    /// Claude native format
    Claude,
    /// OpenAI compatible format
    OpenAI,
}

fn thinking_summary_delta_text(data: &Value) -> Option<&str> {
    if data.get("type").and_then(Value::as_str) != Some("content_block_delta") {
        return None;
    }

    let delta = data.get("delta")?;
    if delta.get("type").and_then(Value::as_str) != Some("thinking_summary_delta") {
        return None;
    }

    delta
        .get("summary")
        .and_then(|summary| {
            summary
                .get("summary")
                .and_then(Value::as_str)
                .or_else(|| summary.as_str())
        })
        .filter(|summary| !summary.is_empty())
}

pub(crate) fn normalize_claude_web_stream_event(data: &str) -> Option<String> {
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return Some(data.to_owned());
    };

    if value.get("type").and_then(Value::as_str) == Some("message_limit") {
        return None;
    }

    let Some(summary) = thinking_summary_delta_text(&value).map(str::to_owned) else {
        return Some(data.to_owned());
    };

    if let Some(delta) = value.get_mut("delta") {
        *delta = serde_json::json!({
            "type": "thinking_delta",
            "thinking": summary,
        });
    }

    Some(value.to_string())
}

#[derive(Debug, Clone)]
pub enum ClaudeContext {
    Web(ClaudeWebContext),
    Code(ClaudeCodeContext),
}

impl ClaudeContext {
    pub fn is_stream(&self) -> bool {
        match self {
            ClaudeContext::Web(ctx) => ctx.stream,
            ClaudeContext::Code(ctx) => ctx.stream,
        }
    }

    pub fn api_format(&self) -> ClaudeApiFormat {
        match self {
            ClaudeContext::Web(ctx) => ctx.api_format,
            ClaudeContext::Code(ctx) => ctx.api_format,
        }
    }

    pub fn is_web(&self) -> bool {
        matches!(self, ClaudeContext::Web(_))
    }

    pub fn is_code(&self) -> bool {
        matches!(self, ClaudeContext::Code(_))
    }

    pub fn stop_sequences(&self) -> &[String] {
        match self {
            ClaudeContext::Web(ctx) => &ctx.stop_sequences,
            ClaudeContext::Code(_) => &[],
        }
    }

    pub fn system_prompt_hash(&self) -> Option<u64> {
        match self {
            ClaudeContext::Web(_) => None,
            ClaudeContext::Code(ctx) => ctx.system_prompt_hash,
        }
    }

    pub fn usage(&self) -> &Usage {
        match self {
            ClaudeContext::Web(ctx) => &ctx.usage,
            ClaudeContext::Code(ctx) => &ctx.usage,
        }
    }

    pub fn anthropic_beta(&self) -> Option<&str> {
        match self {
            ClaudeContext::Web(_) => None,
            ClaudeContext::Code(ctx) => ctx.anthropic_beta.as_deref(),
        }
    }
}
