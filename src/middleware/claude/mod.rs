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

fn content_block_index(value: &Value) -> Option<usize> {
    value
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
}

fn is_thinking_block_start(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("content_block_start")
        && value
            .get("content_block")
            .and_then(|block| block.get("type"))
            .and_then(Value::as_str)
            == Some("thinking")
}

pub(crate) fn thinking_summary_delta_index(data: &str) -> Option<usize> {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return None;
    };
    thinking_summary_delta_text(&value)?;
    content_block_index(&value)
}

pub(crate) fn normalize_claude_web_stream_event(data: &str) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return data.to_owned();
    };

    let Some(event_type) = value.get("type").and_then(Value::as_str) else {
        return data.to_owned();
    };
    if !matches!(
        event_type,
        "message_start"
            | "content_block_start"
            | "content_block_delta"
            | "content_block_stop"
            | "message_delta"
            | "message_stop"
            | "ping"
            | "error"
    ) {
        return data.to_owned();
    }

    if is_thinking_block_start(&value) {
        if let Some(content_block) = value.get_mut("content_block") {
            *content_block = serde_json::json!({
                "type": "thinking",
                "signature": "",
                "thinking": "",
            });
        }
        return value.to_string();
    }

    let Some(summary) = thinking_summary_delta_text(&value).map(str::to_owned) else {
        return data.to_owned();
    };

    if let Some(delta) = value.get_mut("delta") {
        *delta = serde_json::json!({
            "type": "thinking_delta",
            "thinking": summary,
        });
    }

    value.to_string()
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
