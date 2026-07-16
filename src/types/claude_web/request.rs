use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::claude::{
    ContentBlock, ImageSource, Message, MessageContent, OutputEffort, Role, ThinkingMode,
};

/// Claude.ai attachment
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Attachment {
    extracted_content: String,
    file_name: String,
    file_type: String,
    file_size: u64,
}

impl Attachment {
    /// Creates a new Attachment with the given content
    ///
    /// # Arguments
    /// * `content` - The text content for the attachment
    ///
    /// # Returns
    /// A new Attachment instance configured as a text file
    pub fn new(content: String) -> Self {
        Self::new_with_file_name(content, "paste.txt")
    }

    pub fn new_with_file_name(content: String, file_name: impl AsRef<str>) -> Self {
        let file_name =
            normalize_file_name(file_name.as_ref()).unwrap_or_else(|| "paste.txt".to_string());
        let file_type = file_type_from_file_name(&file_name).unwrap_or_else(|| "txt".to_string());

        Attachment {
            file_size: content.len() as u64,
            extracted_content: content,
            file_name,
            file_type,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NormalizedExplicitMessage {
    pub role: Role,
    pub text_blocks: Vec<String>,
    pub attachments: Vec<Attachment>,
    pub images: Vec<ImageSource>,
    pub identity: Value,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ExplicitContentError(pub &'static str);

pub fn normalize_explicit_message(
    message: &Message,
) -> Result<NormalizedExplicitMessage, ExplicitContentError> {
    let mut text_blocks = Vec::new();
    let mut attachments = Vec::new();
    let mut images = Vec::new();
    let mut identity = Vec::new();
    match &message.content {
        MessageContent::Text { content } => {
            text_blocks.push(content.trim().to_string());
            identity.push(serde_json::json!({ "type": "text", "text": content.trim() }));
        }
        MessageContent::Blocks { content } => {
            for block in content {
                normalize_explicit_block(
                    block,
                    &mut text_blocks,
                    &mut attachments,
                    &mut images,
                    &mut identity,
                )?;
            }
        }
    }
    if text_blocks.iter().all(|text| text.is_empty()) && attachments.is_empty() && images.is_empty()
    {
        return Err(ExplicitContentError(
            "Explicit session message has no forwardable content",
        ));
    }
    Ok(NormalizedExplicitMessage {
        role: message.role,
        text_blocks,
        attachments,
        images,
        identity: Value::Array(identity),
    })
}

fn normalize_explicit_block(
    block: &ContentBlock,
    text_blocks: &mut Vec<String>,
    attachments: &mut Vec<Attachment>,
    images: &mut Vec<ImageSource>,
    identity: &mut Vec<Value>,
) -> Result<(), ExplicitContentError> {
    match block {
        ContentBlock::Text { text, .. } => {
            text_blocks.push(text.trim().to_string());
            identity.push(serde_json::json!({ "type": "text", "text": text.trim() }));
        }
        ContentBlock::Image { source, .. } => {
            let forwarded = match source {
                ImageSource::Base64 { .. } | ImageSource::File { .. } => source.clone(),
                ImageSource::Url { url } => ImageSource::from_data_url(url).ok_or(
                    ExplicitContentError("Explicit sessions do not support remote image URLs"),
                )?,
            };
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(forwarded);
        }
        ContentBlock::ImageUrl { image_url } => {
            let forwarded = ImageSource::from_data_url(&image_url.url).ok_or(
                ExplicitContentError("Explicit sessions do not support remote image URLs"),
            )?;
            identity.push(serde_json::json!({ "type": "image_url", "url": image_url.url }));
            images.push(forwarded);
        }
        ContentBlock::Document {
            source,
            context,
            title,
            ..
        } => {
            let file_name = extract_document_file_name(source, title.as_deref());
            if let Some(text) = extract_document_text(source) {
                let attachment = match &file_name {
                    Some(file_name) => Attachment::new_with_file_name(text.clone(), file_name),
                    None => Attachment::new(text.clone()),
                };
                identity.push(serde_json::json!({
                    "type": "document_text", "text": text, "file_name": file_name,
                    "context": context,
                }));
                attachments.push(attachment);
            } else if let Some(file_id) = extract_file_id(source) {
                identity.push(serde_json::json!({ "type": "document_file", "file_id": file_id }));
                images.push(ImageSource::File { file_id });
            } else if let Some((media_type, data)) = extract_base64_file(source) {
                identity.push(serde_json::json!({
                    "type": "document_base64", "media_type": media_type,
                    "data": data, "file_name": file_name,
                }));
                images.push(ImageSource::Base64 {
                    media_type,
                    data,
                    file_name,
                });
            } else {
                return Err(ExplicitContentError(
                    "Explicit session document source is not forwardable",
                ));
            }
        }
        ContentBlock::ContainerUpload { file_id, .. } => {
            identity.push(serde_json::json!({ "type": "container_upload", "file_id": file_id }));
            images.push(ImageSource::File {
                file_id: file_id.clone(),
            });
        }
        ContentBlock::SearchResult { .. }
        | ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. }
        | ContentBlock::ToolUse { .. }
        | ContentBlock::ToolResult { .. }
        | ContentBlock::ToolReference { .. }
        | ContentBlock::ServerToolUse { .. }
        | ContentBlock::WebSearchToolResult { .. }
        | ContentBlock::WebFetchToolResult { .. }
        | ContentBlock::CodeExecutionToolResult { .. }
        | ContentBlock::BashCodeExecutionToolResult { .. }
        | ContentBlock::TextEditorCodeExecutionToolResult { .. }
        | ContentBlock::ToolSearchToolResult { .. }
        | ContentBlock::McpToolUse { .. }
        | ContentBlock::McpToolResult { .. }
        | ContentBlock::Unknown(_) => {
            return Err(ExplicitContentError(
                "Explicit session content block is not forwardable",
            ));
        }
    }
    Ok(())
}

pub fn extract_document_file_name(source: &Value, title: Option<&str>) -> Option<String> {
    title.and_then(normalize_file_name).or_else(|| {
        ["file_name", "filename", "name", "title"]
            .into_iter()
            .find_map(|key| {
                source
                    .get(key)
                    .and_then(Value::as_str)
                    .and_then(normalize_file_name)
            })
    })
}

pub fn extract_document_text(source: &Value) -> Option<String> {
    (source.get("type").and_then(Value::as_str)? == "text").then_some(())?;
    let text = source
        .get("data")
        .or_else(|| source.get("text"))
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

pub fn extract_file_id(source: &Value) -> Option<String> {
    (source.get("type").and_then(Value::as_str)? == "file").then_some(())?;
    source
        .get("file_id")
        .or_else(|| source.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

pub fn extract_base64_file(source: &Value) -> Option<(String, String)> {
    (source.get("type").and_then(Value::as_str)? == "base64").then_some(())?;
    let media_type = source
        .get("media_type")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|media_type| !media_type.is_empty())?
        .to_string();
    let data = source
        .get("data")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|data| !data.is_empty())?
        .to_string();
    Some((media_type, data))
}

pub fn normalize_file_name(file_name: &str) -> Option<String> {
    let file_name = file_name
        .trim()
        .rsplit(|c| c == '/' || c == '\\')
        .next()
        .unwrap_or_default()
        .trim();
    if file_name.is_empty() {
        return None;
    }

    let normalized = file_name
        .chars()
        .filter(|c| !c.is_control())
        .collect::<String>();
    let normalized = normalized.trim();
    (!normalized.is_empty()).then(|| normalized.to_string())
}

fn file_type_from_file_name(file_name: &str) -> Option<String> {
    file_name
        .rsplit_once('.')
        .and_then(|(_, extension)| normalize_file_name(extension))
}

/// Client-generated UUIDs for a single turn's messages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnMessageUuids {
    pub human_message_uuid: String,
    pub assistant_message_uuid: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateConversationParams {
    pub name: String,
    pub model: String,
    pub include_conversation_preferences: bool,
    pub paprika_mode: Option<String>,
    pub compass_mode: Option<String>,
    pub tool_search_mode: String,
    pub is_temporary: bool,
    pub enabled_imagine: bool,
}

/// Request body to be sent to the Claude.ai
#[derive(Deserialize, Serialize, Debug)]
pub struct WebRequestBody {
    pub max_tokens_to_sample: u32,
    pub attachments: Vec<Attachment>,
    pub files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<OutputEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_mode: Option<ThinkingMode>,
    pub rendering_mode: String,
    pub prompt: String,
    pub timezone: String,
    #[serde(skip)]
    pub images: Vec<ImageSource>,
    pub tools: Vec<Tool>,
    /// Parent message UUID for conversation continuation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_message_uuid: Option<String>,
    /// Client-generated UUIDs for this turn's messages
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_message_uuids: Option<TurnMessageUuids>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create_conversation_params: Option<CreateConversationParams>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Tool {
    pub name: String,
    #[serde(rename = "type")]
    pub type_: String,
}

impl Tool {
    pub fn web_search() -> Self {
        Tool {
            name: "web_search".to_string(),
            type_: "web_search_v0".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn web_request_body_serializes_effort_and_thinking_mode() {
        let body = WebRequestBody {
            max_tokens_to_sample: 1024,
            attachments: vec![],
            files: vec![],
            model: Some("claude-opus-4-8".to_string()),
            effort: Some(OutputEffort::Max),
            thinking_mode: Some(ThinkingMode::Auto),
            rendering_mode: "messages".to_string(),
            prompt: "hi".to_string(),
            timezone: "UTC".to_string(),
            images: vec![],
            tools: vec![],
            parent_message_uuid: None,
            turn_message_uuids: None,
            create_conversation_params: None,
        };

        let value = serde_json::to_value(body).unwrap();

        assert_eq!(value["effort"], json!("max"));
        assert_eq!(value["thinking_mode"], json!("auto"));
    }

    #[test]
    fn explicit_content_blocks_are_forwarded_or_rejected() {
        let supported = [
            json!({ "type": "text", "text": "hello" }),
            json!({
                "type": "image",
                "source": { "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=" }
            }),
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,aW1hZ2U=" }
            }),
            json!({
                "type": "document",
                "source": { "type": "text", "data": "document" }
            }),
            json!({ "type": "container_upload", "file_id": "file-id" }),
        ];
        for value in supported {
            let block: ContentBlock = serde_json::from_value(value).unwrap();
            let normalized =
                normalize_explicit_message(&Message::new_blocks(Role::User, vec![block])).unwrap();
            assert!(
                !normalized.text_blocks.is_empty()
                    || !normalized.attachments.is_empty()
                    || !normalized.images.is_empty()
            );
            assert_ne!(normalized.identity, json!([]));
        }

        let unsupported = [
            json!({ "type": "search_result", "content": [], "source": "s", "title": "t" }),
            json!({ "type": "thinking", "signature": "s", "thinking": "t" }),
            json!({ "type": "redacted_thinking", "data": "d" }),
            json!({ "type": "tool_use", "id": "i", "name": "n", "input": {} }),
            json!({ "type": "tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "tool_reference", "tool_name": "n" }),
            json!({ "type": "server_tool_use", "id": "i", "name": "n", "input": {} }),
            json!({ "type": "web_search_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "web_fetch_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "code_execution_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "bash_code_execution_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "text_editor_code_execution_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "tool_search_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "mcp_tool_use", "id": "i", "name": "n", "server_name": "s", "input": {} }),
            json!({ "type": "mcp_tool_result", "tool_use_id": "i", "content": null }),
            json!({ "type": "future_content", "value": true }),
        ];
        for value in unsupported {
            let block: ContentBlock = serde_json::from_value(value).unwrap();
            assert!(
                normalize_explicit_message(&Message::new_blocks(Role::User, vec![block])).is_err()
            );
        }
    }

    #[test]
    fn explicit_documents_and_remote_images_fail_closed() {
        let rejected = [
            json!({ "type": "document", "source": { "type": "url", "url": "https://example.com/a" } }),
            json!({ "type": "document", "source": { "type": "text", "data": "   " } }),
            json!({ "type": "image", "source": { "type": "url", "url": "https://example.com/a.png" } }),
        ];
        for value in rejected {
            let block: ContentBlock = serde_json::from_value(value).unwrap();
            assert!(
                normalize_explicit_message(&Message::new_blocks(Role::User, vec![block])).is_err()
            );
        }
    }
}
