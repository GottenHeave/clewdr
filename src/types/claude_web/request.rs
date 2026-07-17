use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::claude::{
    ContentBlock, ImageSource, Message, MessageContent, OutputEffort, ThinkingMode,
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
    /// Creates a text attachment with the default file name.
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
    pub text_blocks: Vec<String>,
    pub attachments: Vec<Attachment>,
    pub images: Vec<ImageSource>,
    pub identity: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExplicitContentErrorKind {
    InvalidRequest,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ExplicitContentError {
    pub kind: ExplicitContentErrorKind,
    pub message: &'static str,
}

impl ExplicitContentError {
    fn invalid(message: &'static str) -> Self {
        Self {
            kind: ExplicitContentErrorKind::InvalidRequest,
            message,
        }
    }
}

pub fn normalize_explicit_message(
    message: &Message,
) -> Result<NormalizedExplicitMessage, ExplicitContentError> {
    let mut text_blocks = Vec::new();
    let mut attachments = Vec::new();
    let mut images = Vec::new();
    let mut identity = Vec::new();
    match &message.content {
        MessageContent::Text { content } => {
            let text = content.trim();
            if !text.is_empty() {
                text_blocks.push(text.to_owned());
                identity.push(serde_json::json!({ "type": "text", "text": text }));
            }
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
    if text_blocks.is_empty() && attachments.is_empty() && images.is_empty() {
        return Err(ExplicitContentError::invalid(
            "Explicit session message has no forwardable content",
        ));
    }
    Ok(NormalizedExplicitMessage {
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
            let text = text.trim();
            if !text.is_empty() {
                text_blocks.push(text.to_owned());
                identity.push(serde_json::json!({ "type": "text", "text": text }));
            }
        }
        ContentBlock::Image { source, .. } => {
            let source = normalize_image(source)?;
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(source);
        }
        ContentBlock::ImageUrl { image_url } => {
            let source = ImageSource::from_data_url(&image_url.url).ok_or_else(|| {
                ExplicitContentError::invalid(
                    "Explicit sessions do not support remote or invalid image URLs",
                )
            })?;
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(source);
        }
        ContentBlock::Document { source, title, .. } => {
            normalize_document(source, title.as_deref(), attachments, images, identity)?;
        }
        ContentBlock::ContainerUpload { file_id, .. } => {
            let source = normalized_file_source(file_id)?;
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(source);
        }
        _ => {
            return Err(ExplicitContentError::invalid(
                "Explicit session content block is not forwardable",
            ));
        }
    }
    Ok(())
}

fn normalize_image(source: &ImageSource) -> Result<ImageSource, ExplicitContentError> {
    match source {
        ImageSource::Base64 { .. } => Ok(source.clone()),
        ImageSource::File { file_id } => normalized_file_source(file_id),
        ImageSource::Url { url } => ImageSource::from_data_url(url).ok_or_else(|| {
            ExplicitContentError::invalid(
                "Explicit sessions do not support remote or invalid image URLs",
            )
        }),
    }
}

fn normalized_file_source(file_id: &str) -> Result<ImageSource, ExplicitContentError> {
    let file_id = file_id.trim();
    if file_id.is_empty() {
        return Err(ExplicitContentError::invalid("File ID must not be empty"));
    }
    Ok(ImageSource::File {
        file_id: file_id.to_owned(),
    })
}

fn normalize_document(
    source: &Value,
    title: Option<&str>,
    attachments: &mut Vec<Attachment>,
    images: &mut Vec<ImageSource>,
    identity: &mut Vec<Value>,
) -> Result<(), ExplicitContentError> {
    let file_name = extract_document_file_name(source, title);
    match source.get("type").and_then(Value::as_str) {
        Some("text") => {
            let text = extract_document_text(source)
                .ok_or_else(|| ExplicitContentError::invalid("Document text must not be empty"))?;
            let attachment = match &file_name {
                Some(file_name) => Attachment::new_with_file_name(text, file_name),
                None => Attachment::new(text),
            };
            identity.push(serde_json::to_value(&attachment).expect("attachment serializes"));
            attachments.push(attachment);
        }
        Some("file") => {
            let file_id = extract_file_id(source)
                .ok_or_else(|| ExplicitContentError::invalid("Document file ID is missing"))?;
            let source = normalized_file_source(&file_id)?;
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(source);
        }
        Some("base64") => {
            let (media_type, data) = extract_base64_file(source).ok_or_else(|| {
                let message = match source.get("media_type").and_then(Value::as_str) {
                    Some(media_type) if !media_type.trim().is_empty() => "Document data is missing",
                    _ => "Document media type is missing",
                };
                ExplicitContentError::invalid(message)
            })?;
            let source = ImageSource::Base64 {
                media_type,
                data,
                file_name,
            };
            identity.push(serde_json::json!({ "type": "image", "source": source }));
            images.push(source);
        }
        _ => {
            return Err(ExplicitContentError::invalid(
                "Explicit session document source is not forwardable",
            ));
        }
    }
    Ok(())
}

pub(crate) fn extract_document_file_name(source: &Value, title: Option<&str>) -> Option<String> {
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

pub(crate) fn extract_document_text(source: &Value) -> Option<String> {
    (source.get("type").and_then(Value::as_str) == Some("text"))
        .then(|| {
            source
                .get("data")
                .or_else(|| source.get("text"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned)
        })
        .flatten()
}

pub(crate) fn extract_file_id(source: &Value) -> Option<String> {
    (source.get("type").and_then(Value::as_str) == Some("file"))
        .then(|| {
            source
                .get("file_id")
                .or_else(|| source.get("id"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|id| !id.is_empty())
                .map(ToOwned::to_owned)
        })
        .flatten()
}

pub(crate) fn extract_base64_file(source: &Value) -> Option<(String, String)> {
    if source.get("type").and_then(Value::as_str) != Some("base64") {
        return None;
    }
    let field = |name| {
        source
            .get(name)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    };
    Some((field("media_type")?, field("data")?))
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

    fn message(block: Value) -> Message {
        serde_json::from_value(json!({"role":"user", "content":[block]})).unwrap()
    }

    #[test]
    fn explicit_content_block_matrix_preserves_identity_forwarding_and_errors() {
        let equivalent = [
            (
                json!({"type":"text", "text":"hello"}),
                json!({"type":"text", "text":"hello", "cache_control":{"type":"ephemeral"}, "citations":[]}),
                (1, 0, 0),
            ),
            (
                json!({"type":"image", "source":{"type":"base64", "media_type":"image/png", "data":"aW1hZ2U="}}),
                json!({"type":"image_url", "image_url":{"url":"data:image/png;base64,aW1hZ2U="}}),
                (0, 0, 1),
            ),
        ];
        for (first, second, forwarded) in equivalent {
            let first = normalize_explicit_message(&message(first)).unwrap();
            let second = normalize_explicit_message(&message(second)).unwrap();
            assert_eq!(first.identity, second.identity);
            assert_eq!(first.images, second.images);
            assert_eq!(
                (
                    first.text_blocks.len(),
                    first.attachments.len(),
                    first.images.len()
                ),
                forwarded
            );
        }

        for block in [
            json!({"type":"document", "source":{"type":"url", "url":"https://example.com/a"}}),
            json!({"type":"document", "source":{"type":"text", "data":"   "}}),
            json!({"type":"document", "source":{"type":"base64", "media_type":"application/pdf", "data":""}}),
            json!({"type":"image_url", "image_url":{"url":"https://example.com/a.png"}}),
        ] {
            assert_eq!(
                normalize_explicit_message(&message(block))
                    .unwrap_err()
                    .kind,
                ExplicitContentErrorKind::InvalidRequest
            );
        }
    }
}
