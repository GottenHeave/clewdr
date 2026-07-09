use serde::{Deserialize, Serialize};

use crate::types::claude::{ImageSource, OutputEffort, ThinkingMode};

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
        };

        let value = serde_json::to_value(body).unwrap();

        assert_eq!(value["effort"], json!("max"));
        assert_eq!(value["thinking_mode"], json!("auto"));
    }
}
