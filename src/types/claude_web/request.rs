use serde::{Deserialize, Serialize};

use crate::types::claude::ImageSource;

/// Claude.ai attachment
#[derive(Deserialize, Serialize, Debug)]
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
        Attachment {
            file_size: content.len() as u64,
            extracted_content: content,
            file_name: "paste.txt".to_string(),
            file_type: "txt".to_string(),
        }
    }
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
