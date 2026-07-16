use std::{fmt::Write, mem};

use base64::{Engine, prelude::BASE64_STANDARD};
use futures::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use serde_json::Value;
use snafu::GenerateImplicitData;
use tracing::{debug, warn};
use url::Url;
use wreq::multipart::{Form, Part};

use crate::protocol::sessions::SessionOperation;
use crate::{
    claude_web_state::ClaudeWebState,
    config::CLEWDR_CONFIG,
    error::CheckClaudeErr,
    types::{
        claude::{ContentBlock, CreateMessageParams, ImageSource, Message, MessageContent, Role},
        claude_web::request::*,
    },
    utils::{TIME_ZONE, print_out_text},
};

impl ClaudeWebState {
    pub fn transform_request(&self, mut value: CreateMessageParams) -> Option<WebRequestBody> {
        let system = value.system.take();
        let msgs = mem::take(&mut value.messages);
        let system = merge_system(system.unwrap_or_default());
        let merged = merge_messages(msgs, system)?;

        let mut tools = vec![];
        if CLEWDR_CONFIG.load().web_search {
            tools.push(Tool::web_search());
        }
        let effort = value.web_thinking_effort();
        let thinking_mode = value.web_thinking_mode();
        Some(WebRequestBody {
            max_tokens_to_sample: value.max_tokens,
            attachments: merged.attachments,
            files: vec![],
            model: if self.is_pro() {
                Some(value.model)
            } else {
                None
            },
            effort,
            thinking_mode,
            rendering_mode: if value.stream.unwrap_or_default() {
                "messages".to_string()
            } else {
                "raw".to_string()
            },
            prompt: merged.prompt,
            timezone: TIME_ZONE.to_string(),
            images: merged.images,
            tools,
            parent_message_uuid: None,
            turn_message_uuids: None,
            create_conversation_params: None,
        })
    }

    /// Upload files through the conversation-scoped Claude.ai file service.
    pub async fn upload_files(
        &self,
        files: Vec<ImageSource>,
        org_uuid: &str,
        conversation_uuid: &str,
    ) -> Result<Vec<String>, crate::error::ClewdrError> {
        let endpoint = conversation_upload_endpoint(&self.endpoint, org_uuid, conversation_uuid)
            .map_err(|source| crate::error::ClewdrError::Whatever {
                message: "Failed to build conversation upload URL".to_string(),
                source: Some(Box::new(source)),
            })?;

        stream::iter(files)
            .map(|file| {
                let endpoint = endpoint.clone();
                async move {
                    let ImageSource::Base64 {
                        media_type,
                        data,
                        file_name,
                    } = file
                    else {
                        if let ImageSource::File { file_id } = file {
                            return Ok(file_id);
                        }
                        return Err(crate::error::ClewdrError::BadRequest {
                            msg: "Unsupported URL file source",
                        });
                    };
                    let bytes = BASE64_STANDARD.decode(data).map_err(|error| {
                        warn!("Failed to decode uploaded file: {error}");
                        crate::error::ClewdrError::BadRequest {
                            msg: "Invalid base64 file data",
                        }
                    })?;
                    let main_type = media_type.split(';').next().unwrap_or(&media_type);
                    let file_name = file_name
                        .as_deref()
                        .and_then(normalize_file_name)
                        .unwrap_or_else(|| default_upload_file_name(main_type).to_string());
                    let part = Part::bytes(bytes)
                        .file_name(file_name)
                        .mime_str(main_type)
                        .map_err(|error| crate::error::ClewdrError::Whatever {
                            message: "Invalid uploaded file media type".to_string(),
                            source: Some(Box::new(error)),
                        })?;
                    self.upload_file_part(
                        endpoint,
                        part,
                        "Failed to upload file",
                        "Failed to parse file upload response",
                    )
                    .await
                }
            })
            .buffered(5)
            .try_collect()
            .await
    }

    pub async fn upload_protocol_files(
        &self,
        files: Vec<ImageSource>,
        org_uuid: &str,
        conversation_uuid: &str,
        operation: &SessionOperation,
    ) -> Result<Vec<String>, crate::error::ClewdrError> {
        let staged_store = self.staged_files.as_ref().ok_or_else(|| {
            crate::protocol::ProtocolError::new(
                http::StatusCode::NOT_IMPLEMENTED,
                "staged_files_unavailable",
                "Staged file references require filesystem persistence",
            )
        })?;
        let session_store = self
            .protocol_sessions
            .as_ref()
            .expect("protocol session store is configured");
        let principal = self.principal.as_ref().expect("authenticated principal");
        let endpoint = conversation_upload_endpoint(&self.endpoint, org_uuid, conversation_uuid)
            .map_err(|source| crate::error::ClewdrError::Whatever {
                message: "Failed to build conversation upload URL".to_string(),
                source: Some(Box::new(source)),
            })?;
        let mut uploaded = Vec::with_capacity(files.len());
        for file in files {
            let ImageSource::File { file_id } = file else {
                uploaded.extend(
                    self.upload_files(vec![file], org_uuid, conversation_uuid)
                        .await?,
                );
                continue;
            };
            if !file_id.starts_with("file_clewdr_v1_") {
                uploaded.push(file_id);
                continue;
            }
            if let Some(existing) = session_store.file_mapping(operation, &file_id).await {
                uploaded.push(existing);
                continue;
            }
            let staged = staged_store.resolve(principal, &file_id).await?;
            let part = Part::file(&staged.path)
                .await
                .map_err(|source| crate::error::ClewdrError::IoError {
                    loc: snafu::Location::generate(),
                    source,
                })?
                .file_name(staged.filename.clone())
                .mime_str(&staged.mime_type)
                .map_err(|error| crate::error::ClewdrError::Whatever {
                    message: "Invalid staged file media type".to_string(),
                    source: Some(Box::new(error)),
                })?;
            let file_uuid = self
                .upload_file_part(
                    endpoint.clone(),
                    part,
                    "Failed to upload staged file",
                    "Failed to parse staged file upload response",
                )
                .await?;
            if let Some(session) = session_store.get(operation).await {
                staged_store
                    .add_reference(&file_id, &session.session_ref())
                    .await?;
            }
            session_store
                .put_file_mapping(operation, &file_id, &file_uuid)
                .await?;
            uploaded.push(file_uuid);
        }
        Ok(uploaded)
    }

    async fn upload_file_part(
        &self,
        endpoint: Url,
        part: Part,
        upload_error: &'static str,
        parse_error: &'static str,
    ) -> Result<String, crate::error::ClewdrError> {
        let response = self
            .build_request(http::Method::POST, endpoint)
            .multipart(Form::new().part("file", part))
            .send()
            .await
            .map_err(|source| crate::error::ClewdrError::WreqError {
                msg: upload_error,
                source,
            })?
            .check_claude()
            .await?;
        #[derive(serde::Deserialize)]
        struct UploadResponse {
            file_uuid: String,
        }
        response
            .json::<UploadResponse>()
            .await
            .map(|upload| upload.file_uuid)
            .map_err(|source| crate::error::ClewdrError::WreqError {
                msg: parse_error,
                source,
            })
    }
}

fn conversation_upload_endpoint(
    endpoint: &Url,
    org_uuid: &str,
    conversation_uuid: &str,
) -> Result<Url, url::ParseError> {
    endpoint.join(&format!(
        "api/organizations/{org_uuid}/conversations/{conversation_uuid}/wiggle/upload-file"
    ))
}

/// Merged messages and images
#[derive(Default, Debug)]
struct Merged {
    pub attachments: Vec<Attachment>,
    pub prompt: String,
    pub images: Vec<ImageSource>,
}

/// Merges multiple messages into a single text prompt, handling system instructions
/// and extracting any images from the messages
///
/// # Arguments
/// * `msgs` - Vector of messages to merge
/// * `system` - System instructions to prepend
///
/// # Returns
/// * `Option<Merged>` - Merged prompt text, images, and additional metadata, or None if merging fails
fn merge_messages(msgs: Vec<Message>, system: String) -> Option<Merged> {
    if msgs.is_empty() {
        return None;
    }
    let h = CLEWDR_CONFIG
        .load()
        .custom_h
        .to_owned()
        .unwrap_or("Human".to_string());
    let a = CLEWDR_CONFIG
        .load()
        .custom_a
        .to_owned()
        .unwrap_or("Assistant".to_string());

    let user_real_roles = CLEWDR_CONFIG.load().use_real_roles;
    let line_breaks = if user_real_roles { "\n\n\x08" } else { "\n\n" };
    let system = system.trim().to_string();
    let mut w = String::new();
    let mut prompt_parts: Vec<(Role, String)> = vec![];
    let mut attachments: Vec<Attachment> = vec![];

    let mut imgs: Vec<ImageSource> = vec![];

    let chunks = msgs
        .into_iter()
        .filter_map(|m| match m.content {
            MessageContent::Blocks { content } => {
                // collect all text blocks, join them with new line
                let blocks = content
                    .into_iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text, .. } => Some(text.trim().to_string()),
                        ContentBlock::Image { source, .. } => {
                            match source {
                                ImageSource::Base64 { .. } => {
                                    // push image to the list
                                    imgs.push(source);
                                }
                                ImageSource::Url { url } => {
                                    if let Some(source) = ImageSource::from_data_url(&url) {
                                        imgs.push(source);
                                    } else {
                                        warn!("Unsupported image url source");
                                    }
                                }
                                ImageSource::File { .. } => {
                                    imgs.push(source);
                                }
                            }
                            None
                        }
                        ContentBlock::ImageUrl { image_url } => {
                            // oai image
                            if let Some(source) = ImageSource::from_data_url(&image_url.url) {
                                imgs.push(source);
                            }
                            None
                        }
                        ContentBlock::Document { source, title, .. } => {
                            let file_name = extract_document_file_name(&source, title.as_deref());
                            if let Some(text) = extract_document_text(&source) {
                                attachments.push(match file_name {
                                    Some(file_name) => {
                                        Attachment::new_with_file_name(text, file_name)
                                    }
                                    None => Attachment::new(text),
                                });
                            } else if let Some(file_id) = extract_file_id(&source) {
                                imgs.push(ImageSource::File { file_id });
                            } else if let Some((media_type, data)) = extract_base64_file(&source) {
                                imgs.push(ImageSource::Base64 {
                                    media_type,
                                    data,
                                    file_name,
                                });
                            } else {
                                debug!("Unsupported document source for Claude Web request");
                            }
                            None
                        }
                        ContentBlock::ContainerUpload { file_id, .. } => {
                            imgs.push(ImageSource::File { file_id });
                            None
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if blocks.is_empty() {
                    None
                } else {
                    Some((m.role, blocks))
                }
            }
            MessageContent::Text { content } => {
                // plain text
                let content = content.trim().to_string();
                if content.is_empty() {
                    None
                } else {
                    Some((m.role, content))
                }
            }
        })
        // chunk by role
        .chunk_by(|m| m.0);
    // join same role with new line
    let mut msgs = chunks
        .into_iter()
        .map(|(role, grp)| {
            let txt = grp.into_iter().map(|m| m.1).collect::<Vec<_>>().join("\n");
            (role, txt)
        })
        .collect::<Vec<_>>();

    if !system.is_empty() {
        w += system.as_str();
        prompt_parts.push((Role::System, system.clone()));
    } else if let Some((_, first_text)) = msgs.first() {
        w += first_text.as_str();
        prompt_parts.push((Role::User, first_text.clone()));
    }
    for (idx, (role, text)) in msgs.drain(..).enumerate() {
        if system.is_empty() && idx == 0 {
            continue;
        }
        let prefix = match role {
            Role::System => {
                warn!("System message should be merged into the first message");
                continue;
            }
            Role::User => format!("{h}: "),
            Role::Assistant => format!("{a}: "),
        };
        write!(w, "{line_breaks}{prefix}{text}").ok()?;
        prompt_parts.push((role, text));
    }
    if !w.is_empty() {
        print_out_text(w.to_owned(), "paste.txt");
    }

    let mut prompt = prompt_parts
        .into_iter()
        .map(|(role, text)| match role {
            Role::System => text,
            Role::User => text,
            Role::Assistant => format!("{a}: {text}"),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
        .trim()
        .to_string();
    if prompt.is_empty() {
        prompt = CLEWDR_CONFIG.load().custom_prompt.to_owned();
    }
    if prompt.is_empty() && (!attachments.is_empty() || !imgs.is_empty()) {
        prompt = "Please answer using the attached content.".to_string();
    }

    Some(Merged {
        attachments,
        prompt,
        images: imgs,
    })
}

fn default_upload_file_name(media_type: &str) -> &'static str {
    match media_type.to_lowercase().as_str() {
        "image/png" => "image.png",
        "image/jpeg" => "image.jpg",
        "image/jpg" => "image.jpg",
        "image/gif" => "image.gif",
        "image/webp" => "image.webp",
        "application/pdf" => "document.pdf",
        _ => "file",
    }
}

pub(super) fn extract_document_file_name(source: &Value, title: Option<&str>) -> Option<String> {
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

pub(super) fn extract_document_text(source: &Value) -> Option<String> {
    let source_type = source.get("type").and_then(Value::as_str)?;
    if source_type != "text" {
        return None;
    }
    let text = source
        .get("data")
        .or_else(|| source.get("text"))
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

pub(super) fn extract_file_id(source: &Value) -> Option<String> {
    let source_type = source.get("type").and_then(Value::as_str)?;
    if source_type != "file" {
        return None;
    }
    source
        .get("file_id")
        .or_else(|| source.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

pub(super) fn extract_base64_file(source: &Value) -> Option<(String, String)> {
    let source_type = source.get("type").and_then(Value::as_str)?;
    if source_type != "base64" {
        return None;
    }
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{Json, Router, extract::Multipart, routing::post};
    use bytes::Bytes;
    use futures::stream;
    use serde_json::json;
    use url::Url;

    use super::*;
    use crate::{
        claude_web_state::conversation_cache::ConversationCache,
        protocol::{
            AuthPrincipal,
            files::StagedFileStore,
            sessions::{PendingTurn, ProtocolSessionStore},
        },
        services::cookie_actor::CookieActorHandle,
        types::claude::{ContentBlock, CreateMessageParams, Message, MessageContent, Role},
    };

    async fn mock_upload(
        axum::extract::State(count): axum::extract::State<Arc<AtomicUsize>>,
        mut multipart: Multipart,
    ) -> Json<serde_json::Value> {
        let field = multipart.next_field().await.unwrap().unwrap();
        assert_eq!(field.name(), Some("file"));
        assert_eq!(field.file_name(), Some("report.txt"));
        assert_eq!(field.content_type(), Some("text/plain"));
        assert_eq!(
            field.bytes().await.unwrap(),
            Bytes::from_static(b"contents")
        );
        let index = count.fetch_add(1, Ordering::SeqCst) + 1;
        Json(json!({ "file_uuid": format!("claude-file-{index}") }))
    }

    #[tokio::test]
    async fn merge_messages_preserves_text_document_title_as_attachment_file_name() {
        let params = CreateMessageParams {
            max_tokens: 1024,
            model: "claude-sonnet-4-5-20250929".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Blocks {
                    content: vec![ContentBlock::Document {
                        source: json!({
                            "type": "text",
                            "data": "Quarterly notes",
                        }),
                        cache_control: None,
                        citations: None,
                        context: None,
                        title: Some("quarterly-notes.md".to_string()),
                    }],
                },
            }],
            ..Default::default()
        };

        let merged = merge_messages(params.messages, String::new()).expect("message should merge");

        assert_eq!(merged.attachments.len(), 1);
        let attachment = serde_json::to_value(&merged.attachments[0]).unwrap();
        assert_eq!(attachment["file_name"], "quarterly-notes.md");
    }

    #[tokio::test]
    async fn merge_messages_preserves_base64_document_title_for_upload_file_name() {
        let params = CreateMessageParams {
            max_tokens: 1024,
            model: "claude-sonnet-4-5-20250929".to_string(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Blocks {
                    content: vec![ContentBlock::Document {
                        source: json!({
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "JVBERi0xLjQK",
                        }),
                        cache_control: None,
                        citations: None,
                        context: None,
                        title: Some("proposal.pdf".to_string()),
                    }],
                },
            }],
            ..Default::default()
        };

        let merged = merge_messages(params.messages, String::new()).expect("message should merge");

        assert_eq!(merged.images.len(), 1);
        let upload = serde_json::to_value(&merged.images[0]).unwrap();
        assert_eq!(upload["file_name"], "proposal.pdf");
    }

    #[test]
    fn extract_document_file_name_uses_source_file_name_when_title_is_absent() {
        let source = json!({
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQK",
            "file_name": "/tmp/uploaded/report.pdf",
        });

        assert_eq!(
            extract_document_file_name(&source, None).as_deref(),
            Some("report.pdf")
        );
    }

    #[test]
    fn upload_endpoint_matches_claude_web_conversation_upload() {
        let endpoint = Url::parse("https://claude.ai/").unwrap();

        let upload_endpoint =
            conversation_upload_endpoint(&endpoint, "organization-id", "conversation-id").unwrap();

        assert_eq!(
            upload_endpoint.as_str(),
            "https://claude.ai/api/organizations/organization-id/conversations/\
conversation-id/wiggle/upload-file"
        );
    }

    #[tokio::test]
    async fn staged_file_mapping_is_reused_within_one_session_and_isolated_across_sessions() {
        let count = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route(
                "/api/organizations/{org}/conversations/{conversation}/wiggle/upload-file",
                post(mock_upload),
            )
            .with_state(count.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent(temp.path().join("files"))
            .await
            .unwrap();
        let principal = AuthPrincipal::for_authenticated_user();
        let staged = files
            .stage_stream(
                &principal,
                "report.txt",
                "text/plain",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"contents"))]),
            )
            .await
            .unwrap();
        let sessions = ProtocolSessionStore::memory();
        let session_digest = "ab".repeat(32);
        let operation = sessions
            .try_begin(&principal, &session_digest)
            .await
            .unwrap();
        sessions
            .create_provisional(
                &operation,
                &principal,
                &session_digest,
                "cookie".into(),
                "org".into(),
                "conversation".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["user".into()],
                    assistant_uuid_after: "assistant".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();

        let handle = CookieActorHandle::start().await.unwrap();
        let mut state = ClaudeWebState::new(handle, ConversationCache::new());
        state.endpoint = Url::parse(&format!("http://{address}/")).unwrap();
        state.principal = Some(principal);
        state.staged_files = Some(files);
        state.protocol_sessions = Some(sessions);
        let first = state
            .upload_protocol_files(
                vec![ImageSource::File {
                    file_id: staged.id.clone(),
                }],
                "org",
                "conversation",
                &operation,
            )
            .await
            .unwrap();
        let second = state
            .upload_protocol_files(
                vec![ImageSource::File {
                    file_id: staged.id.clone(),
                }],
                "org",
                "conversation",
                &operation,
            )
            .await
            .unwrap();
        assert_eq!(first, vec!["claude-file-1"]);
        assert_eq!(second, first);
        assert_eq!(count.load(Ordering::SeqCst), 1);

        let second_session_digest = "cd".repeat(32);
        let second_operation = state
            .protocol_sessions
            .as_ref()
            .unwrap()
            .try_begin(state.principal.as_ref().unwrap(), &second_session_digest)
            .await
            .unwrap();
        state
            .protocol_sessions
            .as_ref()
            .unwrap()
            .create_provisional(
                &second_operation,
                state.principal.as_ref().unwrap(),
                &second_session_digest,
                "cookie".into(),
                "org".into(),
                "other-conversation".into(),
                "model".into(),
                "system".into(),
                PendingTurn {
                    parent_uuid_before: None,
                    user_digests: vec!["user".into()],
                    assistant_uuid_after: "assistant".into(),
                    replace_from_turn: 0,
                    assistant_digests_before: vec![],
                },
            )
            .await
            .unwrap();
        let third = state
            .upload_protocol_files(
                vec![ImageSource::File { file_id: staged.id }],
                "org",
                "other-conversation",
                &second_operation,
            )
            .await
            .unwrap();
        assert_eq!(third, vec!["claude-file-2"]);
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}

/// Merges system message content into a single string
/// Handles both string and array formats for system messages
///
/// # Arguments
/// * `sys` - System message content as a JSON Value
///
/// # Returns
/// Merged system message as a string
fn merge_system(sys: Value) -> String {
    match sys {
        Value::String(s) => s,
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v["text"].as_str())
            .map(|v| v.trim())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
