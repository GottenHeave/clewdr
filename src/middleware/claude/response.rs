use async_stream::try_stream;
use axum::{
    Json,
    body::{self, Body},
    response::{IntoResponse, Response, Sse, sse::Event},
};
use eventsource_stream::{Event as SourceEvent, Eventsource};
use futures::Stream;
use http::header::CONTENT_TYPE;
use tracing::warn;

use super::{ClaudeApiFormat, transform_stream};
use crate::{
    middleware::claude::{
        ClaudeContext, normalize_claude_web_stream_event, thinking_summary_delta_index,
        transforms_json,
    },
    types::claude::{ContentBlock, CreateMessageResponse, StreamEvent},
};

type EventResult<T> = Result<T, eventsource_stream::EventStreamError<axum::Error>>;

async fn parse_response<T>(resp: Response) -> Result<T, Response>
where
    T: serde::de::DeserializeOwned,
{
    let body = body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .inspect_err(|err| {
            warn!("Failed to read response body: {}", err);
        })
        .unwrap_or_default();
    let Ok(parsed) = serde_json::from_slice::<T>(&body) else {
        return Err(Response::builder()
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap());
    };
    Ok(parsed)
}

fn source_event_template(event: &SourceEvent) -> Event {
    let new_event = Event::default()
        .event(event.event.clone())
        .id(event.id.clone());
    if let Some(retry) = event.retry {
        new_event.retry(retry)
    } else {
        new_event
    }
}

fn thinking_block_start(index: usize) -> Event {
    Event::default()
        .json_data(StreamEvent::ContentBlockStart {
            index,
            content_block: ContentBlock::Thinking {
                signature: String::new(),
                thinking: String::new(),
            },
        })
        .unwrap()
}

fn normalize_stream(
    usage: crate::types::claude::Usage,
    stream: impl Stream<Item = EventResult<SourceEvent>>,
) -> impl Stream<Item = EventResult<Event>> {
    try_stream!({
        let mut started_thinking_indexes = std::collections::HashSet::new();
        for await event in stream {
            let event = event?;
            let summary_delta_index = thinking_summary_delta_index(&event.data);
            let new_event = source_event_template(&event);
            let Some(data) = normalize_claude_web_stream_event(&event.data) else {
                continue;
            };
            let Ok(parsed) = serde_json::from_str::<StreamEvent>(&data) else {
                yield new_event.data(data);
                continue;
            };
            if let StreamEvent::ContentBlockStart {
                index,
                content_block: ContentBlock::Thinking { .. },
            } = &parsed
            {
                started_thinking_indexes.insert(*index);
            }
            if let Some(index) = summary_delta_index {
                if started_thinking_indexes.insert(index) {
                    yield thinking_block_start(index);
                }
            }
            let event = match parsed {
                StreamEvent::MessageStart { mut message } => {
                    message.usage.get_or_insert(usage.to_owned());
                    new_event
                        .json_data(StreamEvent::MessageStart { message })
                        .unwrap()
                }
                StreamEvent::MessageDelta { delta, usage } => {
                    let usage = usage.unwrap_or_default();
                    new_event
                        .json_data(StreamEvent::MessageDelta {
                            delta,
                            usage: Some(usage),
                        })
                        .unwrap()
                }
                _ => new_event.data(data),
            };
            yield event;
        }
    })
}

/// Transforms responses to ensure compatibility with the OpenAI API format
///
/// This middleware function analyzes responses and transforms them when necessary
/// to ensure compatibility between Claude and OpenAI API formats, particularly
/// for streaming responses. If the response is:
///
/// - From the Claude API format: No transformation needed
/// - Not streaming: No transformation needed
/// - Has a non-200 status code: No transformation needed
/// - OpenAI format and streaming: Transforms the stream to match OpenAI event format
///
/// # Arguments
///
/// * `resp` - The original response to be potentially transformed
///
/// # Returns
///
/// The original or transformed response as appropriate
pub async fn to_oai(resp: Response) -> impl IntoResponse {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };
    if ClaudeApiFormat::Claude == cx.api_format() {
        return resp;
    }
    if !cx.is_stream() {
        match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => return Json(transforms_json(response)).into_response(),
            Err(resp) => return resp,
        }
    }
    let stream = resp.into_body().into_data_stream().eventsource();
    let stream = transform_stream(stream);
    Sse::new(stream)
        .keep_alive(Default::default())
        .into_response()
}

pub async fn add_usage_info(resp: Response) -> impl IntoResponse {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };
    let (mut usage, stream) = (cx.usage().to_owned(), cx.is_stream());
    if !stream {
        let mut response = match parse_response::<CreateMessageResponse>(resp).await {
            Ok(response) => response,
            Err(resp) => return resp,
        };
        let output_tokens = response.count_tokens();
        usage.output_tokens = output_tokens;
        response.usage = Some(usage);
        return Json(response).into_response();
    }
    let stream = resp.into_body().into_data_stream().eventsource();
    let stream = normalize_stream(usage, stream);

    Sse::new(stream)
        .keep_alive(Default::default())
        .into_response()
}

pub async fn check_overloaded(mut resp: Response) -> Response {
    let Some(cx) = resp.extensions().get::<ClaudeContext>() else {
        return resp;
    };
    if !cx.is_stream() {
        return resp;
    }
    if resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| !v.contains("text/event-stream"))
    {
        resp.extensions_mut().remove::<ClaudeContext>();
    }
    resp
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::middleware::claude::normalize_claude_web_stream_event;

    #[test]
    fn drops_web_only_message_limit_events() {
        let data = r#"{"type":"message_limit","message_limit":{"type":"within_limit"}}"#;

        assert!(normalize_claude_web_stream_event(data).is_none());
    }

    #[test]
    fn rewrites_thinking_summary_delta_events() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_summary_delta","summary":{"summary":"确认端口已改，更新草稿回复。"}}}"#;
        let normalized = normalize_claude_web_stream_event(data).unwrap();
        let normalized: Value = serde_json::from_str(&normalized).unwrap();

        assert_eq!(
            normalized
                .get("delta")
                .and_then(|delta| delta.get("type"))
                .and_then(Value::as_str),
            Some("thinking_delta")
        );
        assert_eq!(
            normalized
                .get("delta")
                .and_then(|delta| delta.get("thinking"))
                .and_then(Value::as_str),
            Some("确认端口已改，更新草稿回复。")
        );
    }

    #[test]
    fn keeps_regular_content_deltas() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

        assert_eq!(
            normalize_claude_web_stream_event(data).as_deref(),
            Some(data)
        );
    }
}
