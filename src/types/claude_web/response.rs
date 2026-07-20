use async_stream::try_stream;
use axum::{
    BoxError, Json,
    response::{IntoResponse, Sse, sse::Event as SseEvent},
};
use bytes::Bytes;
use eventsource_stream::{EventStream, Eventsource};
use futures::{Stream, TryStreamExt};
use serde::Deserialize;
use url::Url;
use wreq::Proxy;

use crate::{
    claude_code_state::ClaudeCodeState,
    claude_web_state::ClaudeWebState,
    claude_web_state::explicit_session::digest_assistant_output,
    error::{CheckClaudeErr, ClewdrError},
    types::claude::{
        ContentBlock, CountMessageTokensResponse, CreateMessageParams, CreateMessageResponse,
        Message, Role,
    },
    utils::print_out_text,
};

pub async fn merge_sse(
    stream: EventStream<impl Stream<Item = Result<Bytes, wreq::Error>>>,
) -> Result<(String, bool), ClewdrError> {
    // Collect all SSE events so completion text can be merged and message_stop can
    // decide whether an explicit lifecycle may commit the turn.
    #[derive(Deserialize)]
    struct Data {
        completion: String,
    }
    let events = stream.try_collect::<Vec<_>>().await?;
    let saw_message_stop = events
        .iter()
        .any(|event| is_message_stop(&event.event, &event.data));
    let text = events
        .iter()
        .filter_map(|event| serde_json::from_str::<Data>(&event.data).ok())
        .map(|data| data.completion)
        .collect();
    Ok((text, saw_message_stop))
}

fn is_message_stop(event: &str, data: &str) -> bool {
    event == "message_stop"
        || serde_json::from_str::<serde_json::Value>(data)
            .ok()
            .and_then(|value| {
                value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .is_some_and(|kind| kind == "message_stop")
}

impl<S> From<S> for Message
where
    S: Into<String>,
{
    fn from(str: S) -> Self {
        Message::new_blocks(Role::Assistant, vec![ContentBlock::text(str.into())])
    }
}

impl ClaudeWebState {
    /// Converts Claude Web SSE into the requested API format while accounting usage and
    /// finalizing explicit session state only after a complete stream.
    pub async fn transform_response(
        &mut self,
        wreq_res: wreq::Response,
    ) -> Result<axum::response::Response, ClewdrError> {
        let explicit_lifecycle = self.explicit_lifecycle.take();
        if self.stream {
            let mut input_tokens = self.usage.input_tokens as u64;
            let handle = self.cookie_actor_handle.clone();
            let cookie = self.cookie.clone();
            let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
            let last_params = self.last_params.clone();
            let endpoint = self.endpoint.clone();
            let proxy = self.proxy.clone();
            let client = self.client.clone();
            if crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens
                && let Some(tokens) = self.try_code_count_tokens().await
            {
                input_tokens = tokens as u64;
            }

            let stream = wreq_res
                .bytes_stream()
                .eventsource()
                .map_err(axum::Error::new);
            let stream = try_stream! {
                // Accumulate completion deltas for output-token accounting and assistant digest.
                let lifecycle = explicit_lifecycle;
                let mut explicit_finalized = false;
                let mut acc = String::new();
                #[derive(serde::Deserialize)]
                struct Data { completion: String }
                futures::pin_mut!(stream);
                while let Some(event) = stream.try_next().await? {
                    // message_stop is the commit boundary; EOF or downstream drop is uncertain.
                    if is_message_stop(&event.event, &event.data)
                        && let Some(lifecycle) = &lifecycle
                    {
                        let digest = (!acc.is_empty()).then(|| digest_assistant_output(&acc));
                        lifecycle.commit(digest).await.map_err(axum::Error::new)?;
                        explicit_finalized = true;
                    }
                    // Forward every event while accumulating only completion payloads.
                    if let Ok(d) = serde_json::from_str::<Data>(&event.data) {
                        acc.push_str(&d.completion);
                    }
                    let e = SseEvent::default().event(event.event).id(event.id);
                    let e = if let Some(retry) = event.retry { e.retry(retry) } else { e };
                    yield e.data(event.data);
                }
                if let Some(lifecycle) = &lifecycle
                    && !explicit_finalized
                {
                    lifecycle.uncertain().await.map_err(axum::Error::new)?;
                    Err(axum::Error::new(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "Claude Web stream ended without message_stop",
                    )))?;
                }
                if !acc.is_empty() {
                    let mut out = None;
                    // Prefer Claude Code token counting, then fall back to local response counting.
                    if enable_precise
                        && let Some(model) = last_params.as_ref().map(|p| p.model.clone())
                    {
                        out = count_code_output_tokens_for_text(
                            cookie.clone(), endpoint.clone(), proxy.clone(), client.clone(),
                            model, acc.clone(), handle.clone()
                        ).await.map(|v| v as u64);
                    }
                    let out = out.unwrap_or_else(|| {
                        let usage = crate::types::claude::Usage { input_tokens: input_tokens as u32, output_tokens: 0 };
                        let resp = crate::types::claude::CreateMessageResponse::text(acc.clone(), Default::default(), usage);
                        resp.count_tokens() as u64
                    });
                    if let Some(mut c) = cookie.clone() {
                        let family = last_params
                            .as_ref()
                            .map(|p| p.model.as_str())
                            .map(|m| {
                                let m = m.to_ascii_lowercase();
                                if m.contains("opus") {
                                    crate::config::ModelFamily::Opus
                                } else if m.contains("sonnet") {
                                    crate::config::ModelFamily::Sonnet
                                } else {
                                    crate::config::ModelFamily::Other
                                }
                            })
                            .unwrap_or(crate::config::ModelFamily::Other);
                        c.add_and_bucket_usage(input_tokens, out, family);
                        let _ = handle.return_cookie(c, None).await;
                    }
                } else if let Some(mut c) = cookie.clone() {
                    let family = last_params
                        .as_ref()
                        .map(|p| p.model.as_str())
                        .map(|m| {
                            let m = m.to_ascii_lowercase();
                            if m.contains("opus") {
                                crate::config::ModelFamily::Opus
                            } else if m.contains("sonnet") {
                                crate::config::ModelFamily::Sonnet
                            } else {
                                crate::config::ModelFamily::Other
                            }
                        })
                        .unwrap_or(crate::config::ModelFamily::Other);
                    c.add_and_bucket_usage(input_tokens, 0, family);
                    let _ = handle.return_cookie(c, None).await;
                }
            };
            let stream = stream.map_err(|e: axum::Error| -> BoxError { e.into() });
            return Ok(Sse::new(stream)
                .keep_alive(Default::default())
                .into_response());
        }

        let stream = wreq_res.bytes_stream();
        let stream = stream.eventsource();
        let (text, saw_message_stop) = match merge_sse(stream).await {
            Ok(result) => result,
            Err(error) => {
                if let Some(lifecycle) = explicit_lifecycle {
                    lifecycle.uncertain().await?;
                }
                return Err(error);
            }
        };
        if let Some(lifecycle) = explicit_lifecycle {
            if saw_message_stop {
                let digest = (!text.is_empty()).then(|| digest_assistant_output(&text));
                lifecycle.commit(digest).await?;
            } else {
                lifecycle.uncertain().await?;
                return Err(crate::protocol::ProtocolError::new(
                    http::StatusCode::BAD_GATEWAY,
                    "conversation_state_uncertain",
                    "Claude Web response ended without message_stop",
                )
                .into());
            }
        }

        print_out_text(text.to_owned(), "claude_web_non_stream.txt");
        let mut response =
            CreateMessageResponse::text(text.clone(), Default::default(), self.usage.to_owned());

        let enable_precise = crate::config::CLEWDR_CONFIG.load().enable_web_count_tokens;
        let mut usage = self.usage.to_owned();
        if enable_precise && let Some(inp) = self.try_code_count_tokens().await {
            usage.input_tokens = inp;
        }
        let mut output_tokens = response.count_tokens();
        if enable_precise && let Some(model) = self.last_params.as_ref().map(|p| p.model.clone()) {
            let out = count_code_output_tokens_for_text(
                self.cookie.clone(),
                self.endpoint.clone(),
                self.proxy.clone(),
                self.client.clone(),
                model,
                text.clone(),
                self.cookie_actor_handle.clone(),
            )
            .await;
            if let Some(v) = out {
                output_tokens = v;
            }
        }
        usage.output_tokens = output_tokens;
        response.usage = Some(usage.clone());
        self.persist_usage_totals(usage.input_tokens as u64, output_tokens as u64)
            .await;
        Ok(Json(response).into_response())
    }
}

async fn bearer_count_tokens(
    state: &ClaudeCodeState,
    access_token: &str,
    body: &CreateMessageParams,
) -> Option<u32> {
    let url = state.endpoint.join("v1/messages/count_tokens").ok()?;
    let resp = state
        .client
        .post(url.to_string())
        .bearer_auth(access_token)
        .header("anthropic-version", "2023-06-01")
        .json(body)
        .send()
        .await
        .ok()?;
    let resp = resp.check_claude().await.ok()?;
    let v: CountMessageTokensResponse = resp.json().await.ok()?;
    Some(v.input_tokens)
}

impl ClaudeWebState {
    pub(crate) async fn try_code_count_tokens(&mut self) -> Option<u32> {
        self.cookie.as_ref()?;
        let params = self.last_params.as_ref()?.clone();
        let mut code = ClaudeCodeState::new(self.cookie_actor_handle.clone());
        code.cookie = self.cookie.clone();
        code.endpoint = self.endpoint.clone();
        code.proxy = self.proxy.clone();
        code.client = self.client.clone();
        if let Some(ref c) = self.cookie
            && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
        {
            code.set_cookie_header_value(val);
        }

        let org = code.get_organization().await.ok()?;
        let exch = code.exchange_code(&org).await.ok()?;
        code.exchange_token(exch).await.ok()?;
        let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

        let mut body = params.clone();
        body.stream = Some(false);

        bearer_count_tokens(&code, &access, &body).await
    }
}

async fn count_code_output_tokens_for_text(
    cookie: Option<crate::config::CookieStatus>,
    endpoint: Url,
    proxy: Option<Proxy>,
    client: wreq::Client,
    model: String,
    text: String,
    handle: crate::services::cookie_actor::CookieActorHandle,
) -> Option<u32> {
    let mut code = ClaudeCodeState::new(handle.clone());
    code.cookie = cookie.clone();
    code.endpoint = endpoint;
    code.proxy = proxy;
    code.client = client;
    if let Some(ref c) = cookie
        && let Ok(val) = http::HeaderValue::from_str(&c.cookie.to_string())
    {
        code.set_cookie_header_value(val);
    }
    let org = code.get_organization().await.ok()?;
    let exch = code.exchange_code(&org).await.ok()?;
    code.exchange_token(exch).await.ok()?;
    let access = code.cookie.as_ref()?.token.as_ref()?.access_token.clone();

    let body = CreateMessageParams {
        model,
        messages: vec![Message::new_text(Role::Assistant, text)],
        ..Default::default()
    };
    // do not set count_tokens_allowed flag here to avoid races; handled by try_code_count_tokens
    bearer_count_tokens(&code, &access, &body).await
}

#[cfg(test)]
mod explicit_session_tests {
    use axum::{
        Router, body, body::Body, http::header::CONTENT_TYPE, response::Response, routing::get,
    };

    use crate::{
        claude_web_state::{
            ClaudeWebState,
            conversation_cache::{
                ConversationCache, ExplicitSessionKey, explicit_test_conversation,
            },
            explicit_session::{ExplicitLifecycle, ExplicitSessionState, PendingExplicitTurn},
        },
        services::cookie_actor::CookieActorHandle,
    };

    async fn healthy_stream() -> Response {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from(
                "data: {\"type\":\"message_start\"}\n\ndata: {\"type\":\"message_stop\"}\n\n",
            ))
            .unwrap()
    }

    async fn incomplete_stream() -> Response {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from("data: {\"type\":\"message_start\"}\n\n"))
            .unwrap()
    }

    async fn setup_lifecycle() -> (ConversationCache, ExplicitSessionKey, ExplicitLifecycle) {
        let cache = ConversationCache::new();
        let key = ExplicitSessionKey::new("principal", "ab".repeat(32));
        let operation = cache.lock_explicit_operation(&key).await.into_guard();
        let mut conversation = explicit_test_conversation(ExplicitSessionState::InFlight);
        conversation.explicit.as_mut().unwrap().pending = Some(PendingExplicitTurn {
            model: None,
            model_digest: None,
            parent_uuid_before: None,
            user_digests: vec!["user".into()],
            assistant_uuid_after: "assistant".into(),
            replace_from_turn: 0,
            parent_timeline: Vec::new(),
            request_timeline: vec!["user:user".into()],
        });
        cache
            .set_explicit_checked(key.clone(), conversation)
            .await
            .unwrap();
        let lifecycle = ExplicitLifecycle::new(cache.clone(), key.clone(), operation);
        (cache, key, lifecycle)
    }

    async fn upstream_response(path: &'static str) -> wreq::Response {
        let app = Router::new()
            .route("/healthy", get(healthy_stream))
            .route("/incomplete", get(incomplete_stream));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        wreq::Client::new()
            .get(format!("http://{address}/{path}"))
            .send()
            .await
            .unwrap()
    }

    async fn transform(lifecycle: ExplicitLifecycle, path: &'static str) -> Response {
        let handle = CookieActorHandle::start().await.unwrap();
        let mut state = ClaudeWebState::new(handle, ConversationCache::new());
        state.stream = true;
        state.explicit_lifecycle = Some(lifecycle);
        state
            .transform_response(upstream_response(path).await)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn response_consumption_controls_explicit_lifecycle() {
        for (path, consume, expected) in [
            ("healthy", true, ExplicitSessionState::Committed),
            ("incomplete", true, ExplicitSessionState::Uncertain),
            ("healthy", false, ExplicitSessionState::Uncertain),
        ] {
            let (cache, key, lifecycle) = setup_lifecycle().await;
            let response = transform(lifecycle, path).await;
            if consume {
                let result = body::to_bytes(response.into_body(), usize::MAX).await;
                assert_eq!(result.is_ok(), path == "healthy");
            } else {
                drop(response);
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
            let explicit = cache.get_explicit(&key).await.unwrap().explicit.unwrap();
            assert_eq!(explicit.state, expected);
            if expected == ExplicitSessionState::Committed {
                assert_eq!(explicit.turns[0].assistant_uuid_after, "assistant");
            }
        }
    }
}
