use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::{FromRequest, Multipart, Request, State},
    http::StatusCode,
};
use futures::stream;
use serde::{Deserialize, Serialize};

use crate::{
    claude_web_state::conversation_cache::{ConversationCache, ExplicitSessionKey},
    protocol::{AuthPrincipal, ProtocolError, parse_session_id},
    protocol_files::{FileResponse, StagedFileStore},
};

#[derive(Clone)]
pub(crate) struct ResetApiState {
    pub cache: ConversationCache,
    pub files: Option<Arc<StagedFileStore>>,
}

pub(crate) struct ProtocolMultipart(Multipart);

impl<S> FromRequest<S> for ProtocolMultipart
where
    S: Send + Sync,
{
    type Rejection = ProtocolError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Multipart::from_request(request, state)
            .await
            .map(Self)
            .map_err(invalid_multipart)
    }
}

pub(crate) async fn api_stage_file(
    State(store): State<Option<Arc<StagedFileStore>>>,
    Extension(principal): Extension<AuthPrincipal>,
    ProtocolMultipart(mut multipart): ProtocolMultipart,
) -> Result<Json<FileResponse>, ProtocolError> {
    let store = store.ok_or_else(|| {
        ProtocolError::new(
            StatusCode::NOT_IMPLEMENTED,
            "staged_files_unavailable",
            "Staged files are unavailable when filesystem persistence is disabled",
        )
    })?;
    let field = multipart.next_field().await.map_err(invalid_multipart)?;
    let Some(field) = field else {
        return Err(invalid_multipart(
            "Multipart body must contain one file field",
        ));
    };
    if field.name() != Some("file") {
        return Err(invalid_multipart("Multipart field must be named 'file'"));
    }
    let filename = field
        .file_name()
        .ok_or_else(|| invalid_multipart("Multipart file must include a filename"))?
        .to_owned();
    let mime_type = field
        .content_type()
        .ok_or_else(|| invalid_multipart("Multipart file must include a MIME type"))?
        .to_owned();
    let chunks = stream::try_unfold(field, |mut field| async move {
        match field.chunk().await {
            Ok(Some(chunk)) => Ok(Some((chunk, field))),
            Ok(None) => Ok(None),
            Err(error) => Err(error),
        }
    });
    let upload = store
        .stage_stream_with_status(&principal, &filename, &mime_type, chunks)
        .await?;
    match multipart.next_field().await {
        Ok(None) => Ok(Json(upload.into_response())),
        Ok(Some(_)) => {
            upload.rollback(&store).await?;
            Err(invalid_multipart(
                "Multipart body must contain exactly one field",
            ))
        }
        Err(error) => {
            upload.rollback(&store).await?;
            Err(invalid_multipart(error))
        }
    }
}

fn invalid_multipart(error: impl std::fmt::Display) -> ProtocolError {
    ProtocolError::new(
        StatusCode::BAD_REQUEST,
        "invalid_multipart",
        error.to_string(),
    )
}

#[derive(Deserialize)]
pub(crate) struct ResetSessionRequest {
    session_id: String,
}

#[derive(Serialize)]
pub(crate) struct ResetSessionResponse {
    r#type: &'static str,
    session_id: String,
}

pub(crate) async fn api_reset_session(
    State(state): State<ResetApiState>,
    Extension(principal): Extension<AuthPrincipal>,
    Json(request): Json<ResetSessionRequest>,
) -> Result<Json<ResetSessionResponse>, ProtocolError> {
    let digest = parse_session_id(Some(&request.session_id))
        .map_err(|error| {
            ProtocolError::new(
                StatusCode::BAD_REQUEST,
                "invalid_session_id",
                error.to_string(),
            )
        })?
        .ok_or_else(|| {
            ProtocolError::new(
                StatusCode::BAD_REQUEST,
                "invalid_session_id",
                "Reset requires a cherry_topic_v1 session ID",
            )
        })?;
    let key = ExplicitSessionKey::new(principal.as_str(), digest);
    let _operation = state.cache.try_lock_explicit_operation(&key).await?;
    if !state.cache.reset_explicit(&key).await? {
        return Err(ProtocolError::new(
            StatusCode::NOT_FOUND,
            "session_not_found",
            "Session does not exist",
        ));
    }
    if let Some(files) = state.files {
        files.remove_session_references(&key.session_ref()).await?;
    }
    Ok(Json(ResetSessionResponse {
        r#type: "session_reset",
        session_id: request.session_id,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use axum::{
        Extension, Router, body,
        body::Body,
        http::{Request, header::CONTENT_TYPE},
        routing::post,
    };
    use bytes::Bytes;
    use futures::stream;
    use tower::ServiceExt;

    use super::*;
    use crate::claude_web_state::{
        conversation_cache::CachedConversation,
        explicit_session::{ExplicitConversation, ExplicitReusePlan, ExplicitSessionState, plan},
    };

    fn multipart_body(boundary: &str, fields: &[(&str, &str, &str, &[u8])]) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, filename, mime, bytes) in fields {
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
                )
                .as_bytes(),
            );
            body.extend_from_slice(bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        body
    }

    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn files_app(store: Option<Arc<StagedFileStore>>) -> Router {
        Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(store)
            .layer(Extension(AuthPrincipal::for_authenticated_user()))
    }

    async fn post_files(
        app: Router,
        fields: &[(&str, &str, &str, &[u8])],
    ) -> axum::response::Response {
        let boundary = "clewdr-boundary";
        app.oneshot(
            Request::post("/v1/files")
                .header(
                    CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(multipart_body(boundary, fields)))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn files_endpoint_accepts_one_file_and_returns_anthropic_shape() {
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent(temp.path()).await.unwrap();
        let response = post_files(
            files_app(Some(files)),
            &[("file", "report.txt", "text/plain", b"hello")],
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;
        assert_eq!(json["type"], "file");
        assert_eq!(json["filename"], "report.txt");
        assert_eq!(json["mime_type"], "text/plain");
        assert_eq!(json["size_bytes"], 5);
        assert!(json["id"].as_str().unwrap().starts_with("file_clewdr_v1_"));
    }

    #[tokio::test]
    async fn files_endpoint_rolls_back_when_a_second_field_exists() {
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent(temp.path()).await.unwrap();
        let response = post_files(
            files_app(Some(files)),
            &[
                ("file", "report.txt", "text/plain", b"hello"),
                ("extra", "extra.txt", "text/plain", b"extra"),
            ],
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            std::fs::read_dir(temp.path().join("objects"))
                .unwrap()
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn files_endpoint_is_unavailable_without_filesystem_storage() {
        let response = post_files(files_app(None), &[]).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            response_json(response).await["error"]["type"],
            "staged_files_unavailable"
        );
    }

    fn cached_session(state: ExplicitSessionState) -> CachedConversation {
        CachedConversation {
            conv_uuid: "conversation".into(),
            org_uuid: "org".into(),
            cookie_id: "cookie".into(),
            model: "model".into(),
            is_pro: false,
            system_hash: 0,
            turns: Vec::new(),
            created_at: chrono::Utc::now(),
            last_used: chrono::Utc::now(),
            valid: true,
            last_stream_healthy: Arc::new(AtomicBool::new(true)),
            explicit: Some(ExplicitConversation {
                state,
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: Vec::new(),
                pending: None,
                file_mappings: Default::default(),
            }),
        }
    }

    #[tokio::test]
    async fn reset_removes_tombstone_and_allows_recreation() {
        let cache = ConversationCache::new();
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "ab".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        cache
            .set_explicit(
                key.clone(),
                cached_session(ExplicitSessionState::Tombstoned),
            )
            .await;
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent_with_limits(temp.path(), 1, 1)
            .await
            .unwrap();
        let staged = files
            .stage_stream(
                &principal,
                "first.txt",
                "text/plain",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"a"))]),
            )
            .await
            .unwrap();
        files
            .add_reference(&staged.id, &key.session_ref())
            .await
            .unwrap();
        let app = Router::new()
            .route("/v1/sessions/reset", post(api_reset_session))
            .layer(Extension(principal.clone()))
            .with_state(ResetApiState {
                cache: cache.clone(),
                files: Some(files.clone()),
            });
        let response = app
            .oneshot(
                Request::post("/v1/sessions/reset")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(
                        "{{\"session_id\":\"cherry_topic_v1_{digest}\"}}"
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(cache.get_explicit(&key).await.is_none());
        files
            .stage_stream(
                &principal,
                "second.txt",
                "text/plain",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from_static(b"b"))]),
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn concurrent_reset_reports_session_busy() {
        let cache = ConversationCache::new();
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "cd".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let _operation = cache.try_lock_explicit_operation(&key).await.unwrap();
        let error = cache.try_lock_explicit_operation(&key).await.unwrap_err();
        assert_eq!(error.status, StatusCode::CONFLICT);
        assert_eq!(error.code, "session_busy");
    }

    #[tokio::test]
    async fn reset_allows_uncertain_session_to_create_again() {
        let cache = ConversationCache::new();
        let key = ExplicitSessionKey::new("principal", "uncertain");
        cache
            .set_explicit(key.clone(), cached_session(ExplicitSessionState::Uncertain))
            .await;
        assert!(cache.reset_explicit(&key).await.unwrap());
        assert_eq!(
            plan(
                None,
                &["user".into()],
                &["user:user".into()],
                "model",
                "system",
            )
            .unwrap(),
            ExplicitReusePlan::Create
        );
    }
}
