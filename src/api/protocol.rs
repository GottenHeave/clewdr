use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::{FromRequest, Multipart, Request, State},
    http::StatusCode,
};
use futures::stream;
use serde::{Deserialize, Serialize};

use crate::protocol::{
    AuthPrincipal, ProtocolError,
    files::{FileResponse, StagedFileStore},
    parse_session_id,
    sessions::ProtocolSessionStore,
};

#[derive(Clone)]
pub(crate) struct ProtocolApiState {
    pub files: Option<Arc<StagedFileStore>>,
    pub sessions: Arc<ProtocolSessionStore>,
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
    State(state): State<ProtocolApiState>,
    Extension(principal): Extension<AuthPrincipal>,
    ProtocolMultipart(mut multipart): ProtocolMultipart,
) -> Result<Json<FileResponse>, ProtocolError> {
    let store = state.files.ok_or_else(|| {
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

#[derive(Deserialize)]
struct ResetSessionRequest {
    pub session_id: String,
}

pub(crate) struct ResetSessionJson(ResetSessionRequest);

impl<S> FromRequest<S> for ResetSessionJson
where
    S: Send + Sync,
{
    type Rejection = ProtocolError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        Json::<ResetSessionRequest>::from_request(request, state)
            .await
            .map(|Json(request)| Self(request))
            .map_err(|error| {
                ProtocolError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_session_id",
                    format!("Invalid session reset request: {error}"),
                )
            })
    }
}

#[derive(Serialize)]
pub(crate) struct ResetSessionResponse {
    pub r#type: &'static str,
    pub session_id: String,
}

pub(crate) async fn api_reset_session(
    State(state): State<ProtocolApiState>,
    Extension(principal): Extension<AuthPrincipal>,
    ResetSessionJson(request): ResetSessionJson,
) -> Result<Json<ResetSessionResponse>, ProtocolError> {
    let digest = parse_session_id(Some(&request.session_id))?.ok_or_else(|| {
        ProtocolError::new(
            StatusCode::BAD_REQUEST,
            "invalid_session_id",
            "Reset requires a cherry_topic_v1 session ID",
        )
    })?;
    let operation = state.sessions.try_begin(&principal, &digest).await?;
    let removed = state.sessions.reset(&operation, &principal).await?;
    if let Some(files) = state.files {
        files
            .remove_session_references(&removed.session_ref())
            .await?;
    }
    Ok(Json(ResetSessionResponse {
        r#type: "session_reset",
        session_id: request.session_id,
    }))
}

fn invalid_multipart(error: impl std::fmt::Display) -> ProtocolError {
    ProtocolError::new(
        StatusCode::BAD_REQUEST,
        "invalid_multipart",
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use axum::{
        Router, body,
        body::Body,
        http::{Request, header::CONTENT_TYPE},
        routing::post,
    };
    use tower::ServiceExt;

    use super::*;

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

    #[tokio::test]
    async fn files_endpoint_streams_one_named_field_and_returns_anthropic_shape() {
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent(temp.path()).await.unwrap();
        let state = ProtocolApiState {
            files: Some(files),
            sessions: ProtocolSessionStore::memory(),
        };
        let app = Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(state)
            .layer(Extension(AuthPrincipal::for_authenticated_user()));
        let boundary = "clewdr-test-boundary";
        let response = app
            .oneshot(
                Request::post("/v1/files")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(multipart_body(
                        boundary,
                        &[("file", "report.txt", "text/plain", b"hello")],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["type"], "file");
        assert_eq!(body["filename"], "report.txt");
        assert_eq!(body["mime_type"], "text/plain");
        assert_eq!(body["size_bytes"], 5);
        assert!(body["id"].as_str().unwrap().starts_with("file_clewdr_v1_"));
    }

    #[tokio::test]
    async fn files_endpoint_rejects_extra_fields_with_anthropic_error() {
        let temp = tempfile::tempdir().unwrap();
        let state = ProtocolApiState {
            files: Some(StagedFileStore::persistent(temp.path()).await.unwrap()),
            sessions: ProtocolSessionStore::memory(),
        };
        let app = Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(state)
            .layer(Extension(AuthPrincipal::for_authenticated_user()));
        let boundary = "clewdr-test-boundary";
        let response = app
            .oneshot(
                Request::post("/v1/files")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(multipart_body(
                        boundary,
                        &[
                            ("file", "report.txt", "text/plain", b"hello"),
                            ("extra", "extra.txt", "text/plain", b"bad"),
                        ],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_multipart");
        assert_eq!(
            std::fs::read_dir(temp.path().join("objects"))
                .unwrap()
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn no_fs_files_endpoint_returns_explicit_501() {
        let state = ProtocolApiState {
            files: None,
            sessions: ProtocolSessionStore::memory(),
        };
        let app = Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(state)
            .layer(Extension(AuthPrincipal::for_authenticated_user()));
        let boundary = "clewdr-test-boundary";
        let response = app
            .oneshot(
                Request::post("/v1/files")
                    .header(
                        CONTENT_TYPE,
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(multipart_body(
                        boundary,
                        &[("file", "report.txt", "text/plain", b"hello")],
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "staged_files_unavailable");
    }

    #[tokio::test]
    async fn malformed_multipart_uses_anthropic_error_shape() {
        let temp = tempfile::tempdir().unwrap();
        let state = ProtocolApiState {
            files: Some(StagedFileStore::persistent(temp.path()).await.unwrap()),
            sessions: ProtocolSessionStore::memory(),
        };
        let app = Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(state)
            .layer(Extension(AuthPrincipal::for_authenticated_user()));
        let response = app
            .oneshot(
                Request::post("/v1/files")
                    .header(CONTENT_TYPE, "multipart/form-data")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_multipart");
    }
}
