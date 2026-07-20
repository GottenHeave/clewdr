use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::{FromRequest, Multipart, Request, State},
    http::StatusCode,
};
use futures::stream;
use serde::{Deserialize, Serialize};

use crate::{
    claude_web_state::{
        conversation_cache::{ConversationCache, ExplicitSessionKey},
        explicit_session::reset_explicit_session,
    },
    protocol::{AuthPrincipal, ProtocolError, parse_session_id},
    protocol_files::{FileResponse, StagedFileStore},
};

#[derive(Clone)]
pub(crate) struct ResetApiState {
    pub cache: ConversationCache,
    pub files: Option<Arc<StagedFileStore>>,
}

#[derive(Clone)]
pub(crate) struct FileApiState {
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
    State(state): State<FileApiState>,
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
    let _files = state.cache.lock_explicit_files().await;
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
    let _operation = state.cache.lock_explicit_operation(&key).await;
    if state.cache.get_explicit(&key).await.is_none() {
        return Err(ProtocolError::new(
            StatusCode::NOT_FOUND,
            "session_not_found",
            "Session does not exist",
        ));
    }
    match reset_explicit_session(&state.cache, state.files.as_ref(), &key).await {
        Ok(true) => {}
        Ok(false) => {
            return Err(ProtocolError::new(
                StatusCode::NOT_FOUND,
                "session_not_found",
                "Session does not exist",
            ));
        }
        Err(error) => return Err(error),
    }
    Ok(Json(ResetSessionResponse {
        r#type: "session_reset",
        session_id: request.session_id,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

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
        conversation_cache::{explicit_test_conversation, explicit_test_seed},
        explicit_session::ExplicitSessionState,
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

    fn files_app(store: Option<Arc<StagedFileStore>>, cache: ConversationCache) -> Router {
        Router::new()
            .route("/v1/files", post(api_stage_file))
            .with_state(FileApiState {
                cache,
                files: store,
            })
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
    async fn files_endpoint_covers_success_rollback_and_no_fs() {
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent(temp.path()).await.unwrap();
        let response = post_files(
            files_app(Some(files.clone()), ConversationCache::new()),
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
        let response = post_files(
            files_app(Some(files.clone()), ConversationCache::new()),
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
            1
        );
        let response = post_files(files_app(None, ConversationCache::new()), &[]).await;
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            response_json(response).await["error"]["type"],
            "staged_files_unavailable"
        );
    }

    async fn stage(
        files: &StagedFileStore,
        principal: &AuthPrincipal,
        name: &str,
        byte: u8,
    ) -> Result<String, ProtocolError> {
        files
            .stage_stream(
                principal,
                name,
                "text/plain",
                stream::iter([Ok::<_, std::io::Error>(Bytes::from(vec![byte]))]),
            )
            .await
            .map(|file| file.id)
    }

    fn reset_app(state: ResetApiState, principal: AuthPrincipal) -> Router {
        Router::new()
            .route("/v1/sessions/reset", post(api_reset_session))
            .layer(Extension(principal))
            .with_state(state)
    }

    fn reset_request(digest: &str) -> Request<Body> {
        Request::post("/v1/sessions/reset")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                "{{\"session_id\":\"cherry_topic_v1_{digest}\"}}"
            )))
            .unwrap()
    }

    fn reconcile_task(
        cache: ConversationCache,
        files: Arc<StagedFileStore>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let _guard = cache.lock_explicit_files().await;
            files
                .reconcile_references(&cache.explicit_file_references().await)
                .await
                .unwrap();
        })
    }

    #[tokio::test]
    async fn failed_reset_restores_file_references_and_remains_retryable() {
        let principal = AuthPrincipal::for_authenticated_user();
        let digest = "ef".repeat(32);
        let key = ExplicitSessionKey::new(principal.as_str(), &digest);
        let cache_dir = tempfile::tempdir().unwrap();
        let cache_parent = cache_dir.path().join("blocked");
        std::fs::write(&cache_parent, b"block").unwrap();
        let cache = ConversationCache::persistent(cache_parent.join("sessions.json")).await;
        let file_dir = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent_with_limits(file_dir.path(), 1, 2)
            .await
            .unwrap();
        let mut session = explicit_test_conversation(ExplicitSessionState::Committed);
        for (name, byte) in [("first.txt", b'a'), ("second.txt", b'b')] {
            let id = stage(&files, &principal, name, byte).await.unwrap();
            files.add_reference(&id, &key.session_ref()).await.unwrap();
            session
                .explicit
                .as_mut()
                .unwrap()
                .file_mappings
                .insert(id, format!("upstream-{byte}"));
        }
        explicit_test_seed(&cache, key.clone(), session).await;

        let app = reset_app(
            ResetApiState {
                cache: cache.clone(),
                files: Some(files.clone()),
            },
            principal.clone(),
        );
        assert_eq!(
            app.clone()
                .oneshot(reset_request(&digest))
                .await
                .unwrap()
                .status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert!(cache.get_explicit(&key).await.is_some());

        std::fs::remove_file(&cache_parent).unwrap();
        std::fs::create_dir(&cache_parent).unwrap();
        assert_eq!(
            app.oneshot(reset_request(&digest)).await.unwrap().status(),
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn file_reconciliation_serializes_with_upload_and_reset() {
        let principal = AuthPrincipal::for_authenticated_user();
        let key = ExplicitSessionKey::new(principal.as_str(), "coordinated");
        let cache = ConversationCache::new();
        cache
            .set_explicit_checked(
                key.clone(),
                explicit_test_conversation(ExplicitSessionState::Committed),
            )
            .await
            .unwrap();
        let temp = tempfile::tempdir().unwrap();
        let files = StagedFileStore::persistent_with_limits(temp.path(), 1, 1)
            .await
            .unwrap();
        let staged = stage(&files, &principal, "first.txt", b'a').await.unwrap();

        let coordination = cache.lock_explicit_files().await;
        let stage_app = files_app(Some(files.clone()), cache.clone());
        let concurrent_stage = tokio::spawn(async move {
            post_files(stage_app, &[("file", "second.txt", "text/plain", b"b")])
                .await
                .status()
        });
        let reconcile = reconcile_task(cache.clone(), files.clone());
        tokio::task::yield_now().await;
        assert!(!concurrent_stage.is_finished());
        files
            .add_reference(&staged, &key.session_ref())
            .await
            .unwrap();
        cache
            .put_explicit_file_mapping(&key, &staged, "upstream")
            .await
            .unwrap();
        drop(coordination);
        reconcile.await.unwrap();
        assert_eq!(
            concurrent_stage.await.unwrap(),
            StatusCode::INSUFFICIENT_STORAGE
        );

        let coordination = cache.lock_explicit_files().await;
        let reconcile = reconcile_task(cache.clone(), files.clone());
        tokio::task::yield_now().await;
        files
            .remove_session_references(&key.session_ref())
            .await
            .unwrap();
        assert!(cache.reset_explicit(&key).await.unwrap());
        drop(coordination);
        reconcile.await.unwrap();
        stage(&files, &principal, "second.txt", b'b').await.unwrap();
    }
}
