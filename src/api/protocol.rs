use axum::{Extension, Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};

use crate::{
    claude_web_state::conversation_cache::{ConversationCache, ExplicitSessionKey},
    protocol::{AuthPrincipal, ProtocolError, parse_session_id},
};

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
    State(cache): State<ConversationCache>,
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
    let _operation = cache.try_lock_explicit_operation(&key).await?;
    if !cache.reset_explicit(&key).await? {
        return Err(ProtocolError::new(
            StatusCode::NOT_FOUND,
            "session_not_found",
            "Session does not exist",
        ));
    }
    Ok(Json(ResetSessionResponse {
        r#type: "session_reset",
        session_id: request.session_id,
    }))
}

#[cfg(test)]
mod tests {
    use axum::{Extension, Router, body::Body, http::Request, routing::post};
    use tower::ServiceExt;

    use super::*;
    use crate::claude_web_state::{
        conversation_cache::CachedConversation,
        explicit_session::{ExplicitConversation, ExplicitSessionState},
    };

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
            explicit: Some(ExplicitConversation {
                state,
                model_digest: "model".into(),
                system_digest: "system".into(),
                turns: Vec::new(),
                pending: None,
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
        let app = Router::new()
            .route("/v1/sessions/reset", post(api_reset_session))
            .layer(Extension(principal))
            .with_state(cache.clone());
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
    }
}
