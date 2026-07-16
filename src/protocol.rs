use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SESSION_ID_PREFIX: &str = "cherry_topic_v1_";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthPrincipal(String);

impl AuthPrincipal {
    pub fn for_authenticated_user() -> Self {
        Self(hex::encode(Sha256::digest(
            b"clewdr-single-password-principal-v1",
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, thiserror::Error)]
#[error("session ID must contain a 64-character lowercase hexadecimal digest")]
pub struct InvalidSessionId;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct ProtocolError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ProtocolError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    r#type: &'static str,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    r#type: &'a str,
    message: &'a str,
}

impl IntoResponse for ProtocolError {
    fn into_response(self) -> axum::response::Response {
        let body = ErrorEnvelope {
            r#type: "error",
            error: ErrorBody {
                r#type: self.code,
                message: &self.message,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

pub fn parse_session_id(value: Option<&str>) -> Result<Option<String>, InvalidSessionId> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(digest) = value.strip_prefix(SESSION_ID_PREFIX) else {
        return Ok(None);
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InvalidSessionId);
    }
    Ok(Some(digest.to_owned()))
}

#[cfg(test)]
impl AuthPrincipal {
    pub fn test_principal(identity: &str) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"clewdr-test-principal-v1\0");
        digest.update(identity.as_bytes());
        Self(hex::encode(digest.finalize()))
    }
}
