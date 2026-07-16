pub mod files;
pub mod sessions;

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthPrincipal(pub String);

impl AuthPrincipal {
    pub fn for_authenticated_user() -> Self {
        use sha2::{Digest, Sha256};

        Self(hex::encode(Sha256::digest(
            b"clewdr-single-password-principal-v1",
        )))
    }

    #[cfg(test)]
    pub fn test_principal(identity: &str) -> Self {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        digest.update(b"clewdr-test-principal-v1\0");
        digest.update(identity.as_bytes());
        Self(hex::encode(digest.finalize()))
    }
}

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

pub fn parse_session_id(value: Option<&str>) -> Result<Option<String>, ProtocolError> {
    const PREFIX: &str = "cherry_topic_v1_";
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.starts_with(PREFIX) {
        return Ok(None);
    }
    let digest = &value[PREFIX.len()..];
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ProtocolError::new(
            StatusCode::BAD_REQUEST,
            "invalid_session_id",
            "Session ID must contain a 64-character lowercase hexadecimal digest",
        ));
    }
    Ok(Some(digest.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_only_versioned_lowercase_sha256_session_ids() {
        let valid = format!("cherry_topic_v1_{}", "ab".repeat(32));
        assert_eq!(
            parse_session_id(Some(&valid)).unwrap(),
            Some("ab".repeat(32))
        );
        assert!(parse_session_id(Some("legacy-client")).unwrap().is_none());
        assert_eq!(
            parse_session_id(Some(&format!("cherry_topic_v1_{}", "AB".repeat(32))))
                .unwrap_err()
                .code,
            "invalid_session_id"
        );
        assert_eq!(
            parse_session_id(Some("cherry_topic_v1_1234"))
                .unwrap_err()
                .code,
            "invalid_session_id"
        );
    }

    #[test]
    fn principal_is_versioned_and_stable_across_password_rotation() {
        let principal = AuthPrincipal::for_authenticated_user();
        assert_eq!(principal.0.len(), 64);
        assert_eq!(principal, AuthPrincipal::for_authenticated_user());
    }
}
