use gloo_net::http::{Request, Response};
use serde::de::DeserializeOwned;

use crate::{
    storage,
    types::{ConfigData, CookieStatusInfo},
};

fn auth_header() -> String {
    format!("Bearer {}", storage::get("authToken").unwrap_or_default())
}

async fn extract_error(resp: &Response) -> String {
    #[derive(serde::Deserialize)]
    struct ErrBody {
        error: String,
    }
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if let Ok(body) = serde_json::from_str::<ErrBody>(&text) {
        body.error
    } else if text.is_empty() {
        format!("HTTP {status}")
    } else {
        text
    }
}

async fn authed_get<T: DeserializeOwned>(url: &str) -> Result<T, String> {
    let resp = Request::get(url)
        .header("Authorization", &auth_header())
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(extract_error(&resp).await);
    }
    resp.json().await.map_err(|e| e.to_string())
}

async fn authed_post(url: &str, body: &impl serde::Serialize) -> Result<(), String> {
    let resp = Request::post(url)
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(body).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        Err(extract_error(&resp).await)
    }
}

pub async fn get_version() -> Result<String, String> {
    Request::get("/api/version")
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())
}

pub async fn validate_auth(token: &str) -> Result<bool, String> {
    let resp = Request::get("/api/auth")
        .header("Authorization", &format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    Ok(resp.ok())
}

pub async fn get_cookies(force_refresh: bool) -> Result<CookieStatusInfo, String> {
    let url = if force_refresh {
        "/api/cookies?refresh=true"
    } else {
        "/api/cookies"
    };
    authed_get(url).await
}

pub async fn post_cookie(cookie: &str) -> Result<(), String> {
    authed_post("/api/cookie", &serde_json::json!({ "cookie": cookie })).await
}

pub async fn delete_cookie(cookie: &str) -> Result<(), String> {
    let resp = Request::delete("/api/cookie")
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&serde_json::json!({ "cookie": cookie })).unwrap())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        Err(extract_error(&resp).await)
    }
}

pub async fn get_config() -> Result<ConfigData, String> {
    authed_get("/api/config").await
}

pub async fn save_config(config: &ConfigData) -> Result<(), String> {
    authed_post("/api/config", config).await
}
