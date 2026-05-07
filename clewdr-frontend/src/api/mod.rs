use gloo_net::http::Request;

use crate::{
    storage,
    types::{ConfigData, CookieStatusInfo},
};

fn auth_header() -> String {
    format!("Bearer {}", storage::get("authToken").unwrap_or_default())
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
        .header("Content-Type", "application/json")
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
    let resp = Request::get(url)
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("Error {}", resp.status()));
    }
    resp.json::<CookieStatusInfo>()
        .await
        .map_err(|e| e.to_string())
}

pub async fn post_cookie(cookie: &str) -> Result<(), String> {
    let body = serde_json::json!({ "cookie": cookie });
    let resp = Request::post("/api/cookie")
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    match resp.status() {
        200 => Ok(()),
        400 => Err("Invalid cookie format".into()),
        401 => Err("Authentication failed".into()),
        s => {
            let body = resp.text().await.unwrap_or_default();
            Err(format!("Error {s}: {body}"))
        }
    }
}

pub async fn delete_cookie(cookie: &str) -> Result<(), String> {
    let body = serde_json::json!({ "cookie": cookie });
    let resp = Request::delete("/api/cookie")
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("Error {}: {body}", resp.status()))
    }
}

pub async fn get_config() -> Result<ConfigData, String> {
    let resp = Request::get("/api/config")
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.ok() {
        return Err(format!("Error {}", resp.status()));
    }
    resp.json::<ConfigData>().await.map_err(|e| e.to_string())
}

pub async fn save_config(config: &ConfigData) -> Result<(), String> {
    let resp = Request::post("/api/config")
        .header("Authorization", &auth_header())
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(config).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if resp.ok() {
        Ok(())
    } else {
        let text = resp.text().await.unwrap_or_default();
        Err(format!("Error {}: {text}", resp.status()))
    }
}
