use leptos::{ev, prelude::*};
use wasm_bindgen_futures::spawn_local;

use crate::{api, i18n::use_i18n, storage, utils};

#[component]
pub fn AuthGatekeeper(is_authenticated: RwSignal<bool>) -> impl IntoView {
    let i18n = use_i18n();
    let token_input = RwSignal::new(String::new());
    let loading = RwSignal::new(false);
    let error = RwSignal::new(Option::<String>::None);
    let saved_token = storage::get("authToken")
        .map(|t| utils::mask_str(&t, 4))
        .filter(|s| !s.is_empty());

    let on_submit = {
        let i = use_i18n();
        move |ev: ev::SubmitEvent| {
            ev.prevent_default();
            let token = token_input.get_untracked().trim().to_string();
            if token.is_empty() {
                error.set(Some(i.t("auth.enterToken")));
                return;
            }
            loading.set(true);
            error.set(None);
            spawn_local(async move {
                match api::validate_auth(&token).await {
                    Ok(true) => {
                        storage::set("authToken", &token);
                        is_authenticated.set(true);
                    }
                    Ok(false) => error.set(Some(i.t("auth.invalid"))),
                    Err(e) => error.set(Some(e)),
                }
                loading.set(false);
            });
        }
    };

    view! {
        <form on:submit=on_submit class="stack">
            <div>
                <label class="label" for="authToken">
                    {move || i18n.t("auth.token")}
                </label>
                <input
                    id="authToken"
                    type="password"
                    class="input"
                    placeholder=move || i18n.t("auth.tokenPlaceholder")
                    disabled=move || loading.get()
                    prop:value=move || token_input.get()
                    on:input=move |ev| token_input.set(event_target_value(&ev))
                />
            </div>

            {saved_token.map(|masked| view! {
                <p class="text-xs text-mute">
                    {i18n.t("auth.previousToken")} " "
                    <span class="text-mono">{masked}</span>
                </p>
            })}

            <Show when=move || error.get().is_some()>
                <div class="alert alert-error">
                    {move || error.get().unwrap_or_default()}
                </div>
            </Show>

            <button
                type="submit"
                class="btn btn-primary btn-block"
                disabled=move || loading.get()
            >
                {move || if loading.get() { i18n.t("auth.verifying") } else { i18n.t("auth.submitButton") }}
            </button>
        </form>
    }
}
