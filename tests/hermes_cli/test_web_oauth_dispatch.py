"""Regression tests for the OAuth dispatcher in hermes_cli.web_server.

Bug history (2026-05-09): the `_OAUTH_PROVIDER_CATALOG` had two entries
flagged ``flow: "pkce"`` — anthropic and minimax-oauth — and the
dispatcher ``start_oauth_login`` hardcoded ``_start_anthropic_pkce()``
for any pkce-flagged provider. So clicking "Login" next to MiniMax in
the dashboard's Keys tab silently launched the Anthropic/Claude OAuth
flow.

The fix:
  1. Catalog entry for minimax-oauth changed from ``flow: "pkce"`` to
     ``flow: "device_code"`` (the actual UX is verification URI + user
     code + background poll, with PKCE as a security extension).
  2. New MiniMax branch added to ``_start_device_code_flow``.
  3. Dispatcher tightened: pkce branch now requires
     ``provider_id == "anthropic"``, so any future PKCE provider added
     without an explicit branch gets a clean ``400 Unsupported flow``
     instead of silently launching Anthropic OAuth.

These tests pin the corrected behavior.
"""
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from hermes_cli.web_server import app

client = TestClient(app)


def _headers():
    from hermes_cli import web_server as ws

    return {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}


def _asqend_profile_headers():
    return {
        **_headers(),
        "x-asqend-hermes-profile-ref": "hermes-profile:settings",
        "x-asqend-hermes-profile-group-key": "settings-ai-connection",
        "x-asqend-hermes-runtime-session-ref": "settings-runtime-session",
    }


def _asqend_server_key_profile_headers():
    return {
        "authorization": "Bearer asqend-server-key",
        "x-asqend-hermes-profile-ref": "hermes-profile:settings",
        "x-asqend-hermes-profile-group-key": "settings-ai-connection",
        "x-asqend-hermes-runtime-session-ref": "settings-runtime-session",
    }
ASQEND_PROFILE_PAYLOAD = {
    "contractVersion": "2026-06-04.hosted-codex-oauth",
    "profileRef": "hermes-profile:settings",
    "profileGroupKey": "settings-ai-connection",
    "runtimeSessionRef": "settings-runtime-session",
    "credentialMode": "spike_local_codex_subscription",
    "modelProfileRef": None,
    "isolation": {
        "unit": "dedicated_vm",
        "hostRef": "hermes-host",
        "wholeProcessWrapped": False,
    },
}


def _reload_web_server_with_asqend_identity(monkeypatch):
    from hermes_cli import web_server as ws
    import hermes_cli.asqend_identity as identity

    monkeypatch.setattr(
        identity,
        "_BOOT_IDENTITY",
        {
            "org_id": "org-oauth-a",
            "container_ref": "container-oauth-a",
            "source": identity.ASQEND_IDENTITY_SOURCE,
            "version": identity.ASQEND_IDENTITY_VERSION,
        },
    )
    local_client = TestClient(ws.app)
    local_headers = {ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN}
    return ws, local_client, local_headers


def _reset_web_server_without_asqend_identity(monkeypatch, ws):
    import hermes_cli.asqend_identity as identity

    monkeypatch.setattr(identity, "_BOOT_IDENTITY", None)


def _fake_nous_device_data():
    return {
        "device_code": "device-code",
        "user_code": "NOUS-1234",
        "verification_uri": "https://portal.nousresearch.com/device",
        "verification_uri_complete": (
            "https://portal.nousresearch.com/device?user_code=NOUS-1234"
        ),
        "expires_in": 600,
        "interval": 5,
    }


def _invoke_scope_refusal():
    request = httpx.Request("POST", "https://portal.nousresearch.com/oauth/device/code")
    response = httpx.Response(
        400,
        json={
            "error": "invalid_scope",
            "error_description": "unsupported scope inference:invoke",
        },
        request=request,
    )
    return httpx.HTTPStatusError("invalid scope", request=request, response=response)


def test_minimax_login_does_not_launch_anthropic_flow():
    """Click 'Login' on MiniMax → MUST NOT return claude.ai auth_url."""
    fake_user_code_resp = {
        "user_code": "ABCD-1234",
        "verification_uri": "https://api.minimax.io/oauth/verify",
        # `expired_in` < 1e12 so the heuristic treats it as seconds.
        "expired_in": 600,
        "interval": 2000,
        "state": "stub-state",
    }
    with patch(
        "hermes_cli.auth._minimax_request_user_code",
        return_value=fake_user_code_resp,
    ), patch(
        "hermes_cli.auth._minimax_pkce_pair",
        return_value=("verifier-stub", "challenge-stub", "stub-state"),
    ), patch(
        "hermes_cli.web_server._minimax_poller",
        return_value=None,
    ):
        resp = client.post(
            "/api/providers/oauth/minimax-oauth/start",
            headers=_headers(),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The bug used to return Anthropic's auth_url — make sure the response
    # references neither the auth_url field nor anything Claude-related.
    assert "auth_url" not in body
    assert "claude.ai" not in str(body).lower()

    # And the response IS the device-code shape pointing at MiniMax.
    assert body["flow"] == "device_code"
    assert "minimax" in body["verification_url"].lower()
    assert body["user_code"] == "ABCD-1234"
    assert body["expires_in"] == 600


def test_nous_dashboard_device_flow_ignores_legacy_scope_override(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    requested_scopes = []

    def fake_request_device_code(**kwargs):
        requested_scopes.append(kwargs["scope"])
        return _fake_nous_device_data()

    monkeypatch.setenv("HERMES_AGENT_USE_LEGACY_SESSION_KEYS", "true")
    monkeypatch.setattr(auth_mod, "_request_device_code", fake_request_device_code)
    monkeypatch.setattr(ws, "_nous_poller", lambda sid: None)

    result = asyncio.run(ws._start_device_code_flow("nous"))
    try:
        assert requested_scopes == [auth_mod.DEFAULT_NOUS_SCOPE]
        assert result["flow"] == "device_code"
        assert result["user_code"] == "NOUS-1234"
        assert (
            ws._oauth_sessions[result["session_id"]]["scope"]
            == auth_mod.DEFAULT_NOUS_SCOPE
        )
    finally:
        ws._oauth_sessions.pop(result["session_id"], None)


def test_nous_dashboard_device_flow_does_not_retry_legacy_scope_on_invoke_refusal(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    requested_scopes = []

    def fake_request_device_code(**kwargs):
        requested_scopes.append(kwargs["scope"])
        raise _invoke_scope_refusal()

    monkeypatch.delenv("HERMES_AGENT_USE_LEGACY_SESSION_KEYS", raising=False)
    monkeypatch.setattr(auth_mod, "_request_device_code", fake_request_device_code)
    monkeypatch.setattr(ws, "_nous_poller", lambda sid: None)

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(ws._start_device_code_flow("nous"))
    assert requested_scopes == [auth_mod.DEFAULT_NOUS_SCOPE]


def test_codex_dashboard_worker_persists_runtime_provider(tmp_path, monkeypatch):
    from hermes_cli import web_server as ws
    from hermes_cli.auth import get_active_provider
    from hermes_cli.runtime_provider import resolve_runtime_provider

    access_token = "h.eyJleHAiOjk5OTk5OTk5OTl9.s"

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 3,
                    "user_code": "CODEX-1234",
                })
            if url.endswith("/deviceauth/token"):
                return _Resp(200, {
                    "authorization_code": "authorization-code",
                    "code_verifier": "code-verifier",
                })
            return _Resp(200, {
                "access_token": access_token,
                "refresh_token": "codex-refresh",
            })

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)

    sid, _ = ws._new_oauth_session("openai-codex", "device_code")
    try:
        ws._codex_full_login_worker(sid)

        assert ws._oauth_sessions[sid]["status"] == "approved"
        assert get_active_provider() == "openai-codex"

        runtime = resolve_runtime_provider(requested=None)
        assert runtime["provider"] == "openai-codex"
        assert runtime["api_key"] == access_token
        assert runtime["api_mode"] == "codex_responses"
    finally:
        ws._oauth_sessions.pop(sid, None)


def test_codex_dashboard_worker_keeps_polling_after_openai_read_timeout(tmp_path, monkeypatch):
    from hermes_cli import web_server as ws
    from hermes_cli.auth import get_active_provider

    access_token = "h.eyJleHAiOjk5OTk5OTk5OTl9.s"
    token_poll_attempts = 0

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            nonlocal token_poll_attempts
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 3,
                    "user_code": "CODEX-1234",
                })
            if url.endswith("/deviceauth/token"):
                token_poll_attempts += 1
                if token_poll_attempts == 1:
                    raise httpx.ReadTimeout("The read operation timed out")
                return _Resp(200, {
                    "authorization_code": "authorization-code",
                    "code_verifier": "code-verifier",
                })
            return _Resp(200, {
                "access_token": access_token,
                "refresh_token": "codex-refresh",
            })

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)

    sid, _ = ws._new_oauth_session("openai-codex", "device_code")
    try:
        ws._codex_full_login_worker(sid)

        assert token_poll_attempts == 2
        assert ws._oauth_sessions[sid]["status"] == "approved"
        assert ws._oauth_sessions[sid]["user_code"] == "CODEX-1234"
        assert get_active_provider() == "openai-codex"
    finally:
        ws._oauth_sessions.pop(sid, None)


def test_nous_dashboard_poller_preserves_effective_scope_when_token_omits_scope(monkeypatch):
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "nous-effective-scope-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "nous",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://portal.nousresearch.com",
        "client_id": "hermes-cli",
        "device_code": "device-code",
        "interval": 5,
        "expires_at": time.time() + 600,
        "scope": auth_mod.DEFAULT_NOUS_SCOPE,
    }
    captured_state = {}

    def fake_refresh_nous_oauth_from_state(state, **kwargs):
        captured_state.update(state)
        return {**state, "agent_key": "jwt-agent-key"}

    monkeypatch.setattr(
        auth_mod,
        "_poll_for_token",
        lambda **kwargs: {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "refresh_nous_oauth_from_state",
        fake_refresh_nous_oauth_from_state,
    )
    monkeypatch.setattr(auth_mod, "persist_nous_credentials", lambda state: None)

    try:
        ws._nous_poller(session_id)
        assert captured_state["scope"] == auth_mod.DEFAULT_NOUS_SCOPE
        assert ws._oauth_sessions[session_id]["status"] == "approved"
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_minimax_dashboard_poller_accepts_absolute_ms_expired_in():
    """Dashboard MiniMax completion must accept unix-ms token expiry values."""
    from hermes_cli import web_server as ws

    now = datetime.now(timezone.utc)
    abs_ms = int((now.timestamp() + 1800) * 1000)
    session_id = "minimax-absolute-ms-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "minimax-oauth",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "portal_base_url": "https://api.minimax.io",
        "client_id": "client-id",
        "user_code": "ABCD-1234",
        "code_verifier": "verifier",
        "interval_ms": 2000,
        "expired_in_raw": abs_ms,
        "region": "global",
    }
    captured_state = {}

    try:
        with patch(
            "hermes_cli.auth._minimax_poll_token",
            return_value={
                "status": "success",
                "access_token": "access",
                "refresh_token": "refresh",
                "expired_in": abs_ms,
                "token_type": "Bearer",
            },
        ), patch(
            "hermes_cli.auth._minimax_save_auth_state",
            side_effect=lambda state: captured_state.update(state),
        ):
            ws._minimax_poller(session_id)
    finally:
        ws._oauth_sessions.pop(session_id, None)

    assert captured_state["access_token"] == "access"
    assert 1790 <= captured_state["expires_in"] <= 1810
    assert datetime.fromisoformat(captured_state["expires_at"]).year < 9999


def test_anthropic_pkce_branch_still_works():
    """Sanity: the dispatcher tightening doesn't break the legitimate Anthropic PKCE path."""
    fake_anthropic_response = {
        "session_id": "stub-session",
        "flow": "pkce",
        "auth_url": "https://claude.ai/oauth/authorize?code=true&...",
        "expires_in": 600,
    }
    with patch(
        "hermes_cli.web_server._start_anthropic_pkce",
        return_value=fake_anthropic_response,
    ):
        resp = client.post(
            "/api/providers/oauth/anthropic/start",
            headers=_headers(),
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["flow"] == "pkce"
    assert "claude.ai" in body["auth_url"]


def test_xai_oauth_listed_as_loopback_flow():
    """xAI Grok OAuth must surface in the catalog as a first-class loopback flow."""
    resp = client.get("/api/providers/oauth", headers=_headers())
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}
    assert "xai-oauth" in providers
    assert providers["xai-oauth"]["flow"] == "loopback"
    assert "grok" in providers["xai-oauth"]["name"].lower()


def test_oauth_provider_paths_accept_asqend_server_key_without_dashboard_token(monkeypatch):
    """Asqend uses the API server bearer, not the dashboard session header."""
    from hermes_cli import web_server as ws

    monkeypatch.setenv("HERMES_API_SERVER_KEY", "asqend-server-key")
    resp = client.get("/api/providers/oauth", headers=_asqend_server_key_profile_headers())
    assert resp.status_code == 200, resp.text

    blocked = client.post(
        "/api/env/reveal",
        headers={"authorization": "Bearer asqend-server-key"},
        json={"key": "SHOULD_NOT_REVEAL"},
    )
    assert blocked.status_code == 401

    previous_auth_required = getattr(ws.app.state, "auth_required", None)
    ws.app.state.auth_required = True
    try:
        hosted_resp = client.get(
            "/api/providers/oauth",
            headers=_asqend_server_key_profile_headers(),
        )
        assert hosted_resp.status_code == 200, hosted_resp.text
    finally:
        ws.app.state.auth_required = previous_auth_required


def test_codex_oauth_status_echoes_asqend_profile_scope():
    """Hosted Asqend callers need profile proof, not process-global status."""
    resp = client.get("/api/providers/oauth", headers=_asqend_profile_headers())
    assert resp.status_code == 200, resp.text

    providers = {p["id"]: p for p in resp.json()["providers"]}
    codex = providers["openai-codex"]
    assert codex["profile_ref"] == "hermes-profile:settings"
    assert codex["profile_group_key"] == "settings-ai-connection"
    assert codex["runtime_session_ref"] == "settings-runtime-session"
    assert codex["status"]["profile_ref"] == "hermes-profile:settings"
    assert codex["status"]["runtime_session_ref"] == "settings-runtime-session"


def test_oauth_routes_echo_process_start_asqend_identity(monkeypatch):
    """Catalog/start/poll/cancel must prove which org container served OAuth."""
    ws, local_client, local_headers = _reload_web_server_with_asqend_identity(monkeypatch)

    async def fake_start_device_code_flow(provider_id, profile_context=None):
        sid, sess = ws._new_oauth_session(provider_id, "device_code", profile_context)
        sess["user_code"] = "CODEX-IDENTITY"
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": "CODEX-IDENTITY",
            "verification_url": "https://example.invalid/device",
            "expires_in": 600,
            "poll_interval": 5,
        }

    monkeypatch.setattr(ws, "_start_device_code_flow", fake_start_device_code_flow)
    try:
        catalog = local_client.get("/api/providers/oauth", headers=local_headers)
        assert catalog.status_code == 200, catalog.text
        assert catalog.json()["asqend_identity"] == {
            "org_id": "org-oauth-a",
            "container_ref": "container-oauth-a",
            "surface": "dashboard_oauth",
            "source": "process_env",
            "version": catalog.json()["asqend_identity"]["version"],
        }

        start = local_client.post(
            "/api/providers/oauth/openai-codex/start",
            headers=local_headers,
        )
        assert start.status_code == 200, start.text
        sid = start.json()["session_id"]
        assert start.json()["asqend_identity"]["org_id"] == "org-oauth-a"
        assert start.json()["asqend_identity"]["container_ref"] == "container-oauth-a"
        assert start.json()["asqend_identity"]["surface"] == "dashboard_oauth_start"

        poll = local_client.get(
            f"/api/providers/oauth/openai-codex/poll/{sid}",
            headers=local_headers,
        )
        assert poll.status_code == 200, poll.text
        assert poll.json()["asqend_identity"]["surface"] == "dashboard_oauth_poll"
        assert poll.json()["asqend_identity"]["container_ref"] == "container-oauth-a"

        cancel = local_client.delete(
            f"/api/providers/oauth/sessions/{sid}",
            headers=local_headers,
        )
        assert cancel.status_code == 200, cancel.text
        assert cancel.json()["asqend_identity"]["surface"] == "dashboard_oauth_cancel"
        assert cancel.json()["asqend_identity"]["org_id"] == "org-oauth-a"
    finally:
        _reset_web_server_without_asqend_identity(monkeypatch, ws)


def test_oauth_identity_headers_without_process_env_are_rejected(monkeypatch):
    """Asqend expected-identity headers must not be reflected as Hermes identity."""
    from hermes_cli import web_server as ws

    import hermes_cli.asqend_identity as identity
    monkeypatch.setattr(identity, "_BOOT_IDENTITY", None)
    local_client = TestClient(ws.app)

    resp = local_client.get(
        "/api/providers/oauth",
        headers={
            ws._SESSION_HEADER_NAME: ws._SESSION_TOKEN,
            "x-asqend-org-id": "org-header-only",
            "x-asqend-hermes-container-ref": "container-header-only",
        },
    )

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["code"] == "asqend_identity_missing"


def test_codex_oauth_status_does_not_launder_mismatched_profile(monkeypatch):
    """Global/profile-A Codex auth must not unlock Asqend profile B."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "get_codex_auth_status",
        lambda: {
            "logged_in": True,
            "source": "pool:device_code",
            "source_label": "chatgpt",
            "token_preview": "secret-preview",
            "last_refresh": "2026-06-04T20:00:00Z",
            "profile_ref": "hermes-profile:other",
            "profile_group_key": "other-profile",
            "runtime_session_ref": "other-runtime-session",
        },
    )

    resp = client.get("/api/providers/oauth", headers=_asqend_profile_headers())
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}
    codex = providers["openai-codex"]
    assert codex["profile_ref"] == "hermes-profile:settings"
    assert codex["status"]["profile_ref"] == "hermes-profile:settings"
    assert codex["status"]["logged_in"] is False
    assert codex["status"]["token_preview"] is None
    assert codex["status"]["last_refresh"] is None
    assert codex["status"]["error"] == "codex_credentials_not_bound_to_requested_profile"


def test_codex_oauth_status_preserves_matching_logged_in_profile(monkeypatch):
    """Logged-in Codex status must keep profile proof through the HTTP adapter."""
    from hermes_cli import auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "get_codex_auth_status",
        lambda: {
            "logged_in": True,
            "auth_mode": "chatgpt",
            "source": "pool:device_code",
            "api_key": "codex-access-token",
            "last_refresh": "2026-06-04T20:00:00Z",
            "profile_ref": "hermes-profile:settings",
            "profile_group_key": "settings-ai-connection",
            "runtime_session_ref": "settings-runtime-session",
        },
    )

    resp = client.get("/api/providers/oauth", headers=_asqend_profile_headers())
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}
    codex = providers["openai-codex"]
    assert codex["profile_ref"] == "hermes-profile:settings"
    assert codex["status"]["logged_in"] is True
    assert codex["status"]["last_refresh"] == "2026-06-04T20:00:00Z"
    assert codex["status"]["profile_ref"] == "hermes-profile:settings"
    assert codex["status"]["profile_group_key"] == "settings-ai-connection"
    assert codex["status"]["runtime_session_ref"] == "settings-runtime-session"
    assert "error" not in codex["status"]


def test_codex_oauth_start_binds_and_echoes_asqend_profile(monkeypatch):
    """Start must bind the session to the Asqend runtime profile it was issued for."""
    from hermes_cli import web_server as ws

    def _fake_codex_worker(session_id):
        with ws._oauth_sessions_lock:
            sess = ws._oauth_sessions[session_id]
            sess["user_code"] = "CODEX-1234"
            sess["verification_url"] = "https://auth.openai.com/codex/device"
            sess["device_auth_id"] = "device-auth-id"
            sess["interval"] = 5
            sess["expires_in"] = 900
            sess["expires_at"] = time.time() + 900

    monkeypatch.setattr(ws, "_codex_full_login_worker", _fake_codex_worker)

    resp = client.post(
        "/api/providers/oauth/openai-codex/start",
        headers=_asqend_profile_headers(),
        json={"profile": ASQEND_PROFILE_PAYLOAD},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    try:
        assert body["profile_ref"] == "hermes-profile:settings"
        assert body["profile_group_key"] == "settings-ai-connection"
        assert body["runtime_session_ref"] == "settings-runtime-session"

        sess = ws._oauth_sessions[body["session_id"]]
        assert sess["profile_ref"] == "hermes-profile:settings"
        assert sess["profile_group_key"] == "settings-ai-connection"
        assert sess["runtime_session_ref"] == "settings-runtime-session"
    finally:
        ws._oauth_sessions.pop(body.get("session_id"), None)


def test_codex_oauth_start_without_asqend_profile_remains_local_dashboard_compatible(monkeypatch):
    """Local dashboard starts should keep their existing unscoped response shape."""
    from hermes_cli import web_server as ws

    def _fake_codex_worker(session_id):
        with ws._oauth_sessions_lock:
            sess = ws._oauth_sessions[session_id]
            sess["user_code"] = "CODEX-LOCAL"
            sess["verification_url"] = "https://auth.openai.com/codex/device"
            sess["device_auth_id"] = "device-auth-id"
            sess["interval"] = 5
            sess["expires_in"] = 900
            sess["expires_at"] = time.time() + 900

    monkeypatch.setattr(ws, "_codex_full_login_worker", _fake_codex_worker)

    resp = client.post("/api/providers/oauth/openai-codex/start", headers=_headers())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    try:
        assert body["flow"] == "device_code"
        assert body["user_code"] == "CODEX-LOCAL"
        assert "profile_ref" not in body
        assert "runtime_session_ref" not in body
    finally:
        ws._oauth_sessions.pop(body.get("session_id"), None)


def test_codex_oauth_poll_rejects_asqend_profile_mismatch():
    """A session created for one Asqend profile must not be pollable by another."""
    from hermes_cli import web_server as ws

    session_id = "codex-profile-mismatch-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "openai-codex",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "expires_at": time.time() + 900,
        "profile_ref": "hermes-profile:settings",
        "profile_group_key": "settings-ai-connection",
        "runtime_session_ref": "settings-runtime-session",
    }
    try:
        resp = client.get(
            f"/api/providers/oauth/openai-codex/poll/{session_id}",
            headers={
                **_asqend_profile_headers(),
                "x-asqend-hermes-profile-ref": "hermes-profile:other",
            },
        )
        assert resp.status_code == 404, resp.text
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_codex_oauth_cancel_rejects_asqend_profile_mismatch():
    """A caller from another profile must not be able to cancel this session."""
    from hermes_cli import web_server as ws

    session_id = "codex-cancel-profile-mismatch-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "openai-codex",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "expires_at": time.time() + 900,
        "profile_ref": "hermes-profile:settings",
        "profile_group_key": "settings-ai-connection",
        "runtime_session_ref": "settings-runtime-session",
    }
    try:
        resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}",
            headers={
                **_asqend_profile_headers(),
                "x-asqend-hermes-profile-ref": "hermes-profile:other",
            },
        )
        assert resp.status_code == 404, resp.text
        assert session_id in ws._oauth_sessions
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_codex_oauth_cancel_echoes_profile_and_verified_no_write():
    """Cancelling a Codex device-code session must prove the worker cannot persist."""
    from hermes_cli import web_server as ws

    session_id = "codex-cancel-profile-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "openai-codex",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "expires_at": time.time() + 900,
        "profile_ref": "hermes-profile:settings",
        "profile_group_key": "settings-ai-connection",
        "runtime_session_ref": "settings-runtime-session",
    }
    try:
        resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}",
            headers=_asqend_profile_headers(),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["session_id"] == session_id
        assert body["profile_ref"] == "hermes-profile:settings"
        assert body["worker_cancelled"] is True
        assert body["token_write_prevented"] is True
        assert body["cancel_safety"] == "verified_no_write"
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_codex_oauth_cancel_is_unverified_after_token_write_claim():
    """After the worker claims token persistence, cancel can no longer prove no-write."""
    from hermes_cli import web_server as ws

    session_id = "codex-cancel-write-claimed-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "openai-codex",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "expires_at": time.time() + 900,
        "profile_ref": "hermes-profile:settings",
        "profile_group_key": "settings-ai-connection",
        "runtime_session_ref": "settings-runtime-session",
        "token_write_in_progress": True,
    }
    try:
        resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}",
            headers=_asqend_profile_headers(),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["session_id"] == session_id
        assert body["profile_ref"] == "hermes-profile:settings"
        assert body["cancel_safety"] == "unverified"
        assert body.get("token_write_prevented") is not True
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_codex_worker_skips_persist_when_cancelled_after_token_exchange(monkeypatch):
    """If cancel wins during token exchange, Codex tokens must never be saved."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    saved_tokens = []
    session_id = "codex-worker-cancel-no-write-test"

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if url.endswith("/deviceauth/usercode"):
                return _Resp(200, {
                    "device_auth_id": "device-auth-id",
                    "interval": 3,
                    "user_code": "CODEX-1234",
                })
            if url.endswith("/deviceauth/token"):
                return _Resp(200, {
                    "authorization_code": "authorization-code",
                    "code_verifier": "code-verifier",
                })
            ws._oauth_sessions.pop(session_id, None)
            return _Resp(200, {
                "access_token": "codex-access",
                "refresh_token": "codex-refresh",
            })

    monkeypatch.setattr(httpx, "Client", _Client)
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        auth_mod,
        "_save_codex_tokens",
        lambda tokens, **kwargs: saved_tokens.append(dict(tokens)),
    )

    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "openai-codex",
        "flow": "device_code",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
    }
    try:
        ws._codex_full_login_worker(session_id)
        assert saved_tokens == []
        assert session_id not in ws._oauth_sessions
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_xai_loopback_start_returns_authorize_url(monkeypatch):
    """Start MUST bind the loopback listener and hand back an xAI authorize URL."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    class _FakeServer:
        def shutdown(self):
            pass

        def server_close(self):
            pass

    class _FakeThread:
        def join(self, timeout=None):
            pass

    redirect_uri = (
        f"http://{auth_mod.XAI_OAUTH_REDIRECT_HOST}:{auth_mod.XAI_OAUTH_REDIRECT_PORT}"
        f"{auth_mod.XAI_OAUTH_REDIRECT_PATH}"
    )

    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_discovery",
        lambda *a, **k: {
            "authorization_endpoint": "https://auth.x.ai/oauth2/auth",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "_xai_start_callback_server",
        lambda *a, **k: (_FakeServer(), _FakeThread(), {"code": None, "error": None}, redirect_uri),
    )
    # Don't let the background worker run a real callback wait/exchange.
    monkeypatch.setattr(ws, "_xai_loopback_worker", lambda sid: None)

    resp = client.post("/api/providers/oauth/xai-oauth/start", headers=_headers())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    try:
        assert body["flow"] == "loopback"
        assert "user_code" not in body  # loopback has nothing to paste/show
        assert body["auth_url"].startswith("https://auth.x.ai/oauth2/auth?")
        assert "code_challenge" in body["auth_url"]
        sess = ws._oauth_sessions[body["session_id"]]
        assert sess["provider"] == "xai-oauth"
        assert sess["flow"] == "loopback"
    finally:
        ws._oauth_sessions.pop(body["session_id"], None)


def test_xai_loopback_worker_persists_tokens_on_success(monkeypatch):
    """The worker exchanges the callback code and marks the session approved."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    saved = {}
    session_id = "xai-loopback-success-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "server": object(),
        "thread": object(),
        "callback_result": {"code": "auth-code", "state": "st"},
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "verifier": "verifier",
        "challenge": "challenge",
        "state": "st",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "discovery": {"token_endpoint": "https://auth.x.ai/oauth2/token"},
    }

    monkeypatch.setattr(
        auth_mod,
        "_xai_wait_for_callback",
        lambda *a, **k: {"code": "auth-code", "state": "st"},
    )
    monkeypatch.setattr(
        auth_mod,
        "_xai_oauth_exchange_code_for_tokens",
        lambda **k: {
            "access_token": "xai-access",
            "refresh_token": "xai-refresh",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(
        auth_mod,
        "_save_xai_oauth_tokens",
        lambda tokens, **k: saved.update(tokens),
    )
    monkeypatch.setattr(ws, "_add_xai_oauth_pool_entry", lambda *a, **k: None)

    try:
        ws._xai_loopback_worker(session_id)
        assert ws._oauth_sessions[session_id]["status"] == "approved"
        assert saved["access_token"] == "xai-access"
        assert saved["refresh_token"] == "xai-refresh"
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_xai_loopback_worker_fails_on_state_mismatch(monkeypatch):
    """A mismatched OAuth state must fail the session, not persist tokens."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "xai-loopback-state-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "server": object(),
        "thread": object(),
        "callback_result": {},
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "verifier": "verifier",
        "challenge": "challenge",
        "state": "expected-state",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "discovery": {},
    }

    monkeypatch.setattr(
        auth_mod,
        "_xai_wait_for_callback",
        lambda *a, **k: {"code": "auth-code", "state": "ATTACKER-state"},
    )

    def _boom(**kwargs):
        raise AssertionError("token exchange must not run on state mismatch")

    monkeypatch.setattr(auth_mod, "_xai_oauth_exchange_code_for_tokens", _boom)

    try:
        ws._xai_loopback_worker(session_id)
        sess = ws._oauth_sessions[session_id]
        assert sess["status"] == "error"
        assert "state mismatch" in sess["error_message"].lower()
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_xai_loopback_worker_skips_persist_when_cancelled(monkeypatch):
    """If the session is cancelled while waiting, the worker must not persist."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import web_server as ws

    session_id = "xai-loopback-cancel-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "error_message": None,
        "server": object(),
        "thread": object(),
        "callback_result": {},
        "redirect_uri": "http://127.0.0.1:56121/callback",
        "verifier": "verifier",
        "challenge": "challenge",
        "state": "st",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "discovery": {},
    }

    def _wait_then_cancel(*args, **kwargs):
        # Simulate the user cancelling (DELETE /sessions/{id}) while we were
        # blocked on the callback: the session vanishes, then a valid code
        # arrives. The worker must notice and bail before persisting.
        ws._oauth_sessions.pop(session_id, None)
        return {"code": "auth-code", "state": "st"}

    monkeypatch.setattr(auth_mod, "_xai_wait_for_callback", _wait_then_cancel)

    def _must_not_persist(*args, **kwargs):
        raise AssertionError("tokens must not be persisted for a cancelled session")

    monkeypatch.setattr(auth_mod, "_save_xai_oauth_tokens", _must_not_persist)
    monkeypatch.setattr(ws, "_add_xai_oauth_pool_entry", _must_not_persist)

    # Should return cleanly without raising and without persisting.
    ws._xai_loopback_worker(session_id)
    assert session_id not in ws._oauth_sessions


def test_cancel_loopback_session_shuts_down_callback_server():
    """Cancelling a loopback session must free the bound callback port now."""
    from hermes_cli import web_server as ws

    shutdown_calls = {"shutdown": 0, "close": 0, "join": 0}

    class _FakeServer:
        def shutdown(self):
            shutdown_calls["shutdown"] += 1

        def server_close(self):
            shutdown_calls["close"] += 1

    class _FakeThread:
        def join(self, timeout=None):
            shutdown_calls["join"] += 1

    # callback_result is the dict the worker's _xai_wait_for_callback polls.
    callback_result = {"code": None, "error": None}
    session_id = "xai-loopback-cancel-shutdown-test"
    ws._oauth_sessions[session_id] = {
        "session_id": session_id,
        "provider": "xai-oauth",
        "flow": "loopback",
        "created_at": time.time(),
        "status": "pending",
        "server": _FakeServer(),
        "thread": _FakeThread(),
        "callback_result": callback_result,
    }

    try:
        resp = client.delete(
            f"/api/providers/oauth/sessions/{session_id}", headers=_headers()
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        assert shutdown_calls == {"shutdown": 1, "close": 1, "join": 1}
        # The waiting worker must be signalled so it returns promptly instead
        # of spinning until the timeout.
        assert callback_result["error"] == "cancelled"
        assert session_id not in ws._oauth_sessions
    finally:
        ws._oauth_sessions.pop(session_id, None)


def test_unknown_pkce_provider_rejected_cleanly():
    """A future PKCE provider without an explicit branch must NOT silently route to Anthropic.

    Simulates a hypothetical catalog entry with ``flow: "pkce"`` and an
    id other than "anthropic". The dispatcher should fall through past
    the pkce branch (now gated on provider_id) and the device_code
    branch, then hit "Unsupported flow" — proving the bug class is
    structurally prevented.
    """
    from hermes_cli import web_server as ws

    # Inject a hypothetical catalog entry that's pkce-flagged but isn't
    # anthropic. This shape mirrors what would happen if a developer
    # added a new provider entry without remembering to wire up its
    # start function.
    fake_entry = {
        "id": "hypothetical-pkce-provider",
        "name": "Hypothetical PKCE Provider",
        "flow": "pkce",
        "cli_command": "hermes auth add hypothetical-pkce-provider",
        "docs_url": "https://example.com",
        "status_fn": None,
    }
    original_catalog = ws._OAUTH_PROVIDER_CATALOG
    try:
        ws._OAUTH_PROVIDER_CATALOG = original_catalog + (fake_entry,)
        resp = client.post(
            "/api/providers/oauth/hypothetical-pkce-provider/start",
            headers=_headers(),
        )
    finally:
        ws._OAUTH_PROVIDER_CATALOG = original_catalog

    # Either 400 "Unsupported flow" (the explicit fall-through) or any
    # 4xx — what we MUST NOT see is a 200 with claude.ai in the body.
    assert resp.status_code >= 400, resp.text
    assert "claude.ai" not in resp.text.lower()
