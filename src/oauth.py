"""
oauth.py — OAuth 2.0 PKCE flow for GGG's Path of Exile API.

Public client flow (no client_secret):
  1. Generate code_verifier + code_challenge (S256)
  2. Open browser to /oauth/authorize with PKCE params
  3. Catch callback on localhost HTTP server
  4. Exchange auth code for tokens at /oauth/token
  5. Store tokens locally, auto-refresh before expiry

Requires: account:stashes + account:profile scopes.
Client ID must be registered with GGG (email oauth@grindinggear.com).
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

# GGG OAuth endpoints
OAUTH_AUTHORIZE_URL = "https://www.pathofexile.com/oauth/authorize"
OAUTH_TOKEN_URL = "https://www.pathofexile.com/oauth/token"

# Must be registered with GGG — placeholder until approved
CLIENT_ID = os.environ.get("POE_OAUTH_CLIENT_ID", "lama")

SCOPES = "account:stashes account:profile"
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT_RANGE = (8951, 8960)  # Try ports in this range

TOKEN_DIR = Path(os.path.expanduser("~")) / ".poe2-price-overlay"
TOKEN_FILE = TOKEN_DIR / "oauth_tokens.json"

# Refresh 5 minutes before expiry
REFRESH_MARGIN_SECONDS = 300


def _b64url(data: bytes) -> str:
    """Base64url-encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _generate_pkce():
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(48)  # 64 chars
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


class OAuthManager:
    """Manages OAuth tokens for GGG's POE API."""

    def __init__(self):
        self._tokens: Optional[dict] = None
        self._lock = threading.Lock()
        self._load_tokens()

    @property
    def connected(self) -> bool:
        return self._tokens is not None and "access_token" in self._tokens

    @property
    def account_name(self) -> Optional[str]:
        if not self._tokens:
            return None
        return self._tokens.get("account_name")

    def _load_tokens(self):
        """Load stored tokens from disk."""
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE) as f:
                    self._tokens = json.load(f)
                logger.info(f"OAuth tokens loaded (account: {self.account_name})")
            except Exception as e:
                logger.warning(f"Failed to load OAuth tokens: {e}")
                self._tokens = None

    def _save_tokens(self):
        """Persist tokens to disk."""
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(TOKEN_FILE, "w") as f:
                json.dump(self._tokens, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save OAuth tokens: {e}")

    def _clear_tokens(self):
        """Remove stored tokens."""
        self._tokens = None
        try:
            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete token file: {e}")

    def get_headers(self) -> Optional[dict]:
        """Return Authorization headers, auto-refreshing if needed.

        Returns None if not connected or refresh fails.
        """
        with self._lock:
            if not self.connected:
                return None

            # Check if token needs refresh
            expires_at = self._tokens.get("expires_at", 0)
            if time.time() > expires_at - REFRESH_MARGIN_SECONDS:
                if not self._refresh():
                    return None

            return {
                "Authorization": f"Bearer {self._tokens['access_token']}",
                "User-Agent": "LAMA/1.0",
            }

    def authorize(self) -> dict:
        """Start OAuth PKCE flow. Opens browser, waits for callback.

        Returns:
            {"connected": True, "account_name": "..."} on success
            {"error": "..."} on failure
        """
        verifier, challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)

        # Find an available port for the callback server
        redirect_port = None
        server = None
        for port in range(*REDIRECT_PORT_RANGE):
            try:
                server = _CallbackServer((REDIRECT_HOST, port), _CallbackHandler)
                redirect_port = port
                break
            except OSError:
                continue

        if not server:
            return {"error": "Could not find an available port for OAuth callback"}

        redirect_uri = f"http://{REDIRECT_HOST}:{redirect_port}/oauth/callback"

        # Build authorization URL
        params = {
            "client_id": CLIENT_ID,
            "response_type": "code",
            "scope": SCOPES,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        auth_url = f"{OAUTH_AUTHORIZE_URL}?{urlencode(params)}"

        # Open browser
        import webbrowser
        webbrowser.open(auth_url)
        logger.info("OAuth: browser opened, waiting for callback...")

        # Wait for callback (timeout 120s)
        server.timeout = 120
        server.handle_request()

        callback_params = server.callback_params
        server.server_close()

        if not callback_params:
            return {"error": "OAuth callback timed out or was not received"}

        # Verify state
        if callback_params.get("state", [None])[0] != state:
            return {"error": "OAuth state mismatch — possible CSRF"}

        if "error" in callback_params:
            error = callback_params["error"][0]
            desc = callback_params.get("error_description", [""])[0]
            return {"error": f"OAuth denied: {error} — {desc}"}

        code = callback_params.get("code", [None])[0]
        if not code:
            return {"error": "No authorization code received"}

        # Exchange code for tokens
        return self._exchange_code(code, verifier, redirect_uri)

    def _exchange_code(self, code: str, verifier: str, redirect_uri: str) -> dict:
        """Exchange authorization code for access + refresh tokens."""
        try:
            resp = requests.post(
                OAUTH_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "code_verifier": verifier,
                },
                headers={"User-Agent": "LAMA/1.0"},
                timeout=30,
            )

            if resp.status_code != 200:
                return {"error": f"Token exchange failed: HTTP {resp.status_code} — {resp.text[:200]}"}

            data = resp.json()
            self._tokens = {
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "token_type": data.get("token_type", "bearer"),
                "expires_at": time.time() + data.get("expires_in", 36000),
                "scope": data.get("scope", SCOPES),
            }

            # Fetch account name
            self._fetch_account_name()
            self._save_tokens()

            logger.info(f"OAuth: connected as {self.account_name}")
            return {"connected": True, "account_name": self.account_name}

        except Exception as e:
            logger.error(f"OAuth token exchange failed: {e}")
            return {"error": f"Token exchange failed: {e}"}

    def _refresh(self) -> bool:
        """Refresh access token using refresh_token. Returns True on success."""
        refresh_token = self._tokens.get("refresh_token")
        if not refresh_token:
            logger.warning("OAuth: no refresh token available")
            self._clear_tokens()
            return False

        try:
            resp = requests.post(
                OAUTH_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"User-Agent": "LAMA/1.0"},
                timeout=30,
            )

            if resp.status_code != 200:
                logger.error(f"OAuth refresh failed: HTTP {resp.status_code}")
                self._clear_tokens()
                return False

            data = resp.json()
            self._tokens["access_token"] = data["access_token"]
            self._tokens["expires_at"] = time.time() + data.get("expires_in", 36000)
            if "refresh_token" in data:
                self._tokens["refresh_token"] = data["refresh_token"]

            self._save_tokens()
            logger.info("OAuth: token refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"OAuth refresh failed: {e}")
            self._clear_tokens()
            return False

    def _fetch_account_name(self):
        """Fetch account name from /api/profile."""
        if not self._tokens or "access_token" not in self._tokens:
            return
        try:
            resp = requests.get(
                "https://api.pathofexile.com/profile",
                headers={
                    "Authorization": f"Bearer {self._tokens['access_token']}",
                    "User-Agent": "LAMA/1.0",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._tokens["account_name"] = data.get("name", "Unknown")
        except Exception as e:
            logger.warning(f"Failed to fetch account name: {e}")

    def disconnect(self):
        """Revoke tokens and clear stored data."""
        self._clear_tokens()
        logger.info("OAuth: disconnected")

    def get_status(self) -> dict:
        """Return current OAuth status for the API."""
        return {
            "connected": self.connected,
            "account_name": self.account_name,
        }


class _CallbackServer(HTTPServer):
    """Temporary HTTP server to catch the OAuth redirect."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.callback_params: Optional[dict] = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback request."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/oauth/callback":
            self.server.callback_params = parse_qs(parsed.query)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
            <html><body style="background:#0d0b08;color:#c4a456;font-family:sans-serif;
            display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
            <div style="text-align:center">
            <h1>LAMA Connected!</h1>
            <p>You can close this tab and return to the dashboard.</p>
            </div></body></html>
            """)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass
