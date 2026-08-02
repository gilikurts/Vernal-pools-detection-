import time
import requests

from config import CDSE_USERNAME, CDSE_PASSWORD, TOKEN_URL


class CDSEAuth:
    """Manages CDSE OAuth2 access/refresh tokens (password grant)."""

    CLIENT_ID = "cdse-public"

    def __init__(self, username=None, password=None):
        self.username = username or CDSE_USERNAME
        self.password = password or CDSE_PASSWORD
        if not self.username or not self.password:
            raise RuntimeError(
                "CDSE_USERNAME / CDSE_PASSWORD not set. "
                "Copy .env.example to .env and fill in your credentials."
            )
        self._access_token = None
        self._refresh_token = None
        self._expires_at = 0

    def _request(self, grant_type, **extra):
        data = {"client_id": self.CLIENT_ID, "grant_type": grant_type, **extra}
        resp = requests.post(TOKEN_URL, data=data, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = payload["access_token"]
        self._refresh_token = payload["refresh_token"]
        # refresh a bit early to avoid using an expired token mid-request
        self._expires_at = time.time() + payload["expires_in"] - 30
        return self._access_token

    def get_token(self):
        if self._access_token and time.time() < self._expires_at:
            return self._access_token

        if self._refresh_token:
            try:
                return self._request("refresh_token", refresh_token=self._refresh_token)
            except requests.HTTPError:
                pass  # refresh token expired too -> fall back to password grant

        return self._request(
            "password", username=self.username, password=self.password
        )
