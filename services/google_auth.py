from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

LOGGER = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def load_service_account_credentials(scopes: Optional[List[str]] = None) -> Optional[service_account.Credentials]:
    """Load Google service-account credentials from env vars."""

    scopes = scopes or DEFAULT_SCOPES
    json_env_candidates = [
        ("GOOGLE_SERVICE_ACCOUNT_JSON", os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")),
        ("GOOGLE_TTS_CREDENTIALS_JSON", os.getenv("GOOGLE_TTS_CREDENTIALS_JSON")),
    ]
    for env_name, blob in json_env_candidates:
        if blob:
            try:
                info: Dict[str, Any] = json.loads(blob)
            except json.JSONDecodeError:
                LOGGER.error("Invalid JSON in %s; please paste a valid service-account blob.", env_name)
                return None
            return service_account.Credentials.from_service_account_info(info, scopes=scopes)  # type: ignore

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        if not os.path.exists(credentials_path):
            LOGGER.error("Google credentials file '%s' not found.", credentials_path)
            return None
        return service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)  # type: ignore

    return None


def build_authorized_session(scopes: Optional[List[str]] = None) -> Optional[AuthorizedSession]:
    """Create an AuthorizedSession using any configured service-account credentials."""

    credentials = load_service_account_credentials(scopes=scopes)
    if not credentials:
        LOGGER.warning("Google service-account credentials missing; Vertex APIs unavailable.")
        return None
    return AuthorizedSession(credentials)
