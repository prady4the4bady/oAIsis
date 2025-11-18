"""Opus workflow logging utility."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)


class OpusLogger:
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str],
        workflow_id: Optional[str] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.workflow_id = workflow_id or os.getenv("OPUS_WORKFLOW_ID") or ""

    def log_action(
        self,
        action: str,
        status: str = "success",
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.api_url:
            LOGGER.debug("Opus logging disabled; no API URL configured")
            return

        payload: Dict[str, Any] = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "action": action,
            "status": status,
            "data": data or {},
        }
        if self.workflow_id:
            payload.setdefault("data", {})
            payload["data"]["workflow_id"] = self.workflow_id
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
        except requests.HTTPError as err:
            LOGGER.warning("Opus logging failed: %s", err.response.text if err.response else err)
        except requests.RequestException as err:
            LOGGER.warning("Opus logging unavailable: %s", err)

        LOGGER.debug("Opus log sent: %s", json.dumps(payload))
