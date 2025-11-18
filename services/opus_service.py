"""Opus workflow logging + execution helpers."""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import requests
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Parsed subset of workflow outputs with the raw payload attached."""

    navigation_instruction: Optional[str] = None
    navigation_decision: Optional[str] = None
    guidance_text: Optional[str] = None
    priority_level: Optional[str] = None
    input_summary: Optional[str] = None
    memory_action: Optional[str] = None
    error_message: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=lambda: {})


class OpusLogger:
    def __init__(
        self,
        api_url: str,
        api_key: Optional[str],
        workflow_id: Optional[str] = None,
        workflow_url: Optional[str] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/") if api_url else ""
        self.api_key = api_key
        self.workflow_id = workflow_id or os.getenv("OPUS_WORKFLOW_ID") or ""
        url_from_env = workflow_url or os.getenv("OPUS_WORKFLOW_URL") or self.api_url
        self.workflow_url = url_from_env.rstrip("/") if url_from_env else ""

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
            payload["workflow_id"] = self.workflow_id
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

    def run_workflow(
        self,
        *,
        scene_description: str,
        hazards: List[str],
        objects: List[str],
        sensor_signals: Dict[str, Any],
        image: Optional[Image.Image] = None,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Optional[WorkflowResult]:
        """Invoke the configured Opus workflow and return parsed highlights.

        The payload structure mirrors the workflow builder inputs shared by the user.
        When no workflow URL/ID is configured the method silently returns ``None``.
        """

        if not self.workflow_url or not self.workflow_id:
            LOGGER.debug("Opus workflow disabled; missing URL or workflow ID")
            return None

        LOGGER.info("run_workflow TRIGGERED for workflow_id=%s", self.workflow_id)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-service-key"] = self.api_key

        inputs: Dict[str, Any] = {
            "hazard_list": hazards,
            "object_list": objects,
            "scene_description": scene_description or "",
            "system_sensor_signals": sensor_signals,
        }
        if image is not None:
            inputs["camera_frame_file"] = self._image_to_data_url(image)
        if extra_inputs:
            inputs.update(extra_inputs)

        payload: Dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "inputs": inputs,
        }

        LOGGER.debug("DEBUG: sending to %s", self.workflow_url)
        LOGGER.debug("DEBUG HEADERS: %s", headers)
        LOGGER.debug("DEBUG PAYLOAD: %s", json.dumps(payload))

        try:
            resp = requests.post(self.workflow_url, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            body = resp.json()
        except requests.HTTPError as err:
            detail = err.response.text if err.response else str(err)
            LOGGER.warning("Opus workflow HTTP error: %s", detail)
            return None
        except requests.RequestException as err:
            LOGGER.warning("Opus workflow unreachable: %s", err)
            return None
        except ValueError as err:
            LOGGER.warning("Opus workflow returned invalid JSON: %s", err)
            return None

        if not isinstance(body, dict):
            LOGGER.warning("Unexpected Opus response shape: %s", body)
            return None

        body_dict: Dict[str, Any] = cast(Dict[str, Any], body)
        outputs = body_dict.get("outputs")
        if isinstance(outputs, dict):
            parsed = cast(Dict[str, Any], outputs)
        else:
            parsed = body_dict

        result = WorkflowResult(
            navigation_instruction=self._extract_first(parsed, ["navigation_instruction", "instruction_text"]),
            navigation_decision=self._extract_first(parsed, ["navigation_decision", "navigationDecision"]),
            guidance_text=self._extract_first(parsed, ["guidance_text", "output_guidance"]),
            priority_level=self._extract_first(parsed, ["priority_level"]),
            input_summary=self._extract_first(parsed, ["input_summary"]),
            memory_action=self._extract_first(parsed, ["memory_action"]),
            error_message=self._extract_first(parsed, ["error_message"]),
            raw=parsed,
        )
        LOGGER.debug("Opus workflow response: %s", json.dumps(parsed))
        return result

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_first(payload: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
