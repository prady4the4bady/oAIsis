"""Google Maps routing helper."""

from __future__ import annotations

import html
import re
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import List

import requests


GOOGLE_MAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
EMBED_BASE = "https://www.google.com/maps/embed/v1/directions"


@dataclass
class RouteStep:
    instruction: str
    distance_text: str
    duration_text: str


@dataclass
class RoutePlan:
    destination: str
    summary: str
    steps: List[RouteStep]
    embed_url: str


class GoogleMapsService:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY missing.")
        self.api_key = api_key

    def plan_route(self, origin: str, destination: str, mode: str = "walking") -> RoutePlan:
        if not origin or not destination:
            raise ValueError("Both origin and destination are required.")

        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": self.api_key,
        }
        resp = requests.get(GOOGLE_MAPS_DIRECTIONS_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")
        if status != "OK":
            message = payload.get("error_message") or status or "Unknown error"
            raise RuntimeError(f"Directions API error: {message}")

        route = payload["routes"][0]
        leg = route["legs"][0]
        steps: List[RouteStep] = []
        for step in leg["steps"]:
            instruction = _strip_html(step.get("html_instructions", ""))
            steps.append(
                RouteStep(
                    instruction=instruction,
                    distance_text=step["distance"]["text"],
                    duration_text=step["duration"]["text"],
                )
            )

        embed_url = self._build_embed_url(origin, destination, mode)
        summary = route.get("summary") or destination
        return RoutePlan(
            destination=leg.get("end_address", destination),
            summary=summary,
            steps=steps,
            embed_url=embed_url,
        )

    def _build_embed_url(self, origin: str, destination: str, mode: str) -> str:
        return (
            f"{EMBED_BASE}?key={self.api_key}"
            f"&origin={quote_plus(origin)}"
            f"&destination={quote_plus(destination)}"
            f"&mode={mode}&maptype=satellite"
        )


def _strip_html(value: str) -> str:
    clean = html.unescape(value)
    return re.sub(r"<[^>]+>", "", clean).strip()
