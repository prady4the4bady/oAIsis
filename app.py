from __future__ import annotations

import os

if os.name == "nt":
    os.environ.setdefault("PYTHONASYNCIO_USE_SELECTOR_POLICY", "1")

import asyncio
import json
import logging
import platform
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast, no_type_check

import av
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from numpy.typing import NDArray
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer as _webrtc_streamer,  # type: ignore[misc]
)
from streamlit.delta_generator import DeltaGenerator
from services.gemini_service import GeminiService
from services.open_vision import OpenVisionResult, OpenVisionService
from services.opus_service import OpusLogger, WorkflowResult
from services.qdrant_service import QdrantMemory
from services.tts_service import TTSService
from utils.camera_utils import bgr_frame_to_pil, generate_emergency_tone
from utils.embeddings import EmbeddingClient

load_dotenv(override=True)

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

FRAME_INTERVAL_SECONDS = float(os.getenv("ANALYSIS_INTERVAL_SECONDS", "2"))
DEFAULT_MEMORY_ENABLED = (os.getenv("ENABLE_MEMORY", "true").lower() == "true")

LOGGER = logging.getLogger(__name__)
FrameArray = NDArray[np.uint8]
EmbeddingArray = NDArray[Any]

WebRtcStreamerFunc = Callable[..., WebRtcStreamerContext[VideoProcessorBase, Any]]
webrtc_streamer = cast(WebRtcStreamerFunc, _webrtc_streamer)
EmbeddingArray = NDArray[Any]


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def build_rtc_configuration() -> RTCConfiguration:
    default_stun = [
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
        "stun:stun3.l.google.com:19302",
        "stun:stun4.l.google.com:19302",
    ]
    disable_stun = _env_flag("GUIDELY_DISABLE_STUN")
    stun_urls = [url.strip() for url in os.getenv("GUIDELY_STUN_URLS", "").split(",") if url.strip()]
    ice_servers: List[Dict[str, object]] = []
    if not disable_stun:
        ice_servers.append({"urls": stun_urls or default_stun})

    turn_urls_env = os.getenv("GUIDELY_TURN_URLS") or os.getenv("GUIDELY_TURN_URL") or ""
    turn_urls = [url.strip() for url in turn_urls_env.split(",") if url.strip()]
    if turn_urls:
        turn_user = os.getenv("GUIDELY_TURN_USERNAME")
        turn_pass = os.getenv("GUIDELY_TURN_PASSWORD")
        for url in turn_urls:
            turn_entry: Dict[str, object] = {"urls": [url]}
            if turn_user and turn_pass:
                turn_entry["username"] = turn_user
                turn_entry["credential"] = turn_pass
            ice_servers.append(turn_entry)

    rtc_config: Dict[str, object] = {"iceServers": ice_servers}
    if _env_flag("GUIDELY_FORCE_TURN"):
        rtc_config["iceTransportPolicy"] = "relay"

    return cast(RTCConfiguration, rtc_config)

MODE_PROMPTS = {
    "navigation": (
        "You are assisting a blind person navigate. Detect every obstacle (doors, stairs, vehicles, people, "
        "moving objects) within 10 meters. Estimate their distance in meters and relative bearing (left/right/ahead/behind). "
        "Return ONLY valid JSON with this schema: {"
        "\"guidance\": \"free-form narration\", "
        "\"movement\": {\"suggestion\": \"move_forward|turn_left|turn_right|stop|step_back\", \"reason\": \"short clause\"}, "
        "\"objects\": [{\"label\": \"object name\", \"distance_m\": float, \"direction\": \"left|right|ahead|behind\"}] }. "
        "guidance must be concise (<= 2 sentences) and written in the requested language. "
        "movement.suggestion must always be one of the allowed values."
    ),
    "ocr": (
        "Extract ALL visible text from this image. Read signs, labels, documents, menus, anything with text. "
        "Format the output clearly. If multiple text sections, separate them with line breaks."
    ),
    "describe": (
        "Describe this scene in rich detail for a visually impaired person. Include: surroundings, people, objects, "
        "colors, atmosphere. Make it vivid but clear."
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in clear, plain English.",
    "ar": "Respond in Modern Standard Arabic using short sentences.",
    "hi": "Respond in simple Hindi using Devanagari script.",
}

LANGUAGE_LABELS = {
    "English": "en",
    "العربية": "ar",
    "हिन्दी": "hi",
}

HAZARD_KEYWORDS = {
    "exposed wiring": "exposed wiring",
    "broken glass": "broken glass",
    "chemical": "chemical spill",
    "spill": "chemical spill",
    "smoke": "smoke",
    "fire": "fire",
    "debris": "debris",
    "hole": "ground hole",
    "stairs": "stairs",
    "stair": "stairs",
    "vehicle": "vehicle",
    "car": "vehicle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "crowd": "crowd",
    "traffic": "traffic",
}

HAZARD_OBJECT_LABELS = {
    "car": "vehicle",
    "vehicle": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "stairs": "stairs",
    "stair": "stairs",
}

OBJECT_KEYWORDS = {
    "person",
    "people",
    "pedestrian",
    "car",
    "vehicle",
    "bicycle",
    "traffic light",
    "sign",
    "bench",
    "tree",
    "dog",
    "cat",
    "door",
    "stairs",
    "window",
    "bus",
    "truck",
}


@no_type_check
def render_guidance_panel(
    guidance_placeholder: DeltaGenerator, recognized_placeholder: DeltaGenerator
) -> None:
    """Render the textual guidance, workflow output, and recognized locations."""

    guidance_text = st.session_state.get("guidance_text") or "Awaiting camera input…"
    movement = cast(Optional[MovementAdvice], st.session_state.get("movement_instruction"))
    detected_objects = cast(List[DetectedObject], st.session_state.get("detected_objects") or [])
    workflow_instruction = st.session_state.get("workflow_instruction")
    workflow_guidance = st.session_state.get("workflow_guidance")
    workflow_priority = st.session_state.get("workflow_priority")
    workflow_error = st.session_state.get("workflow_error")
    recognized_locations = cast(List[str], st.session_state.get("recognized_locations") or [])

    guidance_panel = guidance_placeholder.container()
    guidance_panel.subheader("Latest guidance")
    guidance_panel.write(guidance_text)

    if movement:
        move_reason = f"Reason: {movement.reason}" if movement.reason else ""
        guidance_panel.info(f"Suggested movement: {movement.suggestion}. {move_reason}".strip())

    if workflow_instruction or workflow_guidance or workflow_priority or workflow_error:
        guidance_panel.divider()
        guidance_panel.caption("Workflow insights")
        if workflow_instruction:
            guidance_panel.write(f"Instruction: {workflow_instruction}")
        if workflow_guidance:
            guidance_panel.write(f"Guidance: {workflow_guidance}")
        if workflow_priority:
            guidance_panel.write(f"Priority: {workflow_priority}")
        if workflow_error:
            guidance_panel.warning(f"Workflow error: {workflow_error}")

    if detected_objects:
        guidance_panel.divider()
        guidance_panel.caption("Detected objects")
        for obj in detected_objects[:6]:
            label = obj.label
            distance = f"{obj.distance_m:.1f} m" if obj.distance_m is not None else "?"
            direction = obj.direction or "unspecified"
            guidance_panel.write(f"• {label} ({direction}, {distance})")

    recognized_panel = recognized_placeholder.container()
    recognized_panel.subheader("Recognized locations")
    if recognized_locations:
        for location in recognized_locations[-5:][::-1]:
            recognized_panel.write(f"📍 {location}")
    else:
        recognized_panel.caption("No saved locations yet.")


@dataclass
class MovementAdvice:
    suggestion: str
    reason: Optional[str] = None


@dataclass
class DetectedObject:
    label: str
    distance_m: Optional[float] = None
    direction: Optional[str] = None


@dataclass
class AnalysisResult:
    guidance: str
    recognized_location: Optional[str]
    embedding: Optional[EmbeddingArray]
    objects: List[DetectedObject] = field(default_factory=lambda: [])
    movement: Optional[MovementAdvice] = None
    provider: str = "gemini"
    mode: str = "navigation"
    language: str = "en"
    match_score: Optional[float] = None
    workflow: Optional[WorkflowResult] = None


class RealtimeAnalyzer:
    """Coordinates open-source vision, Gemini fallback, memory lookup, and Opus logging."""

    def __init__(
        self,
        gemini: GeminiService,
        embeddings: EmbeddingClient,
        opus: OpusLogger,
        open_vision: OpenVisionService,
        qdrant: Optional[QdrantMemory],
    ) -> None:
        self._gemini = gemini
        self._embeddings = embeddings
        self._opus = opus
        self._open_vision = open_vision
        self._qdrant = qdrant

    def analyze(
        self,
        image: Image.Image,
        mode: str,
        language: str,
        memory_enabled: bool,
    ) -> AnalysisResult:
        self._opus.log_action(
            action="scene_analysis_started",
            data={"mode": mode, "language": language},
        )
        try:
            guidance, objects, movement, provider = self._run_vision_stack(
                image=image,
                mode=mode,
                language=language,
            )
            embedding_vector = self._embeddings.embed_text(guidance)
            recognized_location, match_score = self._maybe_match_location(
                embedding_vector,
                memory_enabled,
            )
            result = AnalysisResult(
                guidance=guidance,
                recognized_location=recognized_location,
                embedding=embedding_vector,
                objects=objects,
                movement=movement,
                provider=provider,
                mode=mode,
                language=language,
                match_score=match_score,
            )
            workflow_result = self._maybe_run_workflow(
                image=image,
                guidance=guidance,
                objects=objects,
                movement=movement,
                recognized_location=recognized_location,
                match_score=match_score,
            )
            result.workflow = workflow_result
            self._log_success(result)
            return result
        except Exception as exc:  # noqa: BLE001
            self._opus.log_action(
                action="scene_analysis_completed",
                status="error",
                data={
                    "mode": mode,
                    "language": language,
                    "error": str(exc),
                },
            )
            raise

    def _run_vision_stack(
        self,
        image: Image.Image,
        mode: str,
        language: str,
    ) -> Tuple[str, List[DetectedObject], Optional[MovementAdvice], str]:
        open_result = self._try_open_vision(image=image, mode=mode, language=language)
        if open_result:
            return self._from_open_result(open_result)
        return self._run_gemini(image=image, mode=mode, language=language)

    def _try_open_vision(
        self,
        image: Image.Image,
        mode: str,
        language: str,
    ) -> Optional[OpenVisionResult]:
        if not self._open_vision.available:
            return None
        try:
            return self._open_vision.analyze(image=image, mode=mode, language=language)
        except Exception as exc:  # pragma: no cover - best-effort fallback
            LOGGER.warning("Open vision analysis failed: %s", exc)
            self._opus.log_action(
                action="open_vision_failed",
                status="error",
                data={"error": str(exc)},
            )
            return None

    def _from_open_result(
        self,
        open_result: OpenVisionResult,
    ) -> Tuple[str, List[DetectedObject], Optional[MovementAdvice], str]:
        detected_objects: List[DetectedObject] = []
        for entry in open_result.objects:
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                continue
            detected_objects.append(
                DetectedObject(
                    label=label.strip(),
                    distance_m=cast(Optional[float], entry.get("distance_m")),
                    direction=entry.get("direction"),
                )
            )
        movement_dict = open_result.movement or {}
        movement = None
        suggestion = movement_dict.get("suggestion")
        if isinstance(suggestion, str) and suggestion.strip():
            reason = movement_dict.get("reason")
            movement = MovementAdvice(
                suggestion=suggestion.strip(),
                reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
            )
        return open_result.guidance, detected_objects, movement, "open_vision"

    def _run_gemini(
        self,
        image: Image.Image,
        mode: str,
        language: str,
    ) -> Tuple[str, List[DetectedObject], Optional[MovementAdvice], str]:
        prompt = MODE_PROMPTS[mode]
        language_note = LANGUAGE_INSTRUCTIONS.get(language, "Respond clearly.")
        response_mime = "application/json" if mode == "navigation" else None
        raw_response = self._gemini.analyze_image(
            image=image,
            user_prompt=f"{prompt}\n\n{language_note}",
            response_mime_type=response_mime,
        )
        guidance_text = raw_response
        detected_objects: List[DetectedObject] = []
        movement: Optional[MovementAdvice] = None
        if mode == "navigation":
            guidance_text, detected_objects, movement = parse_navigation_response(raw_response)
        if not guidance_text:
            raise ValueError("Gemini did not return guidance.")
        return guidance_text, detected_objects, movement, "gemini"

    def _maybe_match_location(
        self,
        embedding_vector: Optional[EmbeddingArray],
        memory_enabled: bool,
    ) -> Tuple[Optional[str], Optional[float]]:
        if not memory_enabled or self._qdrant is None or embedding_vector is None:
            return None, None
        match = self._qdrant.search_scene(embedding_vector)
        if not match:
            return None, None
        self._opus.log_action(
            action="location_recognized",
            data={"location": match.location_name, "score": match.score},
        )
        return match.location_name, match.score

    def _log_success(self, result: AnalysisResult) -> None:
        movement_dict: Optional[Dict[str, Optional[str]]] = (
            {
                "suggestion": result.movement.suggestion,
                "reason": result.movement.reason,
            }
            if result.movement
            else None
        )
        data: Dict[str, Any] = {
            "mode": result.mode,
            "language": result.language,
            "provider": result.provider,
            "recognized_location": result.recognized_location,
            "match_score": result.match_score,
            "objects": [
                {
                    "label": obj.label,
                    "distance_m": obj.distance_m,
                    "direction": obj.direction,
                }
                for obj in result.objects
            ],
            "movement": movement_dict,
            "guidance_excerpt": (result.guidance[:480] + "…")
            if len(result.guidance) > 480
            else result.guidance,
        }
        self._opus.log_action(
            action="scene_analysis_completed",
            data=data,
        )

    def _maybe_run_workflow(
        self,
        *,
        image: Image.Image,
        guidance: str,
        objects: List[DetectedObject],
        movement: Optional[MovementAdvice],
        recognized_location: Optional[str],
        match_score: Optional[float],
    ) -> Optional[WorkflowResult]:
        try:
            hazard_names, hazard_objects = self._extract_hazards(guidance, objects)
            object_names = self._object_names(objects, guidance)
            sensor_signals, derived = self._build_sensor_context(movement, objects)
            object_positions = self._serialize_positions(objects)
            hazard_positions = self._serialize_positions(hazard_objects)
            extra_inputs: Dict[str, Any] = {
                "speed": derived["speed"],
                "is_moving": derived["is_moving"],
                "hazard_names": hazard_names,
                "object_names": object_names,
                "ambient_light": sensor_signals["ambient_light"],
                "obstacle_names": hazard_names,
                "object_positions": object_positions,
                "obstacle_positions": hazard_positions,
                "hazard_positions": hazard_positions,
                "orientation_angle": derived["orientation_angle"],
                "user_orientation": derived["orientation_angle"],
                "user_movement_status": "moving" if derived["is_moving"] else "stationary",
                "place_name": recognized_location or "",
                "is_familiar_place": bool(recognized_location),
                "match_score": match_score,
                "gyroscope_readings": sensor_signals["gyroscope"],
                "accelerometer_readings": sensor_signals["accelerometer"],
                "ambient_light_reading": sensor_signals["ambient_light"],
            }
            return self._opus.run_workflow(
                scene_description=guidance,
                hazards=hazard_names,
                objects=object_names,
                sensor_signals=sensor_signals,
                image=image,
                extra_inputs=extra_inputs,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Opus workflow unavailable: %s", exc)
            return None
    def _extract_hazards(
        self,
        guidance: str,
        objects: List[DetectedObject],
    ) -> Tuple[List[str], List[DetectedObject]]:
        hazards: set[str] = set()
        hazard_objects: List[DetectedObject] = []
        lowered = guidance.lower()
        for keyword, label in HAZARD_KEYWORDS.items():
            if keyword in lowered:
                hazards.add(label)
        for obj in objects:
            mapped = HAZARD_OBJECT_LABELS.get(obj.label.lower())
            if mapped:
                hazards.add(mapped)
                hazard_objects.append(obj)
        return sorted(hazards), hazard_objects

    def _object_names(self, objects: List[DetectedObject], guidance: str) -> List[str]:
        if objects:
            names: Set[str] = {obj.label for obj in objects if obj.label}
        else:
            names = set()
            lowered = guidance.lower()
            for keyword in OBJECT_KEYWORDS:
                if keyword in lowered:
                    names.add(keyword)
        return sorted(names) or ["scene"]

    def _build_sensor_context(
        self,
        movement: Optional[MovementAdvice],
        objects: List[DetectedObject],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        suggestion = (movement.suggestion if movement else "").lower()
        is_moving = suggestion not in {"stop", "hold", ""}
        speed_map = {
            "move_forward": 1.2,
            "turn_left": 0.3,
            "turn_right": 0.3,
            "step_back": 0.2,
            "stop": 0.0,
        }
        speed = speed_map.get(suggestion, 0.5 if is_moving else 0.0)
        gyro = [0.0, 0.0, 0.0]
        accel = [0.0, 0.0, 9.81]
        if suggestion == "turn_left":
            gyro[2] = -0.45
        elif suggestion == "turn_right":
            gyro[2] = 0.45
        elif suggestion == "move_forward":
            accel[1] = 0.12
        orientation_angle = self._direction_to_angle(movement, objects)
        try:
            ambient_value = float(os.getenv("GUIDELY_AMBIENT_LIGHT", "350.0"))
        except ValueError:
            ambient_value = 350.0

        sensor_signals: Dict[str, Any] = {
            "gyroscope": gyro,
            "accelerometer": accel,
            "ambient_light": ambient_value,
        }
        derived: Dict[str, Any] = {
            "speed": round(speed, 2),
            "is_moving": is_moving,
            "orientation_angle": orientation_angle,
        }
        return sensor_signals, derived

    def _serialize_positions(self, objects: List[DetectedObject]) -> List[Dict[str, Any]]:
        positions: List[Dict[str, Any]] = []
        for obj in objects:
            entry: Dict[str, Any] = {"label": obj.label}
            if obj.distance_m is not None:
                entry["distance_m"] = obj.distance_m
            if obj.direction:
                entry["direction"] = obj.direction
                coords = self._direction_to_coords(obj.direction)
                if coords:
                    entry["x"], entry["y"] = coords
                entry["orientation_angle"] = self._direction_label_to_angle(obj.direction)
            positions.append(entry)
        return positions

    def _direction_to_angle(
        self,
        movement: Optional[MovementAdvice],
        objects: List[DetectedObject],
    ) -> float:
        if movement and movement.suggestion in {"turn_left", "turn_right", "step_back"}:
            cleaned = movement.suggestion.replace("turn_", "")
            if cleaned == "step_back":
                cleaned = "back"
            return self._direction_label_to_angle(cleaned)
        for obj in objects:
            if obj.direction:
                return self._direction_label_to_angle(obj.direction)
        return 0.0

    @staticmethod
    def _direction_label_to_angle(direction: Optional[str]) -> float:
        mapping = {
            "ahead": 0.0,
            "forward": 0.0,
            "left": -90.0,
            "right": 90.0,
            "behind": 180.0,
            "back": 180.0,
        }
        if not direction:
            return 0.0
        return mapping.get(direction.lower(), 0.0)

    @staticmethod
    def _direction_to_coords(direction: Optional[str]) -> Optional[Tuple[float, float]]:
        mapping = {
            "ahead": (0.0, 1.0),
            "forward": (0.0, 1.0),
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "behind": (0.0, -1.0),
            "back": (0.0, -1.0),
        }
        if not direction:
            return None
        return mapping.get(direction.lower())


class VideoProcessor(VideoProcessorBase):
    """Captures frames at a throttled interval for downstream analysis."""

    def __init__(self) -> None:
        self.frame_queue: "queue.Queue[FrameArray]" = queue.Queue(maxsize=1)
        self._last_enqueued = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24").astype(np.uint8)
        now = time.time()

        if now - self._last_enqueued >= FRAME_INTERVAL_SECONDS:
            self._last_enqueued = now
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(img)
            except queue.Full:
                pass

        return frame


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "guidance_text": "Camera warming up…",
        "analysis_count": 0,
        "recognized_locations": [],
        "last_analysis_time": 0.0,
        "camera_active": False,
        "language": "en",
        "mode": "navigation",
        "memory_enabled": DEFAULT_MEMORY_ENABLED,
        "location_input": "",
        "latest_embedding": None,
        "latest_guidance": None,
        "last_audio_bytes": None,
        "mode_logged": "navigation",
        "detected_objects": [],
        "movement_instruction": None,
        "workflow_instruction": None,
        "workflow_guidance": None,
        "workflow_priority": None,
        "workflow_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_services() -> Dict[str, object]:
    gemini_api_key = os.getenv("GEMINI_API_KEY") or None
    opus_api_url = os.getenv("OPUS_API_URL", "")
    opus_api_key = os.getenv("OPUS_API_KEY", "")
    opus_workflow_url = os.getenv("OPUS_WORKFLOW_URL", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "guidely_scenes")
    project_id = os.getenv("GOOGLE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
    gemini_model = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash-exp")
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

    gemini = GeminiService(
        api_key=gemini_api_key,
        model=gemini_model,
        project_id=project_id,
        location=vertex_location,
    )
    embeddings = EmbeddingClient(
        api_key=gemini_api_key,
        model=embedding_model,
        project_id=project_id,
        location=vertex_location,
    )
    qdrant = None
    if qdrant_url:
        qdrant = QdrantMemory(
            url=qdrant_url,
            api_key=qdrant_key or None,
            collection_name=qdrant_collection,
        )

    opus = OpusLogger(api_url=opus_api_url, api_key=opus_api_key, workflow_url=opus_workflow_url)
    tts = TTSService()
    open_vision = OpenVisionService()
    analyzer = RealtimeAnalyzer(
        gemini=gemini,
        embeddings=embeddings,
        opus=opus,
        open_vision=open_vision,
        qdrant=qdrant,
    )

    return {
        "gemini": gemini,
        "embeddings": embeddings,
        "qdrant": qdrant,
        "opus": opus,
        "tts": tts,
        "open_vision": open_vision,
        "analyzer": analyzer,
    }


def log_mode_change(opus: OpusLogger, new_mode: str) -> None:
    if st.session_state.get("mode_logged") != new_mode:
        opus.log_action(
            action="mode_changed",
            status="success",
            data={"mode": new_mode},
        )
        st.session_state["mode_logged"] = new_mode




def speak_guidance(text: str, language: str, services: Dict[str, object]) -> None:
    tts = cast(TTSService, services["tts"])
    try:
        audio_bytes = tts.synthesize(text=text, language=language)
        st.session_state["last_audio_bytes"] = audio_bytes
    except Exception as exc:
        st.warning(f"TTS unavailable: {exc}")


def handle_emergency(audio_placeholder: DeltaGenerator, opus: OpusLogger) -> None:
    opus.log_action("emergency_triggered", data={"timestamp": time.time()})
    tone = generate_emergency_tone()
    audio_placeholder.audio(tone, format="audio/wav", autoplay=True)
    st.error("Emergency alert sent! Hold device steady and wait for assistance.")


def save_current_location(
    services: Dict[str, object],
    location_name: str,
    description: str,
    embedding: Optional[EmbeddingArray],
) -> None:
    qdrant: Optional[QdrantMemory] = cast(Optional[QdrantMemory], services.get("qdrant"))
    opus: OpusLogger = cast(OpusLogger, services["opus"])

    if not qdrant:
        st.warning("Qdrant is not configured. Set QDRANT_URL to enable memory.")
        return
    if embedding is None:
        st.warning("Run an analysis before saving the location.")
        return

    success = qdrant.save_scene(
        vector=embedding,
        location_name=location_name,
        description=description,
    )
    if success:
        opus.log_action(
            action="location_saved",
            data={"location": location_name},
        )
        st.success(f"Location '{location_name}' saved to memory!")
    else:
        st.error("Unable to save the location. Check Qdrant logs.")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
        }
        .guidance-box {
            background: #1f1f1f;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 18px;
            border: 2px solid #4CAF50;
            min-height: 260px;
        }
        .guidance-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #4CAF50;
        }
        .guidance-body {
            font-size: 1.4rem;
            line-height: 1.6rem;
            max-height: 360px;
            overflow-y: auto;
        }
        .status-panel {
            background: #101010;
            border: 1px solid #2f2f2f;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 1rem;
            line-height: 1.35rem;
        }
        .priority-pill {
            background: #4CAF50;
            color: #0b0b0b;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-left: 0.35rem;
        }
        .warning-text {
            color: #ffb74d;
            margin-top: 0.35rem;
            font-size: 0.95rem;
        }
        .status-panel ul {
            list-style: disc;
            margin-left: 1.25rem;
            margin-top: 0.5rem;
        }
        .movement-pill {
            background: #1d472a;
            color: #d2f5da;
            padding: 0.5rem 0.75rem;
            border-radius: 999px;
            display: inline-block;
            margin-top: 0.5rem;
        }
        button[kind="primary"] {
            font-size: 1.25rem !important;
            min-height: 60px !important;
        }
        .stButton>button.emergency {
            background-color: #f44336 !important;
            color: #fff !important;
            border: none;
            animation: pulse 2s infinite;
            font-size: 1.2rem;
            min-height: 80px;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(244,67,54, 0.7); }
            70% { box-shadow: 0 0 0 20px rgba(244,67,54, 0); }
            100% { box-shadow: 0 0 0 0 rgba(244,67,54, 0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Guidely AI",
        page_icon="🧭",
        layout="wide",
    )
    inject_css()
    init_session_state()

    services = get_services()
    opus = cast(OpusLogger, services["opus"])
    analyzer = cast(RealtimeAnalyzer, services["analyzer"])

    st.sidebar.header("Guidely Controls")

    mode_label = st.sidebar.radio(
        "Mode",
        options=list(MODE_PROMPTS.keys()),
        format_func=lambda key: key.title(),
        index=list(MODE_PROMPTS.keys()).index(st.session_state["mode"]),
    )
    if mode_label != st.session_state["mode"]:
        st.session_state["mode"] = mode_label
    log_mode_change(opus, st.session_state["mode"])

    language_label = st.sidebar.selectbox(
        "Language",
        options=list(LANGUAGE_LABELS.keys()),
        index=list(LANGUAGE_LABELS.values()).index(st.session_state["language"])
        if st.session_state["language"] in LANGUAGE_LABELS.values()
        else 0,
    )
    st.session_state["language"] = LANGUAGE_LABELS[language_label]

    st.session_state["memory_enabled"] = st.sidebar.toggle(
        "Enable scene memory",
        value=st.session_state.get("memory_enabled", True),
    )

    st.sidebar.divider()
    st.session_state["location_input"] = st.sidebar.text_input(
        "Label current location",
        value=st.session_state.get("location_input", ""),
    )
    save_disabled = not st.session_state["location_input"]
    if st.sidebar.button(
        "Save location",
        use_container_width=True,
        disabled=save_disabled,
    ):
        save_current_location(
            services=services,
            location_name=st.session_state["location_input"],
            description=st.session_state.get("guidance_text", ""),
            embedding=st.session_state.get("latest_embedding"),
        )

    st.title("Guidely AI · Navigation Assistant")
    st.caption("Live scene awareness, multilingual guidance, memory, and rapid emergency response.")

    control_col, stats_col = st.columns([2, 1])
    with control_col:
        toggle_label = "START" if not st.session_state["camera_active"] else "STOP"
        if st.button(
            f"{toggle_label} CAMERA", use_container_width=True, type="primary"
        ):
            st.session_state["camera_active"] = not st.session_state["camera_active"]
            st.rerun()

    with stats_col:
        st.metric(label="Analyses", value=st.session_state["analysis_count"])
        st.metric(
            label="Locations recognized",
            value=len(st.session_state.get("recognized_locations", [])),
        )

    rtc_config = build_rtc_configuration()
    with st.sidebar.expander("WebRTC connectivity", expanded=False):
        stun_disabled = _env_flag("GUIDELY_DISABLE_STUN")
        force_turn = _env_flag("GUIDELY_FORCE_TURN")
        st.write(  # type: ignore[misc]
            "STUN disabled" if stun_disabled else "STUN enabled (Google pool or custom)"
        )
        st.write(  # type: ignore[misc]
            "TURN required (relay-only)" if force_turn else "TURN optional"
        )
        turn_urls_env = os.getenv("GUIDELY_TURN_URLS") or os.getenv("GUIDELY_TURN_URL") or ""
        turn_count = len([url for url in turn_urls_env.split(",") if url.strip()])
        st.write(f"TURN URLs configured: {turn_count}")  # type: ignore[misc]
        st.caption("Adjust GUIDELY_* env vars if you need different networking behavior.")

    video_col, guidance_col = st.columns([2, 1])
    with video_col:
        st.subheader("Live camera feed")
        preview_placeholder: Any = st.empty()
        webrtc_ctx = webrtc_streamer(
            key="guidely-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15},
                },
                "audio": False,
            },
            video_processor_factory=VideoProcessor,
            async_processing=True,
            desired_playing_state=st.session_state["camera_active"],
        )

    guidance_placeholder: Any = guidance_col.empty()
    recognized_placeholder: Any = guidance_col.empty()
    audio_placeholder: Any = guidance_col.empty()

    render_guidance_panel(guidance_placeholder, recognized_placeholder)

    col_left, col_right = guidance_col.columns(2)
    if col_left.button("🔊 Replay guidance", use_container_width=True):
        last_audio = st.session_state.get("last_audio_bytes")
        if last_audio:
            audio_placeholder.audio(last_audio, format="audio/mp3", autoplay=True)
        elif st.session_state.get("guidance_text"):
            speak_guidance(
                st.session_state["guidance_text"],
                st.session_state["language"],
                services,
            )
            audio_placeholder.audio(
                st.session_state.get("last_audio_bytes"),
                format="audio/mp3",
                autoplay=True,
            )
        else:
            st.info("No guidance available yet.")

    if col_right.button(
        "🚨 Emergency Alert",
        key="emergency",
        use_container_width=True,
    ):
        handle_emergency(audio_placeholder, opus)

    if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
        processor = cast(VideoProcessor, webrtc_ctx.video_processor)
        try:
            frame = processor.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            image = bgr_frame_to_pil(frame)
            preview_placeholder.image(image, caption="Current camera frame", use_column_width=True)
            try:
                result = analyzer.analyze(
                    image=image,
                    mode=st.session_state["mode"],
                    language=st.session_state["language"],
                    memory_enabled=st.session_state.get("memory_enabled", True),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unable to analyze the frame: {exc}")
                result = None
            if result:
                st.session_state["guidance_text"] = result.guidance
                st.session_state["analysis_count"] += 1
                st.session_state["latest_embedding"] = result.embedding
                st.session_state["latest_guidance"] = result.guidance
                st.session_state["last_analysis_time"] = time.time()
                st.session_state["detected_objects"] = result.objects
                st.session_state["movement_instruction"] = result.movement
                workflow_output = result.workflow
                if workflow_output:
                    st.session_state["workflow_instruction"] = (
                        workflow_output.navigation_instruction
                        or workflow_output.navigation_decision
                    )
                    st.session_state["workflow_guidance"] = (
                        workflow_output.guidance_text
                        or workflow_output.navigation_decision
                    )
                    st.session_state["workflow_priority"] = workflow_output.priority_level
                    st.session_state["workflow_error"] = workflow_output.error_message
                else:
                    st.session_state["workflow_instruction"] = None
                    st.session_state["workflow_guidance"] = None
                    st.session_state["workflow_priority"] = None
                    st.session_state["workflow_error"] = None

                if result.recognized_location:
                    st.session_state.setdefault("recognized_locations", [])
                    if result.recognized_location not in st.session_state["recognized_locations"]:
                        st.session_state["recognized_locations"].append(result.recognized_location)
                    st.success(f"📍 Recognized: {result.recognized_location}")

                render_guidance_panel(guidance_placeholder, recognized_placeholder)
                speak_guidance(
                    text=result.guidance,
                    language=st.session_state["language"],
                    services=services,
                )
                audio_placeholder.audio(
                    st.session_state.get("last_audio_bytes"),
                    format="audio/mp3",
                    autoplay=True,
                )
    elif st.session_state.get("camera_active"):
        guidance_col.info(
            "Waiting for camera connection… check that TURN/STUN settings are reachable from this network."
        )


if __name__ == "__main__":
    main()


def parse_navigation_response(raw_text: str) -> Tuple[str, List[DetectedObject], Optional[MovementAdvice]]:
    cleaned = raw_text.strip()
    if not cleaned:
        return raw_text, [], None

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else cleaned

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return raw_text, [], None

    guidance = str(payload.get("guidance") or raw_text).strip()

    objects: List[DetectedObject] = []
    for entry in cast(List[Dict[str, Any]], payload.get("objects", []) or []):
        label = entry.get("label")
        if not isinstance(label, str):
            continue
        distance = _safe_float(entry.get("distance_m"))
        direction = entry.get("direction")
        objects.append(
            DetectedObject(
                label=label.strip(),
                distance_m=distance,
                direction=direction.strip().lower() if isinstance(direction, str) else None,
            )
        )

    movement = None
    movement_entry = cast(Dict[str, Any], payload.get("movement") or {})
    suggestion = movement_entry.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip():
        reason = movement_entry.get("reason")
        movement = MovementAdvice(
            suggestion=suggestion.strip(),
            reason=reason.strip() if isinstance(reason, str) and reason.strip() else None,
        )

    return guidance or raw_text, objects, movement


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
