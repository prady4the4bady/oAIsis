import os

if os.name == "nt":
    os.environ.setdefault("PYTHONASYNCIO_USE_SELECTOR_POLICY", "1")

import asyncio
import json
import platform
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

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
    webrtc_streamer,
)
from services.gemini_service import GeminiService
from services.opus_service import OpusLogger
from services.qdrant_service import QdrantMemory
from services.tts_service import TTSService
from utils.camera_utils import bgr_frame_to_pil, generate_emergency_tone
from utils.embeddings import EmbeddingClient

load_dotenv()

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

FRAME_INTERVAL_SECONDS = float(os.getenv("ANALYSIS_INTERVAL_SECONDS", "2"))
DEFAULT_MEMORY_ENABLED = (os.getenv("ENABLE_MEMORY", "true").lower() == "true")

FrameArray = NDArray[np.uint8]
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
    objects: List[DetectedObject] = field(default_factory=list)
    movement: Optional[MovementAdvice] = None


class VideoProcessor(VideoProcessorBase):
    """Captures frames at a throttled interval for downstream analysis."""

    def __init__(self) -> None:
        self.frame_queue: "queue.Queue[FrameArray]" = queue.Queue(maxsize=1)
        self._last_enqueued = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
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
    defaults = {
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_services() -> Dict[str, object]:
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    opus_api_url = os.getenv("OPUS_API_URL", "")
    opus_api_key = os.getenv("OPUS_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "guidely_scenes")

    gemini = GeminiService(api_key=gemini_api_key)
    embeddings = EmbeddingClient(api_key=gemini_api_key)
    qdrant = None
    if qdrant_url:
        qdrant = QdrantMemory(
            url=qdrant_url,
            api_key=qdrant_key or None,
            collection_name=qdrant_collection,
        )

    opus = OpusLogger(api_url=opus_api_url, api_key=opus_api_key)
    tts = TTSService()

    return {
        "gemini": gemini,
        "embeddings": embeddings,
        "qdrant": qdrant,
        "opus": opus,
        "tts": tts,
    }


def log_mode_change(opus: OpusLogger, new_mode: str) -> None:
    if st.session_state.get("mode_logged") != new_mode:
        opus.log_action(
            action="mode_changed",
            status="success",
            data={"mode": new_mode},
        )
        st.session_state["mode_logged"] = new_mode


def analyze_image(
    image: Image.Image,
    services: Dict[str, object],
    mode: str,
    language: str,
) -> Optional[AnalysisResult]:
    gemini: GeminiService = services["gemini"]
    embeddings: EmbeddingClient = services["embeddings"]
    qdrant: Optional[QdrantMemory] = services.get("qdrant")
    opus: OpusLogger = services["opus"]

    prompt = MODE_PROMPTS[mode]
    language_note = LANGUAGE_INSTRUCTIONS.get(language, "Respond clearly.")
    opus.log_action(
        action="scene_analysis_started",
        data={"mode": mode, "language": language},
    )

    try:
        response_mime = "application/json" if mode == "navigation" else None
        raw_response = gemini.analyze_image(
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

        embedding_vector = embeddings.embed_text(guidance_text)
        recognized = None

        if (
            st.session_state.get("memory_enabled", True)
            and qdrant is not None
            and embedding_vector is not None
        ):
            match = qdrant.search_scene(embedding_vector)
            if match:
                recognized = match.location_name
                st.session_state.setdefault("recognized_locations", [])
                if recognized not in st.session_state["recognized_locations"]:
                    st.session_state["recognized_locations"].append(recognized)
                opus.log_action(
                    action="location_recognized",
                    data={"location": recognized, "score": match.score},
                )
        opus.log_action(
            action="scene_analysis_completed",
            data={"mode": mode, "language": language},
        )

        return AnalysisResult(
            guidance=guidance_text,
            recognized_location=recognized,
            embedding=embedding_vector,
            objects=detected_objects,
            movement=movement,
        )
    except Exception as exc:
        opus.log_action(
            action="scene_analysis_completed",
            status="error",
            data={"error": str(exc)},
        )
        st.error(f"Unable to analyze the frame: {exc}")
        return None


def speak_guidance(text: str, language: str, services: Dict[str, object]) -> None:
    tts: TTSService = services["tts"]
    try:
        audio_bytes = tts.synthesize(text=text, language=language)
        st.session_state["last_audio_bytes"] = audio_bytes
    except Exception as exc:
        st.warning(f"TTS unavailable: {exc}")


def handle_emergency(audio_placeholder: st.delta_generator.DeltaGenerator, opus: OpusLogger) -> None:
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
    qdrant: Optional[QdrantMemory] = services.get("qdrant")
    opus: OpusLogger = services["opus"]

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


def render_guidance_panel(guidance_placeholder, recognized_placeholder):
    guidance_placeholder.markdown(
        f"""
        <div class='guidance-box'>
            <div class='guidance-title'>Current Guidance</div>
            <div class='guidance-body'>{st.session_state['guidance_text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    panels: List[str] = []
    if st.session_state.get("recognized_locations"):
        panels.append(
            "<div><strong>Recognized locations:</strong> "
            + ", ".join(st.session_state["recognized_locations"])
            + "</div>"
        )

    movement: Optional[MovementAdvice] = st.session_state.get("movement_instruction")
    if movement:
        reason_text = f" — {movement.reason}" if movement.reason else ""
        panels.append(
            f"<div class='movement-pill'><strong>Movement:</strong> {movement.suggestion}{reason_text}</div>"
        )

    detected_objects: List[DetectedObject] = st.session_state.get("detected_objects", [])
    if detected_objects:
        items = "".join(
            f"<li>{obj.label}"
            + (
                f" • {obj.distance_m:.1f}m"
                if isinstance(obj.distance_m, float)
                else ""
            )
            + (f" • {obj.direction}" if obj.direction else "")
            + "</li>"
            for obj in detected_objects
        )
        panels.append(
            "<div><strong>Detected objects:</strong><ul>"
            + items
            + "</ul></div>"
        )

    if panels:
        recognized_placeholder.markdown(
            "<div class='status-panel'>" + "".join(panels) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        recognized_placeholder.empty()


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
    opus: OpusLogger = services["opus"]

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
        st.write(
            "STUN disabled" if stun_disabled else "STUN enabled (Google pool or custom)"
        )
        st.write(
            "TURN required (relay-only)" if force_turn else "TURN optional"
        )
        turn_urls_env = os.getenv("GUIDELY_TURN_URLS") or os.getenv("GUIDELY_TURN_URL") or ""
        turn_count = len([url for url in turn_urls_env.split(",") if url.strip()])
        st.write(f"TURN URLs configured: {turn_count}")
        st.caption("Adjust GUIDELY_* env vars if you need different networking behavior.")

    video_col, guidance_col = st.columns([2, 1])
    with video_col:
        st.subheader("Live camera feed")
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

    guidance_placeholder = guidance_col.empty()
    recognized_placeholder = guidance_col.empty()
    audio_placeholder = guidance_col.empty()

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
        processor: VideoProcessor = webrtc_ctx.video_processor
        try:
            frame = processor.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            image = bgr_frame_to_pil(frame)
            result = analyze_image(
                image=image,
                services=services,
                mode=st.session_state["mode"],
                language=st.session_state["language"],
            )
            if result:
                st.session_state["guidance_text"] = result.guidance
                st.session_state["analysis_count"] += 1
                st.session_state["latest_embedding"] = result.embedding
                st.session_state["latest_guidance"] = result.guidance
                st.session_state["last_analysis_time"] = time.time()
                st.session_state["detected_objects"] = result.objects
                st.session_state["movement_instruction"] = result.movement

                if result.recognized_location:
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
    for entry in payload.get("objects", []) or []:
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
    movement_entry = payload.get("movement") or {}
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
        return float(value)
    except (TypeError, ValueError):
        return None
