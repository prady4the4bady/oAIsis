"""Open-source vision + OCR helper used before falling back to Gemini."""

from __future__ import annotations

import logging
import importlib
import importlib.util
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _load_optional(module_name: str, attribute: str | None = None) -> Any:
    if not importlib.util.find_spec(module_name):  # pragma: no cover - import guard
        return None
    module = importlib.import_module(module_name)
    return getattr(module, attribute) if attribute else module


np = _load_optional("numpy")
pipeline = _load_optional("transformers", "pipeline")
easyocr = _load_optional("easyocr")

LOGGER = logging.getLogger(__name__)


@dataclass
class OpenVisionResult:
    guidance: str
    objects: List[Dict[str, Any]]
    movement: Optional[Dict[str, str]]


class OpenVisionService:
    def __init__(self, device: str | None = None, lazy: bool = True) -> None:
        self.device = device or "cpu"
        self.lazy = lazy
        self.caption_pipeline: Any | None = None
        self._caption_error: Optional[str] = None
        self._caption_lock = threading.Lock()
        self.ocr_readers: Dict[str, Any] = {}
        if not self.lazy:
            self._ensure_caption_pipeline()

    @property
    def available(self) -> bool:
        caption_ready = (self.caption_pipeline is not None) or (
            pipeline is not None and self._caption_error is None
        )
        return caption_ready or easyocr is not None

    def analyze(self, image: Any, mode: str, language: str) -> Optional[OpenVisionResult]:
        if mode == "ocr":
            return self._run_ocr(image, language)
        return self._run_caption(image=image, mode=mode, language=language)

    def _run_ocr(self, image: Any, language: str) -> Optional[OpenVisionResult]:
        reader = self._ensure_ocr_reader(language)
        if reader is None:
            raise RuntimeError("EasyOCR is not installed; cannot run offline OCR.")
        if np is None:
            raise RuntimeError("NumPy is required for EasyOCR preprocessing.")
        gray = np.array(image.convert("L"))
        lines = reader.readtext(gray, detail=0)
        text = "\n".join(line.strip() for line in lines if line.strip())
        if not text:
            text = "No text detected."
        return OpenVisionResult(guidance=text, objects=[], movement=None)

    def _ensure_ocr_reader(self, language: str) -> Any | None:
        if easyocr is None:
            return None
        lang_map = {
            "en": ["en"],
            "ar": ["ar", "en"],
            "hi": ["hi", "en"],
        }
        lang_list = lang_map.get(language, ["en"])
        key = ":".join(lang_list)
        if key not in self.ocr_readers:
            try:
                self.ocr_readers[key] = easyocr.Reader(lang_list, gpu=False)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Unable to initialize EasyOCR for %s: %s", lang_list, exc)
                self.ocr_readers[key] = None
        return self.ocr_readers.get(key)

    def _run_caption(self, image: Any, mode: str, language: str) -> Optional[OpenVisionResult]:
        if not self._ensure_caption_pipeline():
            raise RuntimeError("transformers captioning pipeline is unavailable")
        caption_pipeline = self.caption_pipeline
        if caption_pipeline is None:  # mypy/pyright guard
            raise RuntimeError("transformers captioning pipeline failed to initialize")
        prompt = self._language_prompt(mode=mode, language=language)
        outputs = caption_pipeline(
            image,
            generate_kwargs={"max_new_tokens": 120},
            prompt=prompt,
        )
        caption = outputs[0].get("generated_text", "").strip()
        if not caption:
            caption = "Unable to describe the scene."
        if mode == "navigation":
            objects = self._extract_objects(caption)
            movement = self._heuristic_movement(caption, objects)
            return OpenVisionResult(guidance=caption, objects=objects, movement=movement)
        return OpenVisionResult(guidance=caption, objects=[], movement=None)

    def _language_prompt(self, mode: str, language: str) -> str:
        language_prompts = {
            "en": "Describe the scene in English.",
            "ar": "صف المشهد باللغة العربية.",
            "hi": "दृश्य को हिंदी में वर्णित करें।",
        }
        base = language_prompts.get(language, language_prompts["en"])
        if mode == "navigation":
            return base + " Focus on obstacles, people, and paths."
        if mode == "describe":
            return base + " Include ambience and important objects."
        return base

    def _extract_objects(self, caption: str) -> List[Dict[str, Any]]:
        keywords = [
            "person",
            "people",
            "car",
            "vehicle",
            "door",
            "stair",
            "chair",
            "table",
            "bicycle",
            "animal",
            "tree",
        ]
        found: List[Dict[str, Any]] = []
        lowered = caption.lower()
        for word in keywords:
            if word in lowered:
                direction = self._guess_direction(lowered)
                found.append({"label": word, "distance_m": None, "direction": direction})
        return found

    def _guess_direction(self, text: str) -> Optional[str]:
        if "left" in text:
            return "left"
        if "right" in text:
            return "right"
        if "behind" in text:
            return "behind"
        if "front" in text or "ahead" in text:
            return "ahead"
        return None

    def _heuristic_movement(
        self, caption: str, objects: List[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        lowered = caption.lower()
        danger_terms = {"vehicle", "car", "crowd", "traffic", "obstacle", "busy"}
        if any(term in lowered for term in danger_terms):
            return {"suggestion": "stop", "reason": "Potential obstacle detected"}
        if "stairs" in lowered or "stair" in lowered:
            return {"suggestion": "step_back", "reason": "Stairs nearby"}
        if any(obj.get("direction") == "left" for obj in objects):
            return {"suggestion": "turn_right", "reason": "Objects on the left"}
        if any(obj.get("direction") == "right" for obj in objects):
            return {"suggestion": "turn_left", "reason": "Objects on the right"}
        return {"suggestion": "move_forward", "reason": "Path appears clear"}

    def _ensure_caption_pipeline(self) -> bool:
        if self.caption_pipeline is not None:
            return True
        if pipeline is None or self._caption_error is not None:
            return False
        with self._caption_lock:
            if self.caption_pipeline is not None:
                return True
            try:
                self.caption_pipeline = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-large",
                    device=self.device,
                )
            except Exception as exc:  # pragma: no cover - heavy model failures
                LOGGER.warning("Unable to load BLIP captioning pipeline: %s", exc)
                self._caption_error = str(exc)
                self.caption_pipeline = None
        return self.caption_pipeline is not None
