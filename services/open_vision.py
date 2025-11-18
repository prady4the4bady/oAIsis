"""Open-source vision + OCR helper used before falling back to Gemini."""

from __future__ import annotations

import logging
import importlib
import importlib.util
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
    def __init__(self, device: str | None = None) -> None:
        self.caption_pipeline: Any | None = None
        self.ocr_reader: Any | None = None
        if pipeline is not None:
            try:
                self.caption_pipeline = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-large",
                    device=device or "cpu",
                )
            except Exception as exc:  # pragma: no cover - heavy model failures
                LOGGER.warning("Unable to load BLIP captioning pipeline: %s", exc)
                self.caption_pipeline = None
        if easyocr is not None:
            try:
                self.ocr_reader = easyocr.Reader(["en", "ar", "hi"], gpu=False)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Unable to initialize EasyOCR: %s", exc)
                self.ocr_reader = None

    @property
    def available(self) -> bool:
        return self.caption_pipeline is not None or self.ocr_reader is not None

    def analyze(self, image: Any, mode: str, language: str) -> Optional[OpenVisionResult]:
        if mode == "ocr":
            return self._run_ocr(image)
        return self._run_caption(image=image, mode=mode, language=language)

    def _run_ocr(self, image: Any) -> Optional[OpenVisionResult]:
        if self.ocr_reader is None:
            raise RuntimeError("EasyOCR is not installed; cannot run offline OCR.")
        if np is None:
            raise RuntimeError("NumPy is required for EasyOCR preprocessing.")
        gray = np.array(image.convert("L"))
        lines = self.ocr_reader.readtext(gray, detail=0)
        text = "\n".join(line.strip() for line in lines if line.strip())
        if not text:
            text = "No text detected."
        return OpenVisionResult(guidance=text, objects=[], movement=None)

    def _run_caption(self, image: Any, mode: str, language: str) -> Optional[OpenVisionResult]:
        if self.caption_pipeline is None:
            raise RuntimeError("transformers captioning pipeline is unavailable")
        prompt = self._language_prompt(mode=mode, language=language)
        outputs = self.caption_pipeline(image, generate_kwargs={"max_new_tokens":120}, prompt=prompt)
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
