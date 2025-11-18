"""Camera-related helpers for Guidely AI."""

from __future__ import annotations

import io
import wave
import numpy as np
import numpy.typing as npt
from PIL import Image

SAMPLE_RATE = 44100
EMERGENCY_DURATION_SEC = 2
EMERGENCY_FREQ = 880


def bgr_frame_to_pil(frame: npt.NDArray[np.uint8]) -> Image.Image:
    """Convert OpenCV-style BGR frame into a PIL Image in RGB space."""
    rgb_frame = frame[:, :, ::-1]
    return Image.fromarray(rgb_frame)


def generate_emergency_tone(
    frequency: int = EMERGENCY_FREQ, duration: int = EMERGENCY_DURATION_SEC
) -> bytes:
    """Generate a simple sine-wave emergency tone and return WAV bytes."""
    num_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, num_samples, False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio = (tone * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())

    buffer.seek(0)
    return buffer.read()
