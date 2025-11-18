# Guidely AI – Navigation Assistant

Guidely AI is a Streamlit-based navigation assistant for blind and low-vision users. It streams the device camera, analyzes frames every two seconds with Google Gemini (Vertex AI), stores/retrieves scene memories from Qdrant, speaks guidance using Google Cloud Text-to-Speech, and logs workflow steps in Opus.

## Features
- Live WebRTC camera preview (mobile-friendly) using `streamlit-webrtc`
- Automatic frame capture every two seconds with **offline-first** analysis (BLIP captioning + EasyOCR) before escalating to Gemini 2.0 Flash
- Three cognitive modes: Navigation, OCR, Describe
- Navigation mode streams structured obstacle JSON (object label, distance, direction) and generates explicit movement advice chips
- Multilingual narration (English, Arabic, Hindi) powered by **Edge neural voices first**, with Google Cloud Text-to-Speech as the fallback
- Scene memory via Qdrant + Gemini embeddings (0.85 similarity threshold)
- Emergency alert button with audible siren
- Opus workflow logging for observability

## Project structure
```
app.py                     # Streamlit UI + orchestration
services/
  gemini_service.py        # Gemini vision wrapper
  open_vision.py           # BLIP captioning + EasyOCR fallback before Gemini
  qdrant_service.py        # Vector memory helper
  opus_service.py          # REST logger for Opus
  tts_service.py           # Edge TTS with Google fallback
utils/
  camera_utils.py          # Frame conversion + emergency tone
  embeddings.py            # Gemini embedding helper
.streamlit/config.toml     # Dark theme with green accents
.env.example               # Environment template
requirements.txt           # Python dependencies
```

## Environment variables
| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | Google AI Studio API key (used as fallback when Vertex credentials are absent) |
| `GOOGLE_PROJECT_ID` | Google Cloud project hosting Vertex AI (falls back to `GOOGLE_CLOUD_PROJECT`) |
| `GOOGLE_VERTEX_LOCATION` | Vertex AI region (default `us-central1`) |
| `GOOGLE_GEMINI_MODEL` | Vertex Gemini vision model name (default `gemini-2.0-flash-exp`) |
| `GOOGLE_EMBEDDING_MODEL` | Vertex embedding model name (default `text-embedding-004`) |
| `GOOGLE_SERVICE_ACCOUNT_JSON` / `GOOGLE_APPLICATION_CREDENTIALS` | Inline JSON or file path for a service account with Vertex + TTS access |
| `GUIDELY_EMBEDDING_PROVIDER` | `local-first` (default), `remote-first`, `local-only`, or `google-only` |
| `GUIDELY_LOCAL_EMBED_MODEL` | Sentence-transformers model name (default `sentence-transformers/all-MiniLM-L6-v2`) |
| `GUIDELY_LOCAL_EMBED_DEVICE` | Device string for sentence-transformers (default `cpu`) |
| `QDRANT_URL` | HTTPS endpoint of your Qdrant cluster (optional for memory) |
| `QDRANT_API_KEY` | API key/token for Qdrant (if required) |
| `QDRANT_COLLECTION` | Vector collection name (default `guidely_scenes`) |
| `QDRANT_VECTOR_SIZE` | Embedding dimension (Gemini `text-embedding-004` = **3072**) |
| `OPUS_API_URL` | Opus logging endpoint (optional) |
| `OPUS_WORKFLOW_ID` | Workflow ID to tag every Opus log entry |
| `OPUS_API_KEY` | Bearer token for Opus logging (optional) |
| `ANALYSIS_INTERVAL_SECONDS` | Seconds between automatic frame captures (default 2) |
| `ENABLE_MEMORY` | `true`/`false` toggle for Qdrant usage on startup |
| `GUIDELY_STUN_URLS` | Comma-separated STUN server URLs to override the defaults (leave blank for Google STUN pool) |
| `GUIDELY_TURN_URL` | TURN server URL (for restrictive networks, e.g., `turn:turn.example.com:3478`) |
| `GUIDELY_TURN_URLS` | Comma-separated TURN URLs (TCP/UDP variants) to improve reliability; falls back to `GUIDELY_TURN_URL` if unset |
| `GUIDELY_TURN_USERNAME` / `GUIDELY_TURN_PASSWORD` | Credentials for the TURN server if required |
| `GUIDELY_FORCE_TURN` | Set to `true` to enforce relay-only ICE (TURN required, no direct UDP) |
| `GUIDELY_DISABLE_STUN` | Set to `true` to skip STUN entirely when your network blocks UDP |
| `GUIDELY_TTS_PROVIDER` | `edge` (default), `google`, or `auto` to control provider priority |
| `EDGE_TTS_RATE` | Speaking-rate adjustment for Edge voices (default `-10%`) |
| `EDGE_TTS_VOICE_<LANG>` | Override Edge voice per language code (e.g., `EDGE_TTS_VOICE_EN`) |
| `GOOGLE_TTS_CREDENTIALS_JSON` | (Optional) legacy inline JSON variable kept for backwards compatibility |
| `GOOGLE_TTS_VOICE_OVERRIDE` | Optional Google voice name (e.g., `en-US-Neural2-F`) to use for every language |
| `GOOGLE_TTS_SPEAKING_RATE` / `GOOGLE_TTS_PITCH` | Float overrides for Google’s speaking rate and pitch |
| `EDGE_TTS_VOLUME` | Optional Edge volume tag (e.g., `+0%`) |
| `EDGE_TTS_STYLE` | Optional Edge style (e.g., `cheerful`) |

### Open-source vs. managed services
| Capability | Default stack | Needs API key? |
| --- | --- | --- |
| Vision/scene captioning | **BLIP** (`transformers`, runs locally). Falls back to Gemini Vision when BLIP/EasyOCR unavailable. | Gemini fallback requires `GEMINI_API_KEY` or service account |
| OCR | **EasyOCR** (offline) before Gemini OCR. | Gemini fallback only |
| Embeddings + long-term memory | **SentenceTransformers** (MiniLM) → Qdrant. Gemini Text Embedding 004 only if local model unavailable. | Gemini fallback + optional Qdrant key |
| Speech synthesis | **Edge neural voices** via `edge-tts`, then Google Cloud TTS | Google fallback only |
| Logging | Opus workflow API | Yes if you use Opus |
| TURN relays | Any provider you configure | Only if your TURN vendor requires it |
4. You can override the default Google STUN pool by setting `GUIDELY_STUN_URLS=stun:your.stun.server:3478` (comma separated for multiples).

### Text-to-speech options
Speech runs through `edge-tts` by default for instant, API-free narration. If Edge voices are unavailable or you set `GUIDELY_TTS_PROVIDER=google`, the service falls back to Google Cloud Text-to-Speech using the same service-account credentials as Vertex AI. Customize the behavior via:

- `EDGE_TTS_VOICE_<LANG>` or `EDGE_TTS_VOICE_MAP` for language-specific neural voices
- `EDGE_TTS_RATE`, `EDGE_TTS_VOLUME`, `EDGE_TTS_STYLE` to tweak delivery
- `GOOGLE_TTS_VOICE_OVERRIDE`, `GOOGLE_TTS_SPEAKING_RATE`, `GOOGLE_TTS_PITCH`, `GOOGLE_TTS_AUDIO_ENCODING` for the Google fallback

Because Edge TTS is online but keyless, deployments without Google credentials still narrate guidance immediately.

### Offline-first computer vision
- `services/open_vision.py` loads a BLIP image-captioning pipeline (`Salesforce/blip-image-captioning-large`) and an EasyOCR reader (`en/ar/hi`).
- Navigation mode structures BLIP captions into lightweight obstacle/movement heuristics so the UI still shows objects + “move/stop” chips even if Gemini is unreachable.
- OCR mode uses EasyOCR exclusively and only escalates to Gemini when EasyOCR is missing.
- When BLIP/EasyOCR succeed, Gemini is never called—saving quota and allowing Streamlit Community Cloud deployments without GPU APIs. Provide `transformers`, `torch`, and `easyocr` friendly wheels (already listed in `requirements.txt`).

## Service configuration

### Google Gemini (Vertex AI)
- Provide a Google Cloud **service account** with access to Vertex AI plus Text-to-Speech. The app automatically uses Vertex via REST + OAuth when `GOOGLE_PROJECT_ID` is set; if that variable is absent it falls back to the legacy `GEMINI_API_KEY` path for quick demos.
- Enable the **Gemini 2.0 Flash Experimental** and **Text Embedding 004** publishers inside Vertex AI (or AI Studio when using the fallback key).
- The app compresses each frame (640×480 for analysis) before sending it to Gemini to control latency and cost.
- Gemini vision returns concise guidance tailored by the selected mode/language; the resulting text is immediately embedded via Gemini Text Embedding 004 so Qdrant can store semantic memories.
  - Navigation mode prompt enforces obstacle distance + direction callouts.
  - OCR mode tells Gemini to emit all detected text with line breaks.
  - Describe mode asks for rich ambient narration.

### Qdrant scene memory
1. Create a Qdrant Cloud cluster (or run `docker run -p 6333:6333 qdrant/qdrant`).
2. Create a collection named `guidely_scenes` with **Cosine** distance and vector size **3072**. The service auto-recreates the collection if the stored size mismatches `QDRANT_VECTOR_SIZE`.
3. Set `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, and (optionally) override `QDRANT_VECTOR_SIZE` if you switch embedding models. Local embeddings default to **384** dimensions (MiniLM). Update Qdrant accordingly when you change models.
4. When "Enable scene memory" is toggled on, every analysis stores/retrieves vectors if Qdrant is reachable. Failures fall back gracefully with on-screen warnings.
- Local embeddings come from sentence-transformers and are normalized before storage so cosine similarity in Qdrant behaves well. Set `GUIDELY_EMBEDDING_PROVIDER=google-only` if you prefer managed embeddings exclusively.
5. To smoke-test connectivity at any time, run `python scripts/test_qdrant.py` (loads your `.env`, verifies health, ensures the collection schema, inserts+searches a sample vector, and cleans it up). Add `--skip-insert` if you only want the health/schema checks.

### Opus workflow logging
- Set `OPUS_API_URL`, `OPUS_API_KEY`, and `OPUS_WORKFLOW_ID` from the workflow you shared (e.g., *Guidely AI Navigation Assistant Workflow*).
- The app logs these actions so you can map them to the workflow nodes: `scene_analysis_started`, `scene_analysis_completed`, `location_saved`, `location_recognized`, `emergency_triggered`, `mode_changed`.
- Each payload includes the current mode/language or location metadata so Opus dashboards can branch exactly like the workflow diagram.

### WebRTC connectivity
1. On Windows the app now forces the `WindowsSelectorEventLoopPolicy`, which removes the `AttributeError: 'NoneType' object has no attribute 'sendto'` crash created by aiortc.
2. If your camera takes too long to connect on restricted networks, add at least one TURN server via `GUIDELY_TURN_URL` or the comma-separated `GUIDELY_TURN_URLS`, plus `GUIDELY_TURN_USERNAME` / `GUIDELY_TURN_PASSWORD`.
3. When UDP is blocked entirely, set `GUIDELY_DISABLE_STUN=true` and `GUIDELY_FORCE_TURN=true` so the server only negotiates over your TURN relays (e.g., Metered.live TCP endpoints).
4. To minimize initial bandwidth, the Streamlit camera feed now requests 640×480 @ 15fps; adjust these constraints in `app.py` if you need HD streams.
5. You can override the default Google STUN pool by setting `GUIDELY_STUN_URLS=stun:your.stun.server:3478` (comma separated for multiples).

## Running locally
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Open the URL shown in the terminal. Allow camera access when prompted.

## Deployment notes
- Streamlit Cloud compatible; add all env vars in the dashboard.
- Qdrant is optional; set `Enable scene memory` toggle off if not configured.
  - If you later enable it, simply set the env vars and restart—no code changes needed.
  - The Qdrant helper checks the existing collection size and recreates it if it does not match the embedding dimensionality, preventing subtle similarity-score bugs.
- Use the helper script for validation:

```powershell
python scripts/test_qdrant.py
```

Add `--skip-insert` if your tenant disallows writes during verification.

## End-to-end workflow
1. **Capture** – `streamlit-webrtc` streams the live camera into the `VideoProcessor`, which throttles frames to one every two seconds and converts them to PIL images.
2. **Analyze** – `GeminiService` sends the compressed frame plus a mode-specific prompt to **Gemini 2.0 Flash Experimental**. Gemini responds with guidance tailored to Navigation/OCR/Describe plus the chosen language instructions.
3. **Embed + Memory** – `EmbeddingClient` calls Gemini Text Embedding 004 on the guidance text. If scene memory is enabled, `QdrantMemory` searches for similar embeddings (0.85+ cosine). Recognized locations get surfaced to the UI and logged.
4. **Voice** – The latest guidance text feeds `TTSService` (Google Cloud Text-to-Speech). SSML emphasis highlights warnings, and the audio autoplays while the text remains in a large-font panel.
5. **Logging** – `OpusLogger` records every critical action (`scene_analysis_started/completed`, `location_saved`, `location_recognized`, `mode_changed`, `emergency_triggered`) so your Opus workflow stays in sync.
6. **Emergency** – The emergency button plays a local siren (generated via NumPy) and logs the event to Opus for escalation.

## Testing checklist
- ✅ Camera feed shows real-time video
- ✅ Guidance updates after each 2-second analysis
- ✅ Text-to-speech plays automatically and in the selected language
- ✅ Locations can be saved and recognized when Qdrant is available
- ✅ Emergency button emits audible alert and logs action
- ✅ All modes (Navigation / OCR / Describe) and languages (EN / AR / HI) covered
- ✅ Opus receives structured actions that align with the shared workflow diagram
