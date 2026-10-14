"""Fast face-presence web service.

POST /detect  (multipart/form-data, field "file")
  -> {"has_face": true|false, "face_count": n, "faces": [...], "time_ms": ms, "backend": "yunet"}

GET  /health  -> {"status": "ok", "backend": "yunet"}
GET  /        -> usage info
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .detector import FacePresenceDetector, decode_image

log = logging.getLogger("face-service")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB upload cap

detector: FacePresenceDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = FacePresenceDetector()  # downloads YuNet weights on first run if needed
    detector.warmup()  # pay JIT/first-inference cost at startup, not on first request
    log.info("face detector ready (backend=%s)", detector.backend)
    yield


app = FastAPI(title="Face Presence Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "service": "face-presence",
        "usage": "POST /detect with multipart field 'file' (an image). Returns {has_face: bool}.",
        "endpoints": ["/detect", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "backend": detector.backend if detector else "loading"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(..., description="Image file (jpg/png/webp/...)"),
    threshold: float = Query(0.6, ge=0.1, le=0.99,
                             description="Face score threshold. Higher = stricter."),
    plain: bool = Query(False, description="If true, return raw true/false text."),
    include_faces: bool = Query(False, description="If true, include face boxes."),
):
    if detector is None:
        raise HTTPException(503, "detector not ready")

    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file too large (>{MAX_BYTES // 1024 // 1024} MB)")

    img = decode_image(data)
    if img is None:
        raise HTTPException(400, "undecodable image: upload a valid jpg/png/webp")

    # Temporarily honour per-request threshold without rebuilding the model.
    orig = detector.score_threshold
    try:
        if abs(threshold - orig) > 1e-9 and detector._detector is not None:
            with detector._lock:
                detector._detector.setScoreThreshold(threshold)
        faces, ms = detector.detect(img)
    finally:
        if detector._detector is not None:
            with detector._lock:
                try:
                    detector._detector.setScoreThreshold(orig)
                except Exception:
                    pass

    has_face = len(faces) > 0
    if plain:
        return PlainTextResponse("true" if has_face else "false")

    payload: dict = {
        "has_face": has_face,
        "face_count": len(faces),
        "time_ms": round(ms, 2),
        "backend": detector.backend,
    }
    if include_faces:
        payload["faces"] = faces
    return payload
