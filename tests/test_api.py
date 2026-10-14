import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def _client():
    return TestClient(app)


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def test_health():
    with _client() as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_blank_image_has_no_face():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with _client() as client:
        r = client.post("/detect", files={"file": ("blank.jpg", _encode(blank), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_face"] is False
    assert body["face_count"] == 0


def test_invalid_image_rejected():
    with _client() as client:
        r = client.post("/detect", files={"file": ("x.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400


def test_plain_mode():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    with _client() as client:
        r = client.post("/detect?plain=true",
                        files={"file": ("blank.jpg", _encode(blank), "image/jpeg")})
    assert r.status_code == 200
    assert r.text in ("true", "false")
