"""YuNet-based face-presence detector.

Why YuNet: best CPU speed/accuracy trade-off for "is there a recognizable face?"
  - ~5-20 ms on CPU (vs 100-500 ms for MTCNN/RetinaFace)
  - Far more accurate / fewer false positives than Haar cascades
  - Single small ONNX model (~300 KB), runs via OpenCV DNN (no torch/onnxruntime needed)
  - Robust to pose/lighting; ignores tiny unrecognizable faces when downscaled

Fallback: Haar cascade if YuNet model can't load (fast but less accurate).
"""

from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

# Speed/accuracy knobs
SCORE_THRESHOLD = 0.6   # higher = fewer false positives, lower = higher recall
NMS_THRESHOLD = 0.3
TOP_K = 20
MAX_SIDE = 640          # downscale large images to this max dim for speed


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 100_000:
        return path
    print(f"[detector] downloading YuNet model -> {path}")
    urllib.request.urlretrieve(url, str(path))
    return path


class FacePresenceDetector:
    """Thread-safe singleton wrapper around cv2.FaceDetectorYN + Haar fallback."""

    def __init__(
        self,
        model_path: Path | None = None,
        score_threshold: float = SCORE_THRESHOLD,
        nms_threshold: float = NMS_THRESHOLD,
        top_k: int = TOP_K,
        max_side: int = MAX_SIDE,
    ):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.max_side = max_side
        self._lock = threading.Lock()
        self._detector = None
        self._haar = None
        self.backend = "none"

        model_path = model_path or MODEL_PATH
        try:
            ensure_model(model_path)
            self._detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                (320, 320),  # placeholder, reset per-image via setInputSize
                score_threshold,
                nms_threshold,
                top_k,
            )
            self.backend = "yunet"
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[detector] YuNet init failed ({exc}), falling back to Haar")
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(haar_path)
            self.backend = "haar"

    def warmup(self) -> None:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            self.has_face(dummy)
        except Exception:
            pass

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Downscale large images for speed. Returns (img, scale)."""
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= self.max_side:
            return img, 1.0
        scale = self.max_side / float(m)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return small, scale

    def detect(self, img: np.ndarray) -> tuple[list[dict], float]:
        """Return (faces, elapsed_ms). faces = [{x,y,w,h,score}, ...] in ORIGINAL coords."""
        t0 = time.perf_counter()
        if img is None or img.size == 0:
            return [], 0.0

        small, scale = self._preprocess(img)
        h, w = small.shape[:2]
        faces: list[dict] = []

        with self._lock:
            if self._detector is not None:
                self._detector.setInputSize((w, h))
                _, results = self._detector.detect(small)
                if results is not None:
                    inv = 1.0 / scale
                    for r in results:
                        # YuNet row: x, y, w, h, 10 landmark coords, score (15 values)
                        x, y, ww, hh = (float(v) * inv for v in r[:4])
                        faces.append({
                            "x": int(x), "y": int(y),
                            "w": int(ww), "h": int(hh),
                            "score": float(r[-1]),
                        })
            elif self._haar is not None:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                    minSize=(30, 30))
                inv = 1.0 / scale
                for (x, y, ww, hh) in rects:
                    faces.append({"x": int(x * inv), "y": int(y * inv),
                                  "w": int(ww * inv), "h": int(hh * inv), "score": 1.0})

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return faces, elapsed_ms

    def has_face(self, img: np.ndarray) -> tuple[bool, int, float]:
        faces, ms = self.detect(img)
        return len(faces) > 0, len(faces), ms


def decode_image(data: bytes) -> np.ndarray | None:
    """Fast bytes -> BGR array. Returns None if undecodable."""
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
