# Face Presence Service

Returns `True` if an image contains a recognizable face, `False` otherwise.
Optimized for **speed + accuracy**: YuNet (OpenCV DNN) — ~5–20 ms on CPU,
far more accurate than Haar cascades, far faster than MTCNN / RetinaFace.

## Run

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads the YuNet weights (~300 KB) into `models/`.

## API

### `POST /detect`
Multipart upload, field name `file`:

```powershell
curl -F "file=@photo.jpg" http://localhost:8000/detect
# {"has_face":true,"face_count":1,"time_ms":12.4,"backend":"yunet"}
```

| Query param     | Default | Meaning                                              |
|-----------------|---------|------------------------------------------------------|
| `threshold`     | `0.6`   | Face score 0.1–0.99. Higher = stricter.              |
| `plain`         | `false` | `plain=true` returns raw `true` / `false` text.      |
| `include_faces` | `false` | `include_faces=true` adds `faces: [{x,y,w,h,score}]` |

Minimal boolean-only usage:

```powershell
curl -F "file=@photo.jpg" "http://localhost:8000/detect?plain=true"
# true
```

Python example:

```python
import requests
with open("photo.jpg", "rb") as f:
    print(requests.post("http://localhost:8000/detect", files={"file": f}).json())
    # {'has_face': True, 'face_count': 1, 'time_ms': 11.8, 'backend': 'yunet'}
```

### `GET /health` → `{"status":"ok","backend":"yunet"}`

## Notes

- Images are downscaled to max side 640 px before detection (speed; tiny
  unrecognizable faces are intentionally ignored).
- 10 MB upload cap, invalid images → `400`.
- If YuNet weights can't load, service falls back to Haar cascade (`backend: haar`).

## Docker

```powershell
docker build -t face-service .
docker run -p 8000:8000 face-service
```
