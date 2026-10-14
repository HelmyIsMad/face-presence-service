# API Integration Guide — Face Presence Service

Base URL (local): `http://localhost:8000`
Interactive docs (auto-generated): `http://localhost:8000/docs`

The service answers one question: **does this image contain a recognizable face?**

## 1. Start the server

```powershell
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads YuNet weights (~300 KB) into `models/`.

Check it's up:

```powershell
curl http://localhost:8000/health
# {"status":"ok","backend":"yunet"}
```

## 2. Endpoints

| Method | Path      | Purpose                                              |
|--------|-----------|------------------------------------------------------|
| `GET`  | `/`       | Usage info                                           |
| `GET`  | `/health` | Liveness: `{"status":"ok","backend":"yunet"}`        |
| `POST` | `/detect` | Face check. Multipart field `file` = image bytes     |

### `POST /detect` — request

- Content-Type: `multipart/form-data`
- Field name: **`file`** (required) — any common image: jpg / png / webp / bmp
- Max size: **10 MB**
- Query params (all optional):

| Param           | Default | Allowed     | Meaning                                                     |
|-----------------|---------|-------------|-------------------------------------------------------------|
| `threshold`     | `0.6`   | `0.1`–`0.99`| Face score cutoff. Higher = stricter (fewer false positives) |
| `plain`         | `false` | `true/false`| `true` → response is raw text `true` or `false`             |
| `include_faces` | `false` | `true/false`| `true` → response includes bounding boxes + scores          |

### `POST /detect` — response (JSON, default)

```jsonc
{
  "has_face": true,    // bool — the answer you want
  "face_count": 1,     // int
  "time_ms": 12.4,     // float — detection time server-side
  "backend": "yunet"   // "yunet" | "haar" (fallback)
}
```

With `?include_faces=true`, a `faces` array is added (coords in **original** image pixels):

```jsonc
{
  "has_face": true,
  "face_count": 1,
  "time_ms": 12.4,
  "backend": "yunet",
  "faces": [{ "x": 207, "y": 182, "w": 145, "h": 206, "score": 0.91 }]
}
```

With `?plain=true`, response is `Content-Type: text/plain` body `true` or `false`.

### Status codes

| Code  | Meaning                                      |
|-------|----------------------------------------------|
| `200` | OK (even when `has_face` is `false`)         |
| `400` | Empty file or undecodable image              |
| `413` | File > 10 MB                                 |
| `422` | Missing `file` field                         |
| `503` | Detector still loading (retry shortly)       |

## 3. Examples

### cURL

```bash
# JSON answer
curl -F "file=@photo.jpg" http://localhost:8000/detect

# Boolean only
curl -F "file=@photo.jpg" "http://localhost:8000/detect?plain=true"
# true

# With boxes + stricter threshold
curl -F "file=@photo.jpg" "http://localhost:8000/detect?include_faces=true&threshold=0.7"
```

### PowerShell

```powershell
# JSON answer
Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method Post `
  -Form @{ file = Get-Item "photo.jpg" }

# Boolean only
(Invoke-WebRequest -Uri "http://localhost:8000/detect?plain=true" -Method Post `
  -Form @{ file = Get-Item "photo.jpg" }).Content
# true
```

### Python (`requests`)

```python
import requests

URL = "http://localhost:8000/detect"

with open("photo.jpg", "rb") as f:
    r = requests.post(URL, files={"file": f}, timeout=30)
r.raise_for_status()
print(r.json()["has_face"])  # True / False
```

```python
# Boolean-only helper
def has_face(path: str, base: str = "http://localhost:8000") -> bool:
    with open(path, "rb") as f:
        r = requests.post(f"{base}/detect?plain=true", files={"file": f}, timeout=30)
    r.raise_for_status()
    return r.text.strip() == "true"
```

```python
# With bounding boxes
with open("photo.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:8000/detect",
        params={"include_faces": "true", "threshold": 0.6},
        files={"file": ("photo.jpg", f, "image/jpeg")},
        timeout=30,
    )
print(r.json())
```

### JavaScript — Node 18+ (`fetch`, `FormData`, `Blob`)

```js
import { readFile } from "node:fs/promises";

const bytes = await readFile("photo.jpg");
const form = new FormData();
form.append("file", new Blob([bytes], { type: "image/jpeg" }), "photo.jpg");

const res = await fetch("http://localhost:8000/detect", { method: "POST", body: form });
const { has_face } = await res.json();
console.log(has_face); // true / false
```

### JavaScript — browser (file input)

```html
<input type="file" id="picker" accept="image/*" />
<script>
  picker.onchange = async () => {
    const form = new FormData();
    form.append("file", picker.files[0]);
    const res = await fetch("http://localhost:8000/detect", { method: "POST", body: form });
    const { has_face } = await res.json();
    alert(has_face ? "Face found" : "No face");
  };
</script>
```

## 4. Notes for integrators

- **CORS is open** (`*`) — browser apps can call the API directly.
- Send the **original file bytes**; no base64 needed. Filename/content-type are ignored for decoding.
- `has_face == false` with `200 OK` is a **normal answer**, not an error.
- Large images are downscaled server-side (max side 640 px) for speed; tiny unrecognizable faces are intentionally ignored.
- Typical latency: ~5–20 ms inference + upload time. If you need boxes, pass `include_faces=true`.
- For health checks / load balancers, poll `GET /health`.
