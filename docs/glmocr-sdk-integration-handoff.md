# GLM-OCR SDK Integration Handoff

This machine exposes a local/self-hosted GLM-OCR SDK service for document OCR, layout parsing, table recognition, and formula recognition. Other apps on the LAN can send images or PDFs to it and receive markdown plus structured JSON layout results.

## Current Service

- SDK health URL: `http://192.168.4.35:5002/health`
- SDK OCR URL: `http://192.168.4.35:5002/glmocr/parse`
- Manager proxy URL: `http://192.168.4.35:8077/api/ocr/parse`
- Systemd service: `glmocr-sdk.service`
- Local OCR model backend used by the SDK: `http://127.0.0.1:8009/v1/chat/completions`

The SDK listens on `0.0.0.0:5002`, so other devices on the local network should call the `192.168.4.35` URL, not `0.0.0.0`. The direct SDK endpoint is the preferred integration point when the consuming app only needs OCR. The manager proxy is useful if the app wants a normalized `ok/text/raw` response shape.

## Supported Input Shapes

The direct SDK endpoint accepts JSON POST requests. It does not accept `multipart/form-data` directly. If your app has a local file upload, read the file, base64 encode it, and send a data URL.

### Single Image URL

```json
{
  "image_url": "http://example.local/image.png"
}
```

### Single Image Base64

```json
{
  "image_base64": "<base64 image bytes>",
  "mime_type": "image/png"
}
```

The server will convert this into a `data:image/png;base64,...` URL internally.

### PDF Data URL

```json
{
  "file": "data:application/pdf;base64,<base64 pdf bytes>"
}
```

PDFs should be sent as data URLs. The wrapper detects `data:application/pdf;base64,...` and passes PDF bytes into the SDK pipeline.

### Multiple Inputs

```json
{
  "images": [
    "data:image/png;base64,<base64 image 1>",
    "data:image/jpeg;base64,<base64 image 2>"
  ]
}
```

`images` may contain image URLs, image data URLs, or PDF data URLs. Multiple results are combined in the response.

### Optional Flags

```json
{
  "file": "data:application/pdf;base64,<base64 pdf bytes>",
  "need_layout_visualization": false
}
```

`need_layout_visualization` requests layout visualization generation when supported. The current wrapper returns layout visualization page markers, not image bytes.

## Direct SDK Response

A successful direct SDK response looks like:

```json
{
  "json_result": {},
  "markdown_result": "...",
  "layout_details": {},
  "md_results": "...",
  "layout_visualization": [],
  "data_info": {
    "pages": []
  },
  "usage": {},
  "model": "glm-ocr",
  "id": "chatcmpl-...",
  "created": 1782500000
}
```

Use `markdown_result` as the primary extracted text/markdown. Use `json_result` or `layout_details` when the app needs structured regions, tables, formulas, or layout information. `md_results` is an alias for compatibility.

Errors use JSON and non-2xx status codes where possible:

```json
{
  "error": "No images or file provided"
}
```

## Manager Proxy Response

The llm-stack manager exposes `POST /api/ocr/parse`, which forwards to the SDK and normalizes the response:

```json
{
  "ok": true,
  "text": "...",
  "markdown_result": "...",
  "md_results": "...",
  "json_result": {},
  "layout_details": {},
  "layout_visualization": [],
  "data_info": {},
  "usage": {},
  "raw": {}
}
```

If integrating into a general app, this shape can be easier because failures return:

```json
{
  "ok": false,
  "error": "..."
}
```

The manager proxy accepts the same `file`, `images`, `image_url`, and `image_base64` patterns as the direct SDK endpoint. It also forwards these optional fields when present: `model`, `return_crop_images`, `need_layout_visualization`, `start_page_id`, `end_page_id`, `request_id`, and `user_id`.

## Curl Examples

Health check:

```bash
curl -sf http://192.168.4.35:5002/health
```

Send an image data URL:

```bash
IMAGE_B64="$(base64 -w0 ./scan.png)"
curl -sS http://192.168.4.35:5002/glmocr/parse \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"${IMAGE_B64}\",\"mime_type\":\"image/png\"}"
```

Send a PDF:

```bash
PDF_B64="$(base64 -w0 ./document.pdf)"
curl -sS http://192.168.4.35:5002/glmocr/parse \
  -H 'Content-Type: application/json' \
  -d "{\"file\":\"data:application/pdf;base64,${PDF_B64}\"}"
```

Use the manager proxy:

```bash
PDF_B64="$(base64 -w0 ./document.pdf)"
curl -sS http://192.168.4.35:8077/api/ocr/parse \
  -H 'Content-Type: application/json' \
  -d "{\"file\":\"data:application/pdf;base64,${PDF_B64}\"}"
```

## JavaScript Integration

```js
import { readFile } from "node:fs/promises";
import path from "node:path";

const OCR_URL = process.env.OCR_URL || "http://192.168.4.35:5002/glmocr/parse";

function mimeTypeFor(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".pdf") return "application/pdf";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".webp") return "image/webp";
  return "image/png";
}

export async function ocrFile(filePath) {
  const bytes = await readFile(filePath);
  const mimeType = mimeTypeFor(filePath);
  const dataUrl = `data:${mimeType};base64,${bytes.toString("base64")}`;
  const payload = mimeType === "application/pdf"
    ? { file: dataUrl }
    : { image_url: dataUrl };

  const response = await fetch(OCR_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(300_000),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok || body.error) {
    throw new Error(body.error || `OCR failed with HTTP ${response.status}`);
  }

  return {
    markdown: body.markdown_result || body.md_results || "",
    layout: body.json_result || body.layout_details || null,
    raw: body,
  };
}
```

## Python Integration

```python
import base64
import mimetypes
from pathlib import Path

import requests

OCR_URL = "http://192.168.4.35:5002/glmocr/parse"


def ocr_file(path: str) -> dict:
    file_path = Path(path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    payload = {"file": data_url} if mime_type == "application/pdf" else {"image_url": data_url}

    response = requests.post(OCR_URL, json=payload, timeout=300)
    response.raise_for_status()
    body = response.json()
    if body.get("error"):
        raise RuntimeError(body["error"])

    return {
        "markdown": body.get("markdown_result") or body.get("md_results") or "",
        "layout": body.get("json_result") or body.get("layout_details"),
        "raw": body,
    }
```

## Operational Notes

- Check health before sending work: `GET /health` should return `{"status":"ok"}`.
- OCR can be slow for PDFs because the SDK performs layout detection and region OCR. Use a client timeout of at least 300 seconds for large PDFs.
- The direct SDK server is a Flask development server on the LAN. Treat it as trusted-network only unless you add authentication/reverse proxy controls.
- The service is local/self-hosted. It is configured with MaaS/cloud OCR disabled.
- The SDK layout model must use a single GPU value. Do not pass `cuda:0,1` or `GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES=0,1` as the layout device. Current working config uses `GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES=1`.
- The local OCR model backend can have its own GPU settings; do not copy those multi-GPU values into the SDK layout device field.

## Troubleshooting

Check service state:

```bash
systemctl status glmocr-sdk --no-pager --lines=80
```

Check logs:

```bash
journalctl -u glmocr-sdk --no-pager --lines=120
```

Check listener:

```bash
ss -ltnp | grep ':5002'
```

Common failures:

- `RuntimeError: Invalid device string: 'cuda:0,1'`: the SDK layout device received a comma-separated GPU list. Set `GLMOCR_LAYOUT_CUDA_VISIBLE_DEVICES=1` or another single GPU id, or set `GLMOCR_LAYOUT_DEVICE=cpu`.
- HTTP 404 on `/`: expected. Use `/health` for health checks and `/glmocr/parse` for OCR.
- HTTP 400 `No images or file provided`: request JSON did not include `images`, `file`, `image_url`, or `image_base64`.
- HTTP 500 parse errors: check that the OCR backend on port `8009` is running and that PDFs/images are valid base64 data URLs.
