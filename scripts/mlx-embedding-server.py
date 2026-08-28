#!/usr/bin/env python3
"""Small OpenAI-compatible MLX embedding server for Apple Silicon."""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import struct
import time
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlx_embeddings.utils import load


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str = "float"
    dimensions: int | None = None


MODEL_PATH = os.environ.get("MLX_EMBED_MODEL_PATH", "")
MODEL_ALIAS = os.environ.get("EMBED_MODEL_NAME", "embed")
MAX_BATCH = int(os.environ.get("MLX_EMBED_MAX_BATCH", "16"))
MAX_LENGTH = int(os.environ.get("MLX_EMBED_MAX_LENGTH", "8192"))

_model = None
_tokenizer = None
_load_lock = asyncio.Lock()
_inference_lock = asyncio.Lock()
_started_at = time.time()
_requests = 0


def _load_model_sync():
    global _model, _tokenizer
    if _model is None:
        if not MODEL_PATH:
            raise RuntimeError("MLX_EMBED_MODEL_PATH is not configured")
        _model, _tokenizer = load(MODEL_PATH)
    return _model, _tokenizer


async def _ensure_model():
    if _model is None:
        async with _load_lock:
            if _model is None:
                await asyncio.to_thread(_load_model_sync)
    return _model, _tokenizer


def _embed_sync(texts: list[str], dimensions: int | None):
    model, tokenizer = _load_model_sync()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    input_ids = mx.array(encoded["input_ids"])
    attention_mask = mx.array(encoded["attention_mask"])
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.text_embeds
    mx.eval(embeddings)
    # NumPy cannot consume MLX's bfloat16 buffer format directly.  Cast on the
    # MLX side first, then cross the framework boundary as ordinary float32.
    vectors = np.asarray(embeddings.astype(mx.float32))
    if dimensions is not None:
        if dimensions < 1 or dimensions > vectors.shape[1]:
            raise ValueError(
                f"dimensions must be between 1 and {vectors.shape[1]}"
            )
        vectors = vectors[:, :dimensions]
    # The quantized model normalizes in bfloat16, which can leave norms a few
    # thousandths from one.  Finish in float32 so clients always receive unit
    # vectors, including after Matryoshka dimension truncation.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, np.finfo(np.float32).eps)
    prompt_tokens = int(np.asarray(encoded["attention_mask"]).sum())
    return vectors, prompt_tokens


def _base64_vector(vector: np.ndarray) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Preload at service start so health means the weights really work.
    await _ensure_model()
    yield
    mx.clear_cache()


app = FastAPI(title="Qwen3 MLX Embeddings", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "backend": "mlx-embeddings",
        "model": MODEL_ALIAS,
        "model_path": MODEL_PATH,
        "loaded": _model is not None,
        "requests": _requests,
        "uptime_seconds": round(time.time() - _started_at, 3),
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ALIAS,
                "object": "model",
                "created": int(_started_at),
                "owned_by": "local-mlx",
            }
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    global _requests
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise HTTPException(status_code=400, detail="input must contain text")
    if len(texts) > MAX_BATCH:
        raise HTTPException(
            status_code=400, detail=f"batch exceeds MLX_EMBED_MAX_BATCH={MAX_BATCH}"
        )
    if request.model not in (None, MODEL_ALIAS, MODEL_PATH):
        raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")
    if request.encoding_format not in ("float", "base64"):
        raise HTTPException(status_code=400, detail="encoding_format must be float or base64")

    await _ensure_model()
    try:
        async with _inference_lock:
            vectors, prompt_tokens = await asyncio.to_thread(
                _embed_sync, texts, request.dimensions
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    _requests += 1
    data = []
    for index, vector in enumerate(vectors):
        value = _base64_vector(vector) if request.encoding_format == "base64" else vector.tolist()
        data.append({"object": "embedding", "index": index, "embedding": value})
    return {
        "object": "list",
        "data": data,
        "model": MODEL_ALIAS,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("LISTEN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EMBED_PORT", "8005")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1, server_header=False)


if __name__ == "__main__":
    main()
