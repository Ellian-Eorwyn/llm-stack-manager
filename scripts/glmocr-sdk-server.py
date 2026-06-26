#!/usr/bin/env python3
"""Local GLM-OCR SDK service wrapper for llm-stack.

The stock SDK server is intentionally small; this wrapper keeps the same
`/glmocr/parse` protocol while making the local/self-hosted mode explicit.
"""

from __future__ import annotations

import base64
import json
import multiprocessing
import time
import traceback
import uuid
from typing import Any

from flask import Flask, jsonify, request

from glmocr.config import load_config
from glmocr.pipeline import Pipeline
from glmocr.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _build_response(json_result: Any, markdown_result: str, results: list[Any]) -> dict[str, Any]:
    layout_visualization = []
    for result in results:
        vis = getattr(result, "layout_vis_images", None)
        if vis:
            layout_visualization.extend([f"page:{idx}" for idx in sorted(vis)])
    return {
        "json_result": json_result,
        "markdown_result": markdown_result,
        "layout_details": json_result,
        "md_results": markdown_result,
        "layout_visualization": layout_visualization,
        "data_info": {"pages": []},
        "usage": {},
        "model": "glm-ocr",
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "created": int(time.time()),
    }


def _data_uri_to_bytes(value: str) -> bytes | None:
    if not value.startswith("data:") or "," not in value:
        return None
    header, payload = value.split(",", 1)
    if ";base64" not in header:
        return None
    return base64.b64decode(payload)


def _input_item_to_content(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:application/pdf"):
        data = _data_uri_to_bytes(value)
        if data is not None:
            return {"type": "image_bytes", "data": data}
    return {"type": "image_url", "image_url": {"url": value}}


def _extract_inputs(data: dict[str, Any]) -> list[str]:
    images = data.get("images", [])
    if isinstance(images, str):
        images = [images]
    if not images and isinstance(data.get("file"), str):
        images = [data["file"]]
    if not images and isinstance(data.get("image_url"), str):
        images = [data["image_url"]]
    if not images and isinstance(data.get("image_base64"), str):
        mime_type = str(data.get("mime_type") or "image/png").strip() or "image/png"
        raw = data["image_base64"].strip()
        images = [raw if raw.startswith("data:") else f"data:{mime_type};base64,{raw}"]
    return [item for item in images if isinstance(item, str) and item]


def create_app(config_path: str | None = None) -> Flask:
    config = load_config(config_path)
    configure_logging(level=config.logging.level)
    pipeline = Pipeline(config=config.pipeline)
    pipeline.start()

    app = Flask(__name__)
    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        data = request.get_json(silent=True) or {}
        inputs = _extract_inputs(data)
        if not inputs:
            return jsonify({"error": "No images or file provided"}), 400

        content = []
        for item in inputs:
            converted = _input_item_to_content(item)
            if converted is not None:
                content.append(converted)
        if not content:
            return jsonify({"error": "No usable images or file provided"}), 400

        request_data = {"messages": [{"role": "user", "content": content}]}
        save_layout_visualization = bool(data.get("need_layout_visualization"))

        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=save_layout_visualization,
                )
            )
            if not results:
                return jsonify(_build_response(None, "", []))
            if len(results) == 1:
                result = results[0]
                return jsonify(
                    _build_response(
                        result.json_result,
                        result.markdown_result or "",
                        results,
                    )
                )
            json_result = [result.json_result for result in results]
            markdown_result = "\n\n---\n\n".join(result.markdown_result or "" for result in results)
            return jsonify(_build_response(json_result, markdown_result, results))
        except Exception as exc:
            logger.error("Parse error: %s", exc)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {exc}"}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.teardown_appcontext
    def _teardown(_exc):
        pass

    return app


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="llm-stack GLM-OCR SDK server")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    app = create_app(args.config)
    cfg = app.config["doc_config"].server
    try:
        app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)
    finally:
        app.config["pipeline"].stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
