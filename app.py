from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request,render_template
from werkzeug.exceptions import HTTPException

from search import AnnSearcher

DEFAULT_INDEX_PATH = os.getenv("ANN_INDEX_PATH", "indices/hnsw_M32_ef200.index")
DEFAULT_VECTORS_PATH = os.getenv("ANN_VECTORS_PATH", "results/vectors.npy")
DEFAULT_METADATA_PATH = os.getenv("ANN_METADATA_PATH", "results/obs_metadata.csv")
DEFAULT_CELL_IDS_PATH = os.getenv("ANN_CELL_IDS_PATH", "results/cell_ids.npy")
MAX_K = int(os.getenv("ANN_MAX_K", "100"))


class SearchState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.searcher: AnnSearcher | None = None
        self.load_error: str | None = None
        self.query_count = 0
        self.total_query_ms = 0.0
        self.last_query_ms = 0.0
        self.last_search_ms = 0.0

    def ensure_searcher(self) -> AnnSearcher:
        with self._lock:
            if self.searcher is not None:
                return self.searcher

            self.searcher = AnnSearcher(
                DEFAULT_INDEX_PATH,
                DEFAULT_VECTORS_PATH,
                DEFAULT_METADATA_PATH,
                DEFAULT_CELL_IDS_PATH,
            )
            self.load_error = None
            return self.searcher

    def record_query(self, query_ms: float, search_ms: float) -> dict[str, float | int]:
        with self._lock:
            self.query_count += 1
            self.total_query_ms += query_ms
            self.last_query_ms = query_ms
            self.last_search_ms = search_ms
            avg_ms = self.total_query_ms / self.query_count if self.query_count else 0.0
            return {
                "query_count": self.query_count,
                "last_query_ms": round(self.last_query_ms, 3),
                "last_search_ms": round(self.last_search_ms, 3),
                "avg_query_ms": round(avg_ms, 3),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_ms = self.total_query_ms / self.query_count if self.query_count else 0.0
            return {
                "query_count": self.query_count,
                "last_query_ms": round(self.last_query_ms, 3),
                "last_search_ms": round(self.last_search_ms, 3),
                "avg_query_ms": round(avg_ms, 3),
                "ready": self.searcher is not None,
                "load_error": self.load_error,
            }


state = SearchState()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False

    logger = logging.getLogger("ann_api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)
    app.logger.propagate = False

    @app.before_request
    def _start_timer() -> None:
        request.environ["ann_api_start"] = time.perf_counter()

    @app.after_request
    def _log_request(response):
        started = request.environ.get("ann_api_start")
        elapsed_ms = None
        if isinstance(started, float):
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            response.headers["X-Request-Latency-Ms"] = str(elapsed_ms)
        app.logger.info(
            "%s %s -> %s%s",
            request.method,
            request.path,
            response.status_code,
            f" ({elapsed_ms} ms)" if elapsed_ms is not None else "",
        )
        return response

    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException):
        payload = {
            "error": exc.name,
            "message": exc.description,
            "status": exc.code,
        }
        return jsonify(payload), exc.code

    @app.errorhandler(Exception)
    def _handle_unexpected_exception(exc: Exception):
        app.logger.exception("Unhandled server error: %s", exc)
        payload = {
            "error": "Internal Server Error",
            "message": str(exc),
            "status": 500,
        }
        return jsonify(payload), 500

    @app.errorhandler(FileNotFoundError)
    def _handle_missing_dependency(exc: FileNotFoundError):
        app.logger.warning("Unavailable dependency or artifact: %s", exc)
        payload = {
            "error": "Service Unavailable",
            "message": str(exc),
            "status": 503,
        }
        return jsonify(payload), 503

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.route("/search", methods=["GET", "POST"])
    def search():
        request_started = time.perf_counter()
        searcher = state.ensure_searcher()
        payload = _extract_payload()
        cell_id = payload.get("cell_id")
        vector = payload.get("vector")
        k = _parse_k(payload.get("k", 10))

        if cell_id and vector is not None:
            return jsonify({"error": "cell_id 和 vector 只能提供一个"}), 400
        if not cell_id and vector is None:
            return jsonify({"error": "请提供 cell_id 或 vector"}), 400

        if cell_id:
            app.logger.info("search request by cell_id=%s, k=%s", cell_id, k)
            result = searcher.search_by_cell_id(str(cell_id), k=k)
        else:
            query_vector = _parse_vector(vector)
            app.logger.info("search request by vector, k=%s, dim=%s", k, query_vector.shape[-1])
            result = searcher.search_by_vector(query_vector, k=k)

        total_ms = (time.perf_counter() - request_started) * 1000
        metrics = state.record_query(total_ms, float(result["time_ms"]))

        response = {
            **result,
            "request_time_ms": round(total_ms, 3),
            "metrics": metrics,
        }
        return jsonify(response)

    return app


def _extract_payload() -> dict[str, Any]:
    if request.method == "GET":
        payload: dict[str, Any] = dict(request.args)
        vector_text = payload.get("vector")
        if vector_text:
            payload["vector"] = vector_text
        return payload

    payload = request.get_json(silent=True)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return payload


def _parse_k(value: Any) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("k 必须是整数") from exc
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if k > MAX_K:
        raise ValueError(f"k 不能大于 {MAX_K}")
    return k


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("vector 不能为空")
        try:
            parsed = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError("vector 字符串必须是逗号分隔的数字") from exc
        return np.asarray(parsed, dtype=np.float32)

    try:
        vector = np.asarray(value, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("vector 必须是数字数组") from exc

    if vector.ndim not in (1, 2):
        raise ValueError("vector 维度不合法")
    return vector


app = create_app()


def _sanitize(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        import pandas as pd

        if isinstance(value, pd.Timestamp):
            return str(value)
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass

    return value


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
