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

from dataset_manager import DatasetManager
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
        self.dataset_id: str | None = None
        self.load_error: str | None = None
        self.query_count = 0
        self.total_query_ms = 0.0
        self.last_query_ms = 0.0
        self.last_search_ms = 0.0

    def ensure_searcher(self, dataset_manager: DatasetManager) -> AnnSearcher:
        with self._lock:
            dataset = dataset_manager.get_active_dataset()
            dataset_id = str(dataset["id"])
            if self.searcher is not None and self.dataset_id == dataset_id:
                return self.searcher

            self.searcher = AnnSearcher(
                str(dataset["index_path"]),
                str(dataset["vectors_path"]),
                str(dataset["metadata_path"]),
                str(dataset["cell_ids_path"]),
            )
            self.dataset_id = dataset_id
            self.load_error = None
            return self.searcher

    def clear_searcher(self) -> None:
        with self._lock:
            self.searcher = None
            self.dataset_id = None

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
                "dataset_id": self.dataset_id,
                "load_error": self.load_error,
            }


state = SearchState()
dataset_manager = DatasetManager(
    legacy_vectors=DEFAULT_VECTORS_PATH,
    legacy_metadata=DEFAULT_METADATA_PATH,
    legacy_cell_ids=DEFAULT_CELL_IDS_PATH,
    legacy_index=DEFAULT_INDEX_PATH,
)


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

    @app.errorhandler(ValueError)
    def _handle_bad_request(exc: ValueError):
        payload = {
            "error": "Bad Request",
            "message": str(exc),
            "status": 400,
        }
        return jsonify(payload), 400

    @app.errorhandler(KeyError)
    def _handle_not_found(exc: KeyError):
        payload = {
            "error": "Not Found",
            "message": str(exc).strip("'"),
            "status": 404,
        }
        return jsonify(payload), 404

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

    @app.get("/status")
    def status():
        datasets = dataset_manager.list_datasets()
        return jsonify({
            "search": state.snapshot(),
            "active_dataset_id": datasets["active_dataset_id"],
            "dataset_count": len(datasets["datasets"]),
        })

    @app.get("/datasets")
    def list_datasets():
        return jsonify(dataset_manager.list_datasets())

    @app.post("/datasets")
    def upload_dataset():
        uploaded_file = request.files.get("file")
        dataset = dataset_manager.create_from_upload(
            uploaded_file=uploaded_file,
            name=request.form.get("name"),
            embedding=request.form.get("embedding", "X_pca"),
            dims=_parse_int(request.form.get("dims", 30), "dims"),
            obs_cols=request.form.get("obs_cols", "cell_type,disease,AgeGroup"),
            l2=_parse_bool(request.form.get("l2", "true")),
            index_type=request.form.get("index_type", "hnsw"),
            metric=request.form.get("metric", "l2"),
            M=_parse_int(request.form.get("M", 32), "M"),
            ef=_parse_int(request.form.get("ef", 200), "ef"),
            activate=_parse_bool(request.form.get("activate", "true")),
        )
        state.clear_searcher()
        return jsonify({"message": "数据集上传并构建完成", "dataset": dataset}), 201

    @app.post("/datasets/<dataset_id>/activate")
    def activate_dataset(dataset_id: str):
        dataset = dataset_manager.activate_dataset(dataset_id)
        state.clear_searcher()
        return jsonify({"message": "已切换活动数据集", "dataset": dataset})

    @app.delete("/datasets/<dataset_id>")
    def delete_dataset(dataset_id: str):
        result = dataset_manager.delete_dataset(dataset_id)
        state.clear_searcher()
        return jsonify({"message": "数据集已删除", **result})

    @app.route("/search", methods=["GET", "POST"])
    def search():
        request_started = time.perf_counter()
        searcher = state.ensure_searcher(dataset_manager)
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


def _parse_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


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
