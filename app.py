from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    session,
    redirect,
    url_for,
)
from werkzeug.exceptions import HTTPException

from dataset_manager import DatasetManager
from multi_search import MultiAnnSearcher
from search import AnnSearcher
from user_manager import UserManager, DEFAULT_SESSION_SECRET
from visualize import ScatterDataProvider
from ai_analyzer import (
    analyze_search_result,
    build_dataset_schema,
    execute_query_plan,
    parse_natural_query,
)

DEFAULT_INDEX_PATH = os.getenv("ANN_INDEX_PATH", "indices/hnsw_M32_ef200.index")
DEFAULT_VECTORS_PATH = os.getenv("ANN_VECTORS_PATH", "results/vectors.npy")
DEFAULT_METADATA_PATH = os.getenv("ANN_METADATA_PATH", "results/obs_metadata.csv")
DEFAULT_CELL_IDS_PATH = os.getenv("ANN_CELL_IDS_PATH", "results/cell_ids.npy")
DEFAULT_EVALUATION_REPORT_PATH = Path(
    os.getenv("ANN_EVALUATION_REPORT_PATH", "evaluation_report.json")
)
MAX_K = int(os.getenv("ANN_MAX_K", "100"))


# ───────────────────────── 认证辅助 ─────────────────────────

def login_required(f: Callable) -> Callable:
    """要求用户已登录的装饰器。未登录返回 401 或重定向到登录页。"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "未登录，请先登录"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f: Callable) -> Callable:
    """要求用户为管理员的装饰器。"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "未登录，请先登录"}), 401
            return redirect(url_for("login_page"))
        if session.get("role") != "admin":
            return jsonify({"error": "无权限，需要管理员账号"}), 403
        return f(*args, **kwargs)
    return decorated


# ───────────────────────── Search State ─────────────────────────

class SearchState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.searcher: AnnSearcher | None = None
        self.dataset_id: str | None = None
        self.load_error: str | None = None
        self._reset_metrics_unlocked()

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
            self._reset_metrics_unlocked()

    def record_query(self, query_ms: float, search_ms: float) -> dict[str, float | int]:
        with self._lock:
            now = time.time()
            self.query_count += 1
            self.total_query_ms += query_ms
            self.total_search_ms += search_ms
            self.last_query_ms = query_ms
            self.last_search_ms = search_ms
            self.last_query_at = now
            if self.first_query_at is None:
                self.first_query_at = now
            self.query_latency_ms.append(float(query_ms))
            self.search_latency_ms.append(float(search_ms))
            avg_ms = self.total_query_ms / self.query_count if self.query_count else 0.0
            avg_search_ms = self.total_search_ms / self.query_count if self.query_count else 0.0
            return {
                "query_count": self.query_count,
                "last_query_ms": round(self.last_query_ms, 3),
                "last_search_ms": round(self.last_search_ms, 3),
                "avg_query_ms": round(avg_ms, 3),
                "avg_search_ms": round(avg_search_ms, 3),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            avg_ms = self.total_query_ms / self.query_count if self.query_count else 0.0
            avg_search_ms = self.total_search_ms / self.query_count if self.query_count else 0.0
            p95_query_ms = (
                float(np.percentile(list(self.query_latency_ms), 95))
                if self.query_latency_ms
                else 0.0
            )
            p95_search_ms = (
                float(np.percentile(list(self.search_latency_ms), 95))
                if self.search_latency_ms
                else 0.0
            )
            uptime_seconds = (
                max((self.last_query_at or time.time()) - self.first_query_at, 0.0)
                if self.first_query_at is not None
                else 0.0
            )
            qps = (self.query_count / uptime_seconds) if uptime_seconds > 0 else 0.0
            return {
                "query_count": self.query_count,
                "last_query_ms": round(self.last_query_ms, 3),
                "last_search_ms": round(self.last_search_ms, 3),
                "avg_query_ms": round(avg_ms, 3),
                "avg_search_ms": round(avg_search_ms, 3),
                "p95_query_ms": round(p95_query_ms, 3),
                "p95_search_ms": round(p95_search_ms, 3),
                "qps": round(qps, 3),
                "first_query_at": self.first_query_at,
                "last_query_at": self.last_query_at,
                "ready": self.searcher is not None,
                "dataset_id": self.dataset_id,
                "load_error": self.load_error,
            }

    def _reset_metrics_unlocked(self) -> None:
        self.query_count = 0
        self.total_query_ms = 0.0
        self.total_search_ms = 0.0
        self.last_query_ms = 0.0
        self.last_search_ms = 0.0
        self.first_query_at: float | None = None
        self.last_query_at: float | None = None
        self.query_latency_ms: deque[float] = deque(maxlen=1000)
        self.search_latency_ms: deque[float] = deque(maxlen=1000)


state = SearchState()
dataset_manager = DatasetManager(
    legacy_vectors=DEFAULT_VECTORS_PATH,
    legacy_metadata=DEFAULT_METADATA_PATH,
    legacy_cell_ids=DEFAULT_CELL_IDS_PATH,
    legacy_index=DEFAULT_INDEX_PATH,
)
multi_searcher = MultiAnnSearcher()
user_manager = UserManager()
user_manager.ensure_admin_exists()
scatter_provider = ScatterDataProvider()
evaluation_report_lock = threading.Lock()


# ───────────────────────── App Factory ─────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    app.config["SECRET_KEY"] = os.getenv("ANN_SESSION_SECRET", DEFAULT_SESSION_SECRET)
    app.config["SESSION_COOKIE_NAME"] = "ann_session"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

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

    # ── 请求计时 ──
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

    # ── 错误处理 ──
    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException):
        payload = {"error": exc.name, "message": exc.description, "status": exc.code}
        return jsonify(payload), exc.code

    @app.errorhandler(Exception)
    def _handle_unexpected_exception(exc: Exception):
        app.logger.exception("Unhandled server error: %s", exc)
        return jsonify({"error": "Internal Server Error", "message": str(exc), "status": 500}), 500

    @app.errorhandler(ValueError)
    def _handle_bad_request(exc: ValueError):
        return jsonify({"error": "Bad Request", "message": str(exc), "status": 400}), 400

    @app.errorhandler(KeyError)
    def _handle_not_found(exc: KeyError):
        return jsonify({"error": "Not Found", "message": str(exc).strip("'"), "status": 404}), 404

    @app.errorhandler(FileNotFoundError)
    def _handle_missing_dependency(exc: FileNotFoundError):
        app.logger.warning("Unavailable dependency or artifact: %s", exc)
        return jsonify({"error": "Service Unavailable", "message": str(exc), "status": 503}), 503

    # ═══════════════════════════════════════════════════════
    #  认证页面路由
    # ═══════════════════════════════════════════════════════

    @app.get("/login")
    def login_page():
        if "user" in session:
            return redirect(url_for("index"))
        return render_template("login.html")

    @app.get("/register")
    def register_page():
        if "user" in session:
            return redirect(url_for("index"))
        return render_template("register.html")

    @app.get("/admin")
    @admin_required
    def admin_page():
        return render_template("admin.html")

    # ═══════════════════════════════════════════════════════
    #  认证 API
    # ═══════════════════════════════════════════════════════

    @app.post("/api/auth/register")
    def api_register():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        username = payload.get("username", "").strip()
        password = payload.get("password", "")
        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400
        try:
            user = user_manager.register(username, password)
            session["user"] = user.username
            session["role"] = user.role
            return jsonify({"message": "注册成功", "user": user.to_safe_dict()}), 201
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

    @app.post("/api/auth/login")
    def api_login():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "请求体必须是 JSON 对象"}), 400
        username = payload.get("username", "").strip()
        password = payload.get("password", "")
        if not username or not password:
            return jsonify({"error": "用户名和密码不能为空"}), 400
        user = user_manager.authenticate(username, password)
        if user is None:
            return jsonify({"error": "用户名或密码错误"}), 401
        session["user"] = user.username
        session["role"] = user.role
        return jsonify({"message": "登录成功", "user": user.to_safe_dict()})

    @app.post("/api/auth/logout")
    def api_logout():
        session.clear()
        return jsonify({"message": "已退出登录"})

    @app.get("/api/auth/me")
    def api_me():
        if "user" not in session:
            return jsonify({"error": "未登录"}), 401
        username = session["user"]
        user = user_manager.get_user(username)
        if user is None:
            session.clear()
            return jsonify({"error": "用户不存在"}), 401
        return jsonify({"user": user.to_safe_dict()})

    # ═══════════════════════════════════════════════════════
    #  管理员 API
    # ═══════════════════════════════════════════════════════

    @app.get("/api/admin/users")
    @admin_required
    def api_admin_list_users():
        return jsonify({"users": user_manager.list_users()})

    @app.delete("/api/admin/users/<username>")
    @admin_required
    def api_admin_delete_user(username: str):
        if username == session.get("user"):
            return jsonify({"error": "不能删除自己的账号"}), 400
        try:
            user_manager.delete_user(username)
            return jsonify({"message": f"用户 '{username}' 已删除"})
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/admin/users/<username>/role")
    @admin_required
    def api_admin_change_role(username: str):
        payload = request.get_json(silent=True) or {}
        new_role = payload.get("role", "")
        if new_role not in ("user", "admin"):
            return jsonify({"error": "角色必须是 'user' 或 'admin'"}), 400
        try:
            user = user_manager.change_role(username, new_role)
            return jsonify({"message": "角色已更新", "user": user.to_safe_dict()})
        except (ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/admin/users/<username>/reset-password")
    @admin_required
    def api_admin_reset_password(username: str):
        from werkzeug.security import generate_password_hash
        payload = request.get_json(silent=True) or {}
        new_password = payload.get("new_password", "")
        if len(new_password) < 6:
            return jsonify({"error": "密码至少需要 6 个字符"}), 400
        user = user_manager.get_user(username)
        if user is None:
            return jsonify({"error": f"用户 '{username}' 不存在"}), 404
        user_manager._users[username].password_hash = generate_password_hash(new_password)
        user_manager._save()
        return jsonify({"message": f"用户 '{username}' 密码已重置"})

    # ═══════════════════════════════════════════════════════
    #  应用页面路由
    # ═══════════════════════════════════════════════════════

    @app.get("/")
    @login_required
    def index():
        return render_template("index.html")

    @app.get("/status")
    @login_required
    def status():
        datasets = dataset_manager.list_datasets()
        return jsonify({
            "search": state.snapshot(),
            "active_dataset_id": datasets["active_dataset_id"],
            "dataset_count": len(datasets["datasets"]),
            "user": {"username": session.get("user"), "role": session.get("role")},
        })

    @app.get("/metrics")
    @login_required
    def metrics():
        dataset = dataset_manager.get_active_dataset()
        report = _ensure_evaluation_report_for_dataset(dataset)
        return jsonify(_build_metrics_payload(dataset, report))

    @app.get("/datasets")
    @login_required
    def list_datasets():
        return jsonify(dataset_manager.list_datasets())

    @app.post("/datasets")
    @admin_required
    def upload_dataset():
        uploaded_file = request.files.get("file")
        dataset = dataset_manager.create_from_upload(
            uploaded_file=request.files.get("file"),
            name=request.form.get("name"),
            dims=_parse_int(request.form.get("dims", 30), "dims"),
            index_type=request.form.get("index_type", "hnsw"),
            nlist=_parse_optional_int(request.form.get("nlist"), "nlist"),
            activate=_parse_bool(request.form.get("activate", True)),
        )
        state.clear_searcher()
        return jsonify({"message": "数据集上传并构建完成", "dataset": dataset}), 201

    @app.post("/datasets/<dataset_id>/activate")
    @login_required
    def activate_dataset(dataset_id: str):
        dataset = dataset_manager.activate_dataset(dataset_id)
        state.clear_searcher()
        return jsonify({"message": "已切换活动数据集", "dataset": dataset})

    @app.delete("/datasets/<dataset_id>")
    @admin_required
    def delete_dataset(dataset_id: str):
        result = dataset_manager.delete_dataset(dataset_id)
        state.clear_searcher()
        return jsonify({"message": "数据集已删除", **result})
    
    @app.get("/scatter_data")
    @login_required
    def scatter_data():
        dataset = dataset_manager.get_active_dataset()

        max_points_raw = request.args.get("max_points")
        if max_points_raw:
            max_points = _parse_int(max_points_raw, "max_points")
        else:
            max_points = None

        fields_raw = request.args.get("fields", "")
        fields = [x.strip() for x in fields_raw.split(",") if x.strip()] or None

        payload = scatter_provider.get_scatter_data(
            dataset,
            max_points=max_points,
            metadata_fields=fields,
        )

        return jsonify(payload)


    @app.post("/scatter_search")
    @login_required
    def scatter_search():
        request_started = time.perf_counter()

        searcher = state.ensure_searcher(dataset_manager)

        payload = request.get_json(silent=True) or {}

        cell_id = payload.get("cell_id")
        if not cell_id:
            return jsonify({"error": "请提供 cell_id"}), 400

        k = _parse_k(payload.get("k", 10))
        filters = _parse_filters(payload.get("filters"))
        search_params = _parse_search_params(payload.get("search_params"))

        result = searcher.search_by_cell_id(
            str(cell_id),
            k=k,
            filters=filters,
            search_params=search_params,
        )

        total_ms = (time.perf_counter() - request_started) * 1000

        metrics = state.record_query(
            total_ms,
            float(result["time_ms"]),
        )

        return jsonify(
            {
                **result,
                "request_time_ms": round(total_ms, 3),
                "metrics": metrics,
            }
        )

    @app.route("/search", methods=["GET", "POST"])
    @login_required
    def search():
        request_started = time.perf_counter()
        searcher = state.ensure_searcher(dataset_manager)
        payload = _extract_payload()
        cell_id = payload.get("cell_id")
        vector = payload.get("vector")
        k = _parse_k(payload.get("k", 10))
        filters = _parse_filters(payload.get("filters"))
        search_params = _parse_search_params(payload.get("search_params"))
        if cell_id and vector is not None:
            return jsonify({"error": "cell_id 和 vector 只能提供一个"}), 400
        if not cell_id and vector is None:
            return jsonify({"error": "请提供 cell_id 或 vector"}), 400
        if cell_id:
            app.logger.info("search request by cell_id=%s, k=%s, filters=%s", cell_id, k, filters)
            result = searcher.search_by_cell_id(
                str(cell_id),
                k=k,
                filters=filters,
                search_params=search_params,
            )
        else:
            query_vector = _parse_vector(vector)
            app.logger.info("search request by vector, k=%s, dim=%s, filters=%s", k, query_vector.shape[-1], filters)
            result = searcher.search_by_vector(
                query_vector,
                k=k,
                filters=filters,
                search_params=search_params,
            )
        total_ms = (time.perf_counter() - request_started) * 1000
        metrics = state.record_query(total_ms, float(result["time_ms"]))
        return jsonify({**result, "request_time_ms": round(total_ms, 3), "metrics": metrics})

    @app.post("/ai/query")
    @login_required
    def ai_query():
        request_started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "请提供自然语言查询 question"}), 400

        searcher = state.ensure_searcher(dataset_manager)
        dataset = dataset_manager.get_active_dataset()
        schema = build_dataset_schema(searcher)
        plan = parse_natural_query(question, schema)
        result = execute_query_plan(searcher, plan)
        analysis = analyze_search_result(result, dataset)

        total_ms = (time.perf_counter() - request_started) * 1000
        metrics = state.record_query(total_ms, float(result.get("time_ms", 0.0)))
        return jsonify(
            {
                **result,
                "analysis": analysis,
                "request_time_ms": round(total_ms, 3),
                "metrics": metrics,
            }
        )

    @app.post("/ai/analyze")
    @login_required
    def ai_analyze():
        request_started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        dataset = dataset_manager.get_active_dataset()

        if isinstance(payload.get("search_result"), dict):
            result = payload["search_result"]
        else:
            searcher = state.ensure_searcher(dataset_manager)
            cell_id = payload.get("cell_id")
            vector = payload.get("vector")
            k = _parse_k(payload.get("k", 10))
            filters = _parse_filters(payload.get("filters"))
            if cell_id and vector is not None:
                return jsonify({"error": "cell_id 和 vector 只能提供一个"}), 400
            if not cell_id and vector is None:
                return jsonify({"error": "请提供 search_result、cell_id 或 vector"}), 400
            if cell_id:
                result = searcher.search_by_cell_id(str(cell_id), k=k, filters=filters)
            else:
                result = searcher.search_by_vector(_parse_vector(vector), k=k, filters=filters)

        analysis = analyze_search_result(result, dataset)
        total_ms = (time.perf_counter() - request_started) * 1000
        return jsonify(
            {
                "analysis": analysis,
                "request_time_ms": round(total_ms, 3),
            }
        )

    # ═══════════════════════════════════════════════════════
    #  多数据集联合检索 API
    # ═══════════════════════════════════════════════════════

    @app.post("/api/multi_search")
    @login_required
    def multi_search():
        request_started = time.perf_counter()
        payload = request.get_json(silent=True) or {}

        # --- 确定检索目标数据集 ---
        raw_dataset_ids = payload.get("dataset_ids")
        if raw_dataset_ids and isinstance(raw_dataset_ids, list) and len(raw_dataset_ids) > 0:
            target_ids = [str(did) for did in raw_dataset_ids]
        else:
            # 未指定时用所有已加载的数据集
            loaded = multi_searcher.get_loaded_datasets()
            target_ids = [ds["id"] for ds in loaded]
            if not target_ids:
                return jsonify({"error": "没有已加载的数据集，请先加载数据集或指定 dataset_ids"}), 400

        # --- 加载未加载的数据集 ---
        for ds_id in target_ids:
            if not multi_searcher.is_loaded(ds_id):
                try:
                    dataset = dataset_manager.get_active_dataset()
                    all_datasets = dataset_manager.list_datasets().get("datasets", [])
                    match = next((d for d in all_datasets if str(d["id"]) == ds_id), None)
                    if match is None:
                        # 尝试用 active 数据集
                        if str(dataset["id"]) == ds_id:
                            match = dataset
                    if match:
                        multi_searcher.load_dataset(match)
                    else:
                        return jsonify({"error": f"数据集不存在: {ds_id}"}), 404
                except Exception as exc:
                    app.logger.warning("加载数据集 %s 失败: %s", ds_id, exc)
                    return jsonify({"error": f"无法加载数据集 {ds_id}: {exc}"}), 400

        # --- 获取查询向量 ---
        cell_id = payload.get("cell_id")
        vector_raw = payload.get("vector")

        if cell_id and vector_raw is not None:
            return jsonify({"error": "cell_id 和 vector 只能提供一个"}), 400
        if not cell_id and vector_raw is None:
            return jsonify({"error": "请提供 cell_id 或 vector"}), 400

        if cell_id:
            # 在已加载数据集中查找 cell_id
            found_ds, found_vec = multi_searcher.find_cell_id(str(cell_id), target_ids)
            if found_ds is None:
                return jsonify({
                    "error": f"细胞ID '{cell_id}' 在指定数据集中未找到。"
                             f"请确认该细胞属于已加载的数据集，或改用向量方式查询"
                }), 404
            app.logger.info(
                "multi_search by cell_id=%s (found in dataset=%s), target_datasets=%s",
                cell_id, found_ds, target_ids,
            )
            query_vector = found_vec
        else:
            query_vector = _parse_vector(vector_raw)
            app.logger.info(
                "multi_search by vector, dim=%s, target_datasets=%s",
                query_vector.shape[-1], target_ids,
            )

        # --- 解析参数 ---
        k = _parse_k(payload.get("k", 10))
        filters = _parse_filters(payload.get("filters"))
        search_params = _parse_search_params(payload.get("search_params"))

        # --- 执行联合检索 ---
        result = multi_searcher.search(
            vector=query_vector,
            dataset_ids=target_ids,
            k=k,
            filters=filters,
            search_params=search_params,
            query_cell_id=cell_id,
        )

        total_ms = (time.perf_counter() - request_started) * 1000
        result["request_time_ms"] = round(total_ms, 3)

        return jsonify(result)

    @app.post("/api/multi_load")
    @login_required
    def multi_load_datasets():
        """将指定数据集加载到联合检索器中。

        请求体 JSON:
        {
            "dataset_ids": ["ds1", "ds2", ...],
            "activate": true       // 是否也同时切换到新加载（仅首项）
        }
        """
        payload = request.get_json(silent=True) or {}
        raw_ids = payload.get("dataset_ids")
        if not raw_ids or not isinstance(raw_ids, list) or len(raw_ids) == 0:
            return jsonify({"error": "请提供 dataset_ids 列表"}), 400

        all_datasets = dataset_manager.list_datasets().get("datasets", [])
        id_to_ds = {str(d["id"]): d for d in all_datasets}
        # 也包含 active 数据集
        try:
            active_ds = dataset_manager.get_active_dataset()
            id_to_ds[str(active_ds["id"])] = active_ds
        except FileNotFoundError:
            pass

        loaded: list[dict] = []
        errors: list[dict] = []
        for ds_id in raw_ids:
            did = str(ds_id)
            if multi_searcher.is_loaded(did):
                loaded.append({"id": did, "status": "already_loaded"})
                continue
            ds_info = id_to_ds.get(did)
            if ds_info is None:
                errors.append({"id": did, "error": "数据集不存在"})
                continue
            try:
                multi_searcher.load_dataset(ds_info)
                loaded.append({"id": did, "name": ds_info.get("name", did), "status": "loaded"})
            except Exception as exc:
                errors.append({"id": did, "error": str(exc)})

        return jsonify({
            "message": f"已加载 {len(loaded)} 个数据集，{len(errors)} 个失败",
            "loaded": loaded,
            "errors": errors if errors else None,
        })

    @app.delete("/api/multi_load/<dataset_id>")
    @login_required
    def multi_unload_dataset(dataset_id: str):
        """从联合检索器中卸载指定数据集。"""
        if not multi_searcher.is_loaded(dataset_id):
            return jsonify({"error": f"数据集 {dataset_id} 未在加载状态"}), 404

        multi_searcher.unload_dataset(dataset_id)
        return jsonify({
            "message": f"数据集 {dataset_id} 已卸载",
            "loaded": [d["id"] for d in multi_searcher.get_loaded_datasets()],
        })

    @app.get("/api/multi_status")
    @login_required
    def multi_status():
        """查询联合检索器当前加载的数据集状态。"""
        loaded = multi_searcher.get_loaded_datasets()
        return jsonify({
            "loaded_count": len(loaded),
            "loaded_datasets": loaded,
        })

    @app.post("/api/multi_merge")
    @admin_required
    def multi_merge_index():
        """将多个数据集的向量合并为一个统一索引（高级功能）。

        请求体 JSON:
        {
            "dataset_ids": ["ds1", "ds2"],
            "output_path": "indices/merged/index",
            "index_type": "hnsw",
            "activate": true
        }
        """
        payload = request.get_json(silent=True) or {}
        dataset_ids = payload.get("dataset_ids")
        if not dataset_ids or not isinstance(dataset_ids, list) or len(dataset_ids) < 2:
            return jsonify({"error": "需要至少 2 个数据集 ID"}), 400

        output_path = payload.get("output_path", "indices/merged/merged_hnsw.index")
        index_type = payload.get("index_type", "hnsw")
        activate = payload.get("activate", False)

        if index_type not in ("hnsw", "flat", "ivf_hnsw"):
            return jsonify({"error": "index_type 仅支持 hnsw/flat/ivf_hnsw"}), 400

        # 确保数据集均已加载
        for ds_id in dataset_ids:
            if not multi_searcher.is_loaded(ds_id):
                return jsonify({"error": f"数据集 {ds_id} 未加载，请先 POST /api/multi_load"}), 400

        try:
            result = multi_searcher.build_merged_index(
                dataset_ids=dataset_ids,
                output_path=output_path,
                index_type=index_type,
            )
            app.logger.info("合并索引创建完成: %s", output_path)

            # 可选：将合并索引作为新数据集注册
            if activate:
                merged_ds = _register_merged_dataset(result, output_path)
                msg = "合并索引已创建并激活"
                return jsonify({"message": msg, "merge_result": result, "dataset": merged_ds}), 201

            return jsonify({"message": "合并索引已创建", "merge_result": result})

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("合并索引创建失败: %s", exc)
            return jsonify({"error": f"合并索引创建失败: {exc}"}), 500

    # ═══════════════════════════════════════════════════════

    @app.get("/filter_options")
    @login_required
    def filter_options():
        dataset = dataset_manager.get_active_dataset()
        options = _load_filter_options(
            dataset["metadata_path"],
            fields=["cell_type", "disease", "AgeGroup", "sex", "Treatment", "Phase"],
        )
        return jsonify({
            "dataset_id": dataset["id"],
            "options": options,
        })

    return app


# ───────────────────────── 辅助函数 ─────────────────────────

def _load_filter_options(
    metadata_path: str | Path,
    fields: list[str] | None = None,
) -> dict[str, list[str]]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata 文件不存在: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}

        target_fields = fields or [name for name in reader.fieldnames if name != "cell_id"]
        options: dict[str, set[str]] = {name: set() for name in target_fields}

        for row in reader:
            for name in target_fields:
                value = row.get(name)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    options[name].add(text)

    return {
        name: sorted(values, key=lambda item: item.lower())
        for name, values in options.items()
    }


def _build_metrics_payload(dataset: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    live = state.snapshot()
    dataset_info = _build_dataset_metrics_info(dataset)
    report_dataset_id = str(report.get("dataset", {}).get("id", "")) if isinstance(report, dict) else ""
    has_matching_report = bool(report_dataset_id) and report_dataset_id == str(dataset["id"])
    static_perf = report.get("performance_summary", {}) if has_matching_report else {}
    report_path = _evaluation_report_path_for_dataset(dataset)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": dataset_info,
        "performance_summary": {
            "query_count": live["query_count"],
            "ann_avg_latency_ms": live["avg_search_ms"],
            "ann_p95_latency_ms": live["p95_search_ms"],
            "ann_qps": live["qps"],
            "index_size_mb": dataset_info["index_size_mb"],
            "ground_truth_avg_latency_ms": static_perf.get("ground_truth_avg_latency_ms"),
            "avg_request_latency_ms": live["avg_query_ms"],
            "p95_request_latency_ms": live["p95_query_ms"],
            "last_query_ms": live["last_query_ms"],
            "last_search_ms": live["last_search_ms"],
        },
        "recall": report.get("recall", {}) if has_matching_report else {},
        "ann_search": report.get("ann_search", {}) if has_matching_report else {},
        "ground_truth": report.get("ground_truth", {}) if has_matching_report else {},
        "live_metrics": live,
        "evaluation_report": {
            "available": has_matching_report,
            "path": str(report_path),
            "dataset_id": report_dataset_id or None,
            "generated_at": report.get("generated_at") if has_matching_report else None,
        },
    }


def _build_dataset_metrics_info(dataset: dict[str, Any]) -> dict[str, Any]:
    summary = _load_json_file(dataset.get("summary_path"))
    embedding = summary.get("embedding", {}) if isinstance(summary, dict) else {}
    embedding_shape = embedding.get("shape", []) if isinstance(embedding, dict) else []
    index_path = Path(str(dataset["index_path"]))

    total_vectors = dataset.get("n_obs") or summary.get("n_obs") or 0
    dimension = summary.get("dimension") or (embedding_shape[1] if len(embedding_shape) >= 2 else 0)

    return {
        "id": str(dataset["id"]),
        "name": str(dataset.get("name", dataset["id"])),
        "metric": str(dataset.get("metric", "unknown")),
        "index_type": str(dataset.get("index_type", "unknown")),
        "total_vectors": int(total_vectors) if total_vectors else 0,
        "dimension": int(dimension) if dimension else 0,
        "index_size_mb": round(index_path.stat().st_size / (1024 * 1024), 3) if index_path.exists() else 0.0,
    }


def _evaluation_report_path_for_dataset(dataset: dict[str, Any]) -> Path:
    if str(dataset.get("id")) == "default":
        return DEFAULT_EVALUATION_REPORT_PATH
    summary_path = dataset.get("summary_path")
    if summary_path:
        return Path(str(summary_path)).with_name("evaluation_report.json")
    return DEFAULT_EVALUATION_REPORT_PATH


def _ensure_evaluation_report_for_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    report_path = _evaluation_report_path_for_dataset(dataset)
    report = _load_json_file(report_path)
    if str(report.get("dataset", {}).get("id", "")) == str(dataset["id"]):
        return report

    with evaluation_report_lock:
        report = _load_json_file(report_path)
        if str(report.get("dataset", {}).get("id", "")) == str(dataset["id"]):
            return report

        from evaluate import (
            DEFAULT_RANDOM_SEED,
            DEFAULT_SAMPLE_SIZE,
            _ensure_faiss_available,
            evaluate_dataset,
        )

        _ensure_faiss_available()
        generated = evaluate_dataset(
            dataset=dataset,
            sample_size=DEFAULT_SAMPLE_SIZE,
            seed=DEFAULT_RANDOM_SEED,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return generated


def _load_json_file(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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

def _parse_search_params(value: Any) -> dict[str, Any] | None:
    """解析前端 ANN 精度控制参数。"""
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"search_params 不是合法 JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise ValueError("search_params 必须是 JSON 对象")

    result: dict[str, Any] = {}

    if "precision_pct" in value and value["precision_pct"] is not None:
        pct = float(value["precision_pct"])
        result["precision_pct"] = max(0.0, min(100.0, pct))

    for key in ("ef_search", "nprobe"):
        if key in value and value[key] is not None:
            iv = int(value[key])
            if iv <= 0:
                raise ValueError(f"{key} 必须大于 0")
            result[key] = iv

    return result or None


def _parse_filters(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"过滤条件的键必须为字符串: {k}")
            # 支持三种格式：字符串精确匹配 / 列表多选 (OR) / 对象范围过滤 {"op": ">", "value": 5}
            if isinstance(v, str):
                continue
            if isinstance(v, list):
                if not v:
                    raise ValueError(f"过滤条件 '{k}' 的列表不能为空")
                if not all(isinstance(item, str) for item in v):
                    raise ValueError(f"过滤条件 '{k}' 的列表元素必须为字符串")
                continue
            if isinstance(v, dict):
                if "op" not in v or "value" not in v:
                    raise ValueError(f"过滤条件 '{k}' 的对象格式必须包含 'op' 和 'value'")
                valid_ops = {">", "<", ">=", "<=", "=="}
                if v["op"] not in valid_ops:
                    raise ValueError(f"过滤条件 '{k}' 的操作符 '{v['op']}' 不合法，支持: {valid_ops}")
                continue
            raise ValueError(f"过滤条件 '{k}' 的值格式不合法，须为字符串、列表或 {{op, value}} 对象")
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"filters 参数不是合法的 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("filters 参数必须是 JSON 对象")
        return _parse_filters(parsed)
    raise ValueError("filters 参数格式不合法，请传入 JSON 对象或字符串")


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

def _parse_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
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
    except Exception as exc:
        raise ValueError("vector 必须是数字数组") from exc
    if vector.ndim not in (1, 2):
        raise ValueError("vector 维度不合法")
    return vector


def _load_evaluation_report() -> dict[str, Any]:
    path = DEFAULT_EVALUATION_REPORT_PATH
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"error": "评估报告尚未生成", "path": str(path)}


def _register_merged_dataset(merge_result: dict, output_path: str) -> dict[str, Any]:
    """将合并索引注册为 dataset_manager 中的一个新数据集。"""
    import time as _time
    from pathlib import Path as _Path

    merged_name = f"merged-{'-'.join(merge_result.get('merged_datasets', ['unknown']))}"
    merged_id = f"merged-{int(_time.time())}"

    output_path_obj = _Path(output_path)

    ds_entry = {
        "id": merged_id,
        "name": merged_name,
        "source_path": "",
        "vectors_path": str(output_path_obj.parent / "merged_vectors.npy"),
        "metadata_path": "",
        "cell_ids_path": str(output_path_obj.parent / "merged_cell_ids.npy"),
        "summary_path": "",
        "index_path": str(output_path_obj),
        "index_type": merge_result.get("index_type", "hnsw"),
        "metric": merge_result.get("metric", "l2"),
        "embedding": {"key": "merged", "shape": [merge_result.get("total_vectors", 0), merge_result.get("dim", 0)]},
        "n_obs": merge_result.get("total_vectors", 0),
        "created_at": int(_time.time()),
        "readonly": True,
    }

    manifest = dataset_manager._load_manifest()
    manifest["datasets"][merged_id] = ds_entry
    manifest["active_dataset_id"] = merged_id
    dataset_manager._write_manifest(manifest)
    return ds_entry


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
    except Exception:
        pass
    return value


app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1").lower() in {"1", "true", "yes", "on"}
    app.run(host="127.0.0.1", port=5000, debug=debug)
