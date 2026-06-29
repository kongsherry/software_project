from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc


class ScatterDataProvider:
    """
    功能：
    1. 从当前活动数据集的 .h5ad 文件中读取二维坐标；
    2. 优先读取 adata.obsm["X_umap"]；
    3. 如果没有 X_umap，则读取 adata.obsm["X_tsne"]；
    4. 将 cell_id、x、y、cell_type、metadata 打包成 JSON 数据，供前端画散点图使用。
    """

    def __init__(self) -> None:
        self._cache_key: tuple[Any, ...] | None = None
        self._cache_payload: dict[str, Any] | None = None

    def get_scatter_data(
        self,
        dataset: dict[str, Any],
        *,
        max_points: int | None = None,
        metadata_fields: list[str] | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """返回前端散点图需要的数据。

        参数：
            dataset: DatasetManager.get_active_dataset() 返回的当前活动数据集
            max_points: 可选，限制返回点数，用于降采样
            metadata_fields: 可选，指定要返回哪些元数据字段
            random_state: 降采样随机种子
        """

        source_path = Path(str(dataset.get("source_path", "")))
        metadata_path = Path(str(dataset.get("metadata_path", "")))
        dataset_id = str(dataset.get("id", "active"))

        if not source_path.exists():
            raise FileNotFoundError(
                f"找不到活动数据集原始 .h5ad 文件: {source_path}。"
                "请先上传包含 X_umap 或 X_tsne 的 .h5ad 数据集。"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(f"找不到元数据文件: {metadata_path}")

        source_mtime = source_path.stat().st_mtime_ns
        metadata_mtime = metadata_path.stat().st_mtime_ns
        fields_key = tuple(metadata_fields or [])

        cache_key = (
            dataset_id,
            str(source_path),
            source_mtime,
            str(metadata_path),
            metadata_mtime,
            max_points,
            fields_key,
            random_state,
        )

        if self._cache_key == cache_key and self._cache_payload is not None:
            return self._cache_payload

        started = time.perf_counter()

        # backed="r" 表示只读模式，避免一次性把巨大的表达矩阵全部读入内存
        adata = sc.read_h5ad(str(source_path), backed="r")

        basis = None
        if "X_umap" in adata.obsm_keys():
            basis = "X_umap"
        elif "X_tsne" in adata.obsm_keys():
            basis = "X_tsne"
        else:
            available = list(map(str, adata.obsm_keys()))
            raise KeyError(
                f"数据集中没有 obsm['X_umap'] 或 obsm['X_tsne']，"
                f"当前可用 obsm: {available}"
            )

        coords = np.asarray(adata.obsm[basis], dtype=np.float32)

        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"obsm['{basis}'] 应为 n×2 或 n×d 矩阵，当前 shape={coords.shape}"
            )

        coords = coords[:, :2]

        cell_ids = np.asarray(list(adata.obs_names), dtype=object).astype(str)

        metadata = pd.read_csv(metadata_path)

        if "cell_id" not in metadata.columns:
            metadata.insert(0, "cell_id", cell_ids[: len(metadata)])

        if len(metadata) != len(cell_ids):
            raise ValueError(
                f"元数据行数({len(metadata)})与 h5ad 细胞数({len(cell_ids)})不一致，"
                "无法安全拼接散点数据"
            )

        n_total = int(coords.shape[0])
        indices = np.arange(n_total)
        sampled = False

        if max_points is not None and 0 < int(max_points) < n_total:
            rng = np.random.default_rng(int(random_state))
            indices = np.sort(
                rng.choice(indices, size=int(max_points), replace=False)
            )
            sampled = True

        default_fields = ["cell_type", "disease", "AgeGroup", "batch", "sample"]

        if metadata_fields:
            wanted_fields = metadata_fields
        else:
            wanted_fields = [c for c in default_fields if c in metadata.columns]

        wanted_fields = [
            c for c in wanted_fields
            if c in metadata.columns and c != "cell_id"
        ]

        points: list[dict[str, Any]] = []

        for i in indices.tolist():
            meta_row = metadata.iloc[i]

            meta = {
                field: _sanitize(meta_row[field])
                for field in wanted_fields
            }

            points.append(
                {
                    "cell_id": str(cell_ids[i]),
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "cell_type": (
                        _sanitize(meta_row["cell_type"])
                        if "cell_type" in metadata.columns
                        else None
                    ),
                    "metadata": meta,
                }
            )

        payload = {
            "dataset": {
                "id": dataset_id,
                "name": dataset.get("name"),
                "n_obs": n_total,
            },
            "basis": basis,
            "sampled": sampled,
            "returned": len(points),
            "total": n_total,
            "metadata_fields": wanted_fields,
            "time_ms": round((time.perf_counter() - started) * 1000, 3),
            "points": points,
        }

        self._cache_key = cache_key
        self._cache_payload = payload

        return payload


def _sanitize(value: Any) -> Any:
    """把 numpy / pandas 类型转换成 Flask 可以 jsonify 的普通 Python 类型。"""

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return str(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value