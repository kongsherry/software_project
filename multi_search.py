"""多数据集联合检索模块

支持同时加载多个单细胞数据集的 ANN 索引，
实现跨数据集细胞搜索与结果合并排序。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from search import AnnSearcher

logger = logging.getLogger("ann_api.multi_search")


class MultiAnnSearcher:
    """多数据集联合检索器

    工作流程：
    1. 加载多个数据集的索引、向量、元数据；
    2. 根据查询（向量或细胞ID）在所有选中的数据集中独立检索；
    3. 合并各数据集结果，按距离全局排序，去重并返回 Top-K。

    不同数据集应在同一 embedding 空间中（如均经过 PCA + L2 归一化），
    这样才能保证距离度量跨数据集可比。
    """

    def __init__(self) -> None:
        self._searchers: dict[str, AnnSearcher] = {}
        self._dataset_info: dict[str, dict[str, Any]] = {}

    # ──────────── 生命周期管理 ────────────

    def load_dataset(self, dataset: dict[str, Any]) -> AnnSearcher:
        """加载单个数据集的检索器。"""
        ds_id = str(dataset["id"])
        if ds_id not in self._searchers:
            logger.info("加载多检索数据集: %s (%s)", dataset.get("name", ds_id), ds_id)
            searcher = AnnSearcher(
                str(dataset["index_path"]),
                str(dataset["vectors_path"]),
                str(dataset["metadata_path"]),
                str(dataset["cell_ids_path"]),
            )
            self._searchers[ds_id] = searcher
            self._dataset_info[ds_id] = {
                "id": ds_id,
                "name": dataset.get("name", ds_id),
                "n_obs": searcher.total_vectors,
                "dim": searcher.dim,
                "metric": searcher.metric,
            }
        return self._searchers[ds_id]

    def load_datasets(self, datasets: list[dict[str, Any]]) -> None:
        """批量加载多个数据集。"""
        for ds in datasets:
            self.load_dataset(ds)

    def unload_dataset(self, dataset_id: str) -> None:
        """卸载指定数据集，释放内存。"""
        self._searchers.pop(dataset_id, None)
        self._dataset_info.pop(dataset_id, None)
        logger.info("已卸载多检索数据集: %s", dataset_id)

    def clear(self) -> None:
        """卸载所有已加载的数据集。"""
        count = len(self._searchers)
        self._searchers.clear()
        self._dataset_info.clear()
        logger.info("已清除全部 %d 个多检索数据集", count)

    def get_loaded_datasets(self) -> list[dict[str, Any]]:
        """返回已加载数据集的摘要信息。"""
        return list(self._dataset_info.values())

    def is_loaded(self, dataset_id: str) -> bool:
        """检查某数据集是否已加载。"""
        return dataset_id in self._searchers

    def get_searcher(self, dataset_id: str) -> AnnSearcher | None:
        """获取指定数据集对应的检索器。"""
        return self._searchers.get(dataset_id)

    # ──────────── 跨数据集查找 ────────────

    def find_cell_id(
        self, cell_id: str, dataset_ids: list[str] | None = None
    ) -> tuple[Optional[str], Optional[np.ndarray]]:
        """在指定（或全部已加载）数据集中查找细胞 ID。"""
        search_ids = dataset_ids or list(self._searchers.keys())
        for ds_id in search_ids:
            searcher = self._searchers.get(ds_id)
            if searcher is None:
                continue
            try:
                vec = searcher.get_vector_by_cell_id(cell_id)
                return ds_id, vec
            except KeyError:
                continue
        return None, None

    # ──────────── 联合检索 ────────────

    def search(
        self,
        vector: np.ndarray,
        dataset_ids: list[str],
        k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        search_params: Optional[dict[str, Any]] = None,
        query_cell_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """在多个数据集中联合检索。

        每个数据集独立检索 k 个最近邻，合并后按距离全局排序取 Top-K。
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        all_results: list[dict[str, Any]] = []
        search_times: dict[str, Any] = {}
        dataset_dims: dict[str, int] = {}
        errors: dict[str, str] = {}

        for ds_id in dataset_ids:
            searcher = self._searchers.get(ds_id)
            if searcher is None:
                errors[ds_id] = "数据集未加载"
                continue

            # 维度检查
            if vector.shape[1] != searcher.dim:
                errors[ds_id] = (
                    f"向量维度 {vector.shape[1]} 与数据集维度 {searcher.dim} 不匹配，已跳过"
                )
                continue

            ds_name = self._dataset_info.get(ds_id, {}).get("name", ds_id)
            ds_info = self._dataset_info.get(ds_id, {})
            dataset_dims[ds_id] = searcher.dim

            # 每个数据集独立搜索，取 k 个候选项
            try:
                result = searcher.search_by_vector(
                    vector,
                    k=k,
                    filters=self._adapt_filters_for_dataset(filters, searcher),
                    search_params=search_params,
                )
            except Exception as exc:
                logger.warning("数据集 %s 检索失败: %s", ds_id, exc)
                errors[ds_id] = str(exc)
                continue

            search_times[ds_id] = {
                "time_ms": result.get("time_ms", 0),
                "found": len(result.get("results", [])),
                "strategy": (
                    result.get("filter_info", {}).get("strategy")
                    if result.get("filter_info")
                    else "ann"
                ),
            }

            # 为每个结果标注来源数据集
            for r in result.get("results", []):
                r["dataset_id"] = ds_id
                r["dataset_name"] = ds_name
                all_results.append(r)

        # 按距离全局升序排列（距离越小越相似）
        all_results.sort(
            key=lambda x: (
                x["distance"] if x["distance"] is not None else float("inf")
            )
        )

        # 取前 k 个，重新编号 rank，并去除重复 cell_id（同一细胞跨数据集）
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for r in all_results:
            key = (r.get("dataset_id", ""), r.get("cell_id", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                if len(deduped) >= k:
                    break

        for i, r in enumerate(deduped, 1):
            r["rank"] = i

        # 构建数据集的维度信息汇总
        dims_info: dict[str, int | str] = {}
        if dataset_dims:
            unique_dims = set(dataset_dims.values())
            if len(unique_dims) == 1:
                dims_info = {"all": unique_dims.pop()}
            else:
                dims_info = {ds_id: d for ds_id, d in dataset_dims.items()}

        return {
            "query": {
                "cell_id": query_cell_id,
                "k": k,
                "total_datasets_requested": len(dataset_ids),
                "datasets_queried": list(search_times.keys()),
                "datasets_with_errors": errors if errors else None,
            },
            "datasets": {
                "queried": search_times,
                "dimensions": dims_info,
                "loaded": list(self._dataset_info.keys()),
            },
            "time_ms": round(
                max(
                    (t.get("time_ms", 0) for t in search_times.values()),
                    default=0.0,
                ),
                3,
            ),
            "results": deduped,
            "total_raw": len(all_results),
        }

    # ──────────── 过滤适配 ────────────

    @staticmethod
    def _adapt_filters_for_dataset(
        filters: dict[str, Any] | None,
        searcher: AnnSearcher,
    ) -> dict[str, Any] | None:
        """移除当前数据集中不存在的过滤字段，避免因字段缺失导致检索失败。"""
        if not filters:
            return None

        available_columns = set(searcher.metadata.columns)
        adapted: dict[str, Any] = {}
        skipped: list[str] = []

        for key, value in filters.items():
            if key in available_columns:
                adapted[key] = value
            else:
                skipped.append(key)

        if skipped:
            logger.debug(
                "以下过滤字段在当前数据集中不存在，已跳过: %s", skipped
            )

        return adapted or None

    # ──────────── 索引合并 ────────────

    def build_merged_index(
        self,
        dataset_ids: list[str],
        output_path: str,
        index_type: str = "hnsw",
        M: int = 32,
        ef_construction: int = 200,
    ) -> dict[str, Any]:
        """将多个数据集的向量合并，构建统一的联合索引。"""
        from index_builder import build_index_from_vectors, ensure_faiss_vectors
        import numpy as np

        # 收集所有向量并检查维度一致性
        all_vectors: list[np.ndarray] = []
        all_cell_ids: list[str] = []
        all_dataset_tags: list[str] = []
        dim = None

        for ds_id in dataset_ids:
            searcher = self._searchers.get(ds_id)
            if searcher is None:
                raise ValueError(f"数据集未加载: {ds_id}")

            if dim is None:
                dim = searcher.dim
            elif searcher.dim != dim:
                raise ValueError(
                    f"数据集 {ds_id} 的维度 ({searcher.dim}) "
                    f"与已合并数据集的维度 ({dim}) 不一致"
                )

            ds_name = self._dataset_info.get(ds_id, {}).get("name", ds_id)
            vectors = ensure_faiss_vectors(searcher.vectors)
            all_vectors.append(vectors)

            # 为细胞 ID 添加数据集前缀以避免冲突
            for cid in searcher.cell_ids:
                tagged = f"{ds_id}::{cid}"
                all_cell_ids.append(tagged)
                all_dataset_tags.append(ds_id)

        if not all_vectors:
            raise ValueError("没有可合并的数据集")

        merged_vectors = np.vstack(all_vectors)
        merged_cell_ids = np.array(all_cell_ids, dtype=object)
        merged_dataset_tags = np.array(all_dataset_tags, dtype=object)

        # 保存合并后的向量和细胞 ID
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        vect_path = output_dir / "merged_vectors.npy"
        cell_ids_path = output_dir / "merged_cell_ids.npy"

        np.save(str(vect_path), merged_vectors)
        np.save(str(cell_ids_path), merged_cell_ids)

        # 构建索引
        summary = build_index_from_vectors(
            vectors_path=str(vect_path),
            output_path=output_path,
            index_type=index_type,
            metric="l2",
            M=M,
            efConstruction=ef_construction,
        )

        return {
            **summary,
            "merged_datasets": dataset_ids,
            "total_vectors": int(merged_vectors.shape[0]),
            "dim": int(merged_vectors.shape[1]),
            "dataset_tag_counts": {
                ds_id: int(np.sum(merged_dataset_tags == ds_id))
                for ds_id in dataset_ids
            },
        }


def _sanitize(value: Any) -> Any:
    """将 numpy/Pandas 类型转为 JSON 兼容的 Python 原生类型。"""
    import pandas as pd

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if pd.isna(value):
        return None
    return value
