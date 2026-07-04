import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import faiss

from index_builder import AnnIndexBuilder


class AnnSearcher:
    """单细胞高维向量 ANN 检索器 (P3模块)

    封装了通过细胞ID或原始向量进行近似最近邻检索的逻辑，
    并将检索结果关联细胞元数据（如细胞类型、疾病、年龄组等）。
    """

    def __init__(
        self,
        index_path: str,
        vectors_path: str,
        metadata_path: str,
        cell_ids_path: str,
    ) -> None:
        self._index_path = Path(index_path)
        self._vectors_path = Path(vectors_path)
        self._metadata_path = Path(metadata_path)
        self._cell_ids_path = Path(cell_ids_path)

        self.vectors: np.ndarray = np.load(str(vectors_path))
        self.cell_ids: np.ndarray = np.load(str(cell_ids_path), allow_pickle=True)
        self.metadata: pd.DataFrame = pd.read_csv(str(metadata_path))
        self._id_to_idx: dict[str, int] = {str(cid): i for i, cid in enumerate(self.cell_ids)}

        builder = AnnIndexBuilder()
        builder.load_index(str(index_path))
        self.index: faiss.Index = builder.get_faiss_index()
        self._metric_type: int = getattr(self.index, "metric_type", builder.metric)

    @property
    def total_vectors(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def metric(self) -> str:
        return "ip" if self._metric_type == faiss.METRIC_INNER_PRODUCT else "l2"

    def search_by_cell_id(
        self, cell_id: str, k: int = 10, filters: Optional[dict] = None
    ) -> dict[str, Any]:
        """根据细胞ID查找对应向量，再执行ANN检索。

        Args:
            cell_id: 细胞标识符
            k: 返回的最近邻数量
            filters: 元数据过滤条件，如 {"cell_type": "hepatocyte", "disease": "healthy"}

        Returns:
            包含查询信息、耗时和检索结果的字典
        """
        idx = self._id_to_idx.get(cell_id)
        if idx is None:
            raise KeyError(f"未找到细胞ID: {cell_id}（可用的ID示例: {self.cell_ids[:3].tolist()}）")

        query_vector = self.vectors[idx : idx + 1]
        if filters:
            return self._search_with_filters(query_vector, k, filters, query_cell_id=cell_id)
        return self._execute_search(query_vector, k, query_cell_id=cell_id)

    def search_by_vector(
        self, vector: np.ndarray, k: int = 10, filters: Optional[dict] = None
    ) -> dict[str, Any]:
        """直接根据输入向量执行ANN检索。

        Args:
            vector: 查询向量，shape 应为 (d,) 或 (1, d)
            k: 返回的最近邻数量
            filters: 元数据过滤条件，如 {"cell_type": "hepatocyte", "disease": "healthy"}

        Returns:
            包含查询信息、耗时和检索结果的字典
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[1] != self.dim:
            raise ValueError(f"查询向量维度 {vector.shape[1]} 与索引维度 {self.dim} 不匹配")

        if filters:
            return self._search_with_filters(vector, k, filters)
        return self._execute_search(vector, k)

    def _execute_search(
        self, query: np.ndarray, k: int, query_cell_id: Optional[str] = None
    ) -> dict[str, Any]:
        start = time.perf_counter()
        distances, indices = self.index.search(query, k)
        elapsed = time.perf_counter() - start

        dists = distances[0].tolist()
        idxs = indices[0].tolist()

        results = []
        for rank, (dist, idx_) in enumerate(zip(dists, idxs), start=1):
            if idx_ < 0:
                results.append({"rank": rank, "cell_id": None, "distance": None, "metadata": None})
                continue

            cid = str(self.cell_ids[idx_])
            meta = self.metadata.iloc[idx_].to_dict()
            results.append({
                "rank": rank,
                "cell_id": cid,
                "distance": float(dist),
                "metadata": {k: _sanitize(v) for k, v in meta.items()},
            })

        return {
            "query": {
                "cell_id": query_cell_id,
                "k": k,
                "metric": self.metric,
            },
            "time_ms": round(elapsed * 1000, 3),
            "results": results,
        }


    def _search_with_filters(
        self,
        query: np.ndarray,
        k: int,
        filters: dict,
        query_cell_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """混合策略检索：根据过滤后细胞数量自动选择预过滤或后过滤。

        - 预过滤：过滤后细胞数 < 1000，构建 FAISS IDMap 子集索引进行精确搜索。
        - 后过滤：过滤后细胞数 >= 1000，先取更多候选（max(k*3, 200)），再按条件筛选补全至 K 个。

        filters 支持三种值格式：
        - 字符串: 精确匹配 (向后兼容)
        - 列表: 多选 OR 匹配，如 {"cell_type": ["T-cell", "B-cell"]}
        - 字典: 数值范围，如 {"donor_age": {"op": ">", "value": 50}}
        """
        start = time.perf_counter()

        # 构建布尔掩码 — 所有过滤条件均为 AND 关系
        mask = pd.Series([True] * len(self.metadata))
        for key, value in filters.items():
            if key not in self.metadata.columns:
                raise ValueError(
                    f"元数据中不存在过滤列: '{key}'，"
                    f"可用列: {list(self.metadata.columns)}"
                )
            # 数值范围过滤: value 为 {"op": ">", "value": 5} 格式
            if isinstance(value, dict) and "op" in value and "value" in value:
                op = value["op"]
                val = value["value"]
                col = pd.to_numeric(self.metadata[key], errors="coerce")
                if op == ">":
                    mask &= (col > val)
                elif op == "<":
                    mask &= (col < val)
                elif op == ">=":
                    mask &= (col >= val)
                elif op == "<=":
                    mask &= (col <= val)
                elif op == "==":
                    mask &= (col == val)
                else:
                    raise ValueError(f"不支持的操作符: {op}")
            elif isinstance(value, list):
                # 多选过滤 (OR 逻辑): 匹配列表中任意一个值
                str_values = [str(v) for v in value]
                mask &= (self.metadata[key].astype(str).isin(str_values))
            else:
                # 兼容旧格式：字符串精确匹配
                mask &= (self.metadata[key].astype(str) == str(value))

        filtered_indices = np.where(mask)[0]
        filtered_count = int(len(filtered_indices))

        if filtered_count == 0:
            elapsed = time.perf_counter() - start
            return {
                "query": {
                    "cell_id": query_cell_id,
                    "k": k,
                    "metric": self.metric,
                    "filters": filters,
                },
                "time_ms": round(elapsed * 1000, 3),
                "results": [],
                "filter_info": {
                    "filtered_count": 0,
                    "strategy": "none",
                    "filters": filters,
                },
            }

        FILTER_THRESHOLD = 1000

        if filtered_count < FILTER_THRESHOLD:
            strategy = "pre_filter"
            actual_k = min(k, filtered_count)
            # 预过滤：提取子集向量，构建 IDMap 精确索引
            subset_vectors = self.vectors[filtered_indices]
            subset_ids = np.arange(filtered_count, dtype=np.int64)

            if self._metric_type == faiss.METRIC_INNER_PRODUCT:
                base_index = faiss.IndexFlatIP(self.dim)
            else:
                base_index = faiss.IndexFlatL2(self.dim)

            id_map_index = faiss.IndexIDMap(base_index)
            id_map_index.add_with_ids(subset_vectors, subset_ids)

            sub_dists, sub_indices = id_map_index.search(query, actual_k)

            elapsed = time.perf_counter() - start

            dists = sub_dists[0].tolist()
            idxs = sub_indices[0].tolist()

            results = []
            for rank, (dist, sub_idx) in enumerate(zip(dists, idxs), start=1):
                if sub_idx < 0:
                    results.append({
                        "rank": rank,
                        "cell_id": None,
                        "distance": None,
                        "metadata": None,
                    })
                    continue

                orig_idx = int(filtered_indices[sub_idx])
                cid = str(self.cell_ids[orig_idx])
                meta = self.metadata.iloc[orig_idx].to_dict()
                results.append({
                    "rank": rank,
                    "cell_id": cid,
                    "distance": float(dist),
                    "metadata": {k: _sanitize(v) for k, v in meta.items()},
                })
        else:
            strategy = "post_filter"
            # 后过滤：先取更多候选
            extra_k = min(max(k * 3, 200), self.total_vectors)
            sub_dists, sub_indices = self.index.search(query, extra_k)

            dists = sub_dists[0].tolist()
            idxs = sub_indices[0].tolist()

            filtered_set = set(int(i) for i in filtered_indices)
            results = []
            rank = 1
            for dist, idx_ in zip(dists, idxs):
                if idx_ < 0:
                    continue
                if idx_ in filtered_set:
                    cid = str(self.cell_ids[idx_])
                    meta = self.metadata.iloc[idx_].to_dict()
                    results.append({
                        "rank": rank,
                        "cell_id": cid,
                        "distance": float(dist),
                        "metadata": {k: _sanitize(v) for k, v in meta.items()},
                    })
                    rank += 1
                    if len(results) >= k:
                        break

            elapsed = time.perf_counter() - start

        return {
            "query": {
                "cell_id": query_cell_id,
                "k": k,
                "metric": self.metric,
                "filters": filters,
            },
            "time_ms": round(elapsed * 1000, 3),
            "results": results,
            "filter_info": {
                "filtered_count": int(filtered_count),
                "strategy": strategy,
                "filters": filters,
            },
        }


def _sanitize(value: Any) -> Any:
    """将 numpy/Pandas 类型转为 JSON 兼容的 Python 原生类型。"""
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


def main() -> None:
    """命令行测试入口，验证 AnnSearcher 的两个检索接口。"""
    import argparse

    p = argparse.ArgumentParser(description="P3: ANN 检索测试")
    p.add_argument("--index", default="indices/hnsw_M32_ef200.index", help="索引文件路径")
    p.add_argument("--vectors", default="results/vectors.npy", help="向量矩阵文件路径")
    p.add_argument("--metadata", default="results/obs_metadata.csv", help="元数据 CSV 路径")
    p.add_argument("--cell-ids", default="results/cell_ids.npy", help="细胞 ID 文件路径")
    p.add_argument("--cell-id", default=None, help="按细胞 ID 检索（不指定则用向量检索）")
    p.add_argument("--k", type=int, default=10, help="返回 Top-K 结果")
    args = p.parse_args()

    searcher = AnnSearcher(args.index, args.vectors, args.metadata, args.cell_ids)
    print(f"索引向量总数: {searcher.total_vectors}, 维度: {searcher.dim}, 度量: {searcher.metric}")

    if args.cell_id:
        result = searcher.search_by_cell_id(args.cell_id, k=args.k)
    else:
        first = searcher.vectors[0:1]
        result = searcher.search_by_vector(first, k=args.k)
        print(f"（未指定 --cell-id，使用索引 0 的向量进行查询: {searcher.cell_ids[0]}）")

    print(f"\n检索耗时: {result['time_ms']} ms")
    print(f"度量方式: {result['query']['metric']}")
    print(f"Top-{result['query']['k']} 结果:")
    for r in result["results"]:
        meta = r.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        print(f"  [{r['rank']}] cell_id={r['cell_id']}, distance={r['distance']:.4f} | {meta_str}")


if __name__ == "__main__":
    main()
