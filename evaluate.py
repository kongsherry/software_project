from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import faiss
except ModuleNotFoundError as exc:
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

if TYPE_CHECKING:
    from dataset_manager import DatasetManager

DEFAULT_REPORT_PATH = Path(os.getenv("ANN_EVALUATION_REPORT_PATH", "evaluation_report.json"))
DEFAULT_SAMPLE_SIZE = int(os.getenv("ANN_EVAL_SAMPLE_SIZE", "500"))
DEFAULT_RANDOM_SEED = int(os.getenv("ANN_EVAL_RANDOM_SEED", "42"))
RECALL_K_VALUES = (1, 10, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ANN recall and latency metrics.")
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Dataset id from results/manifest.json. Defaults to the active dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of query vectors to sample. Default: {DEFAULT_SAMPLE_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for query sampling. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help=f"Output JSON report path. Default: {DEFAULT_REPORT_PATH}",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")

    _ensure_faiss_available()
    from dataset_manager import DatasetManager

    dataset_manager = DatasetManager()
    dataset = _resolve_dataset(dataset_manager, args.dataset_id)
    report = evaluate_dataset(
        dataset=dataset,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Evaluation report saved to: {report_path}")
    print(f"Dataset: {report['dataset']['id']} ({report['dataset']['name']})")
    print(f"Queries: {report['sample']['actual_query_count']}")
    print(
        "Recall: "
        f"@1={report['recall']['recall@1']:.4f}, "
        f"@10={report['recall']['recall@10']:.4f}, "
        f"@50={report['recall']['recall@50']:.4f}"
    )
    print(
        "ANN performance: "
        f"QPS={report['ann_search']['qps']:.2f}, "
        f"avg={report['ann_search']['average_response_time_ms']:.3f} ms"
    )


def evaluate_dataset(dataset: dict[str, Any], sample_size: int, seed: int) -> dict[str, Any]:
    vectors_path = Path(str(dataset["vectors_path"]))
    index_path = Path(str(dataset["index_path"]))
    cell_ids_path = Path(str(dataset["cell_ids_path"]))

    if not vectors_path.exists():
        raise FileNotFoundError(f"Vector file not found: {vectors_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    vectors = np.load(str(vectors_path)).astype(np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={vectors.shape}")

    total_vectors, dim = int(vectors.shape[0]), int(vectors.shape[1])
    actual_query_count = min(sample_size, total_vectors)
    if actual_query_count == 0:
        raise ValueError("Dataset contains no vectors")

    cell_ids = _load_cell_ids(cell_ids_path, total_vectors)
    query_indices = np.random.default_rng(seed).choice(
        total_vectors,
        size=actual_query_count,
        replace=False,
    )
    query_vectors = vectors[query_indices]

    ann_index = faiss.read_index(str(index_path))
    metric_type = _resolve_metric_type(
        loaded_metric=getattr(ann_index, "metric_type", None),
        dataset_metric=dataset.get("metric"),
    )
    max_k = min(max(RECALL_K_VALUES), total_vectors)

    gt_index, gt_build_ms = _build_ground_truth_index(vectors, metric_type)
    gt_started = time.perf_counter()
    gt_distances, gt_indices = gt_index.search(query_vectors, max_k)
    gt_total_ms = (time.perf_counter() - gt_started) * 1000

    ann_indices, ann_latency_ms, ann_total_ms = _search_ann_index(ann_index, query_vectors, max_k)

    recall = {
        f"recall@{k}": round(_compute_recall(gt_indices, ann_indices, min(k, max_k)), 6)
        for k in RECALL_K_VALUES
    }
    effective_k = {str(k): min(k, max_k) for k in RECALL_K_VALUES}
    performance_summary = {
        "recall_at_1": recall["recall@1"],
        "recall_at_10": recall["recall@10"],
        "recall_at_50": recall["recall@50"],
        "ann_qps": round(actual_query_count / (ann_total_ms / 1000), 3) if ann_total_ms else None,
        "ann_avg_latency_ms": round(float(np.mean(ann_latency_ms)), 3),
        "ann_p95_latency_ms": round(float(np.percentile(ann_latency_ms, 95)), 3),
        "ground_truth_avg_latency_ms": round(gt_total_ms / actual_query_count, 3),
        "index_size_mb": round(index_path.stat().st_size / (1024 * 1024), 3),
        "query_count": actual_query_count,
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": str(dataset["id"]),
            "name": str(dataset.get("name", dataset["id"])),
            "metric": _metric_name(metric_type),
            "index_type": dataset.get("index_type", "unknown"),
            "vectors_path": str(vectors_path),
            "index_path": str(index_path),
            "cell_ids_path": str(cell_ids_path),
            "total_vectors": total_vectors,
            "dimension": dim,
            "index_build": dataset.get("index_build"),
        },
        "performance_summary": performance_summary,
        "recall": recall,
        "ann_search": {
            "engine": type(ann_index).__name__,
            "top_k_searched": max_k,
            "total_search_time_ms": round(ann_total_ms, 3),
            "average_response_time_ms": round(float(np.mean(ann_latency_ms)), 3),
            "p95_response_time_ms": round(float(np.percentile(ann_latency_ms, 95)), 3),
            "qps": performance_summary["ann_qps"],
        },
        "ground_truth": {
            "engine": "faiss.IndexFlatIP"
            if metric_type == faiss.METRIC_INNER_PRODUCT
            else "faiss.IndexFlatL2",
            "top_k_searched": max_k,
            "build_time_ms": round(gt_build_ms, 3),
            "total_search_time_ms": round(gt_total_ms, 3),
            "average_response_time_ms": round(gt_total_ms / actual_query_count, 3),
        },
        "index_size": {
            "bytes": int(index_path.stat().st_size),
            "megabytes": round(index_path.stat().st_size / (1024 * 1024), 3),
        },
        "sample": {
            "requested_query_count": int(sample_size),
            "actual_query_count": actual_query_count,
            "random_seed": int(seed),
            "effective_k": effective_k,
            "query_indices": [int(i) for i in query_indices.tolist()],
        },
        "ground_truth_top10": _build_ground_truth_top10(
            query_indices=query_indices,
            gt_indices=gt_indices,
            gt_distances=gt_distances,
            cell_ids=cell_ids,
        ),
    }

    return report


def _resolve_dataset(dataset_manager: "DatasetManager", dataset_id: str | None) -> dict[str, Any]:
    if dataset_id is None:
        return dataset_manager.get_active_dataset()

    datasets = dataset_manager.list_datasets()["datasets"]
    for dataset in datasets:
        if str(dataset["id"]) == dataset_id:
            return dataset

    raise KeyError(f"Dataset not found: {dataset_id}")


def _load_cell_ids(cell_ids_path: Path, expected_length: int) -> np.ndarray | None:
    if not cell_ids_path.exists():
        return None

    cell_ids = np.load(str(cell_ids_path), allow_pickle=True)
    if len(cell_ids) != expected_length:
        raise ValueError(
            f"cell_ids length mismatch: expected {expected_length}, got {len(cell_ids)}"
        )
    return cell_ids


def _resolve_metric_type(loaded_metric: Any, dataset_metric: Any) -> int:
    if loaded_metric in {faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT}:
        return int(loaded_metric)
    if str(dataset_metric).lower() == "ip":
        return faiss.METRIC_INNER_PRODUCT
    return faiss.METRIC_L2


def _metric_name(metric_type: int) -> str:
    return "ip" if metric_type == faiss.METRIC_INNER_PRODUCT else "l2"


def _build_ground_truth_index(vectors: np.ndarray, metric_type: int) -> tuple[faiss.Index, float]:
    started = time.perf_counter()
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return index, elapsed_ms


def _search_ann_index(
    ann_index: faiss.Index,
    query_vectors: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, list[float], float]:
    indices_list: list[np.ndarray] = []
    latency_ms: list[float] = []

    total_started = time.perf_counter()
    for query in query_vectors:
        single_query = query.reshape(1, -1)
        started = time.perf_counter()
        _, indices = ann_index.search(single_query, top_k)
        latency_ms.append((time.perf_counter() - started) * 1000)
        indices_list.append(indices[0].copy())
    total_elapsed_ms = (time.perf_counter() - total_started) * 1000

    return np.vstack(indices_list), latency_ms, total_elapsed_ms


def _compute_recall(gt_indices: np.ndarray, ann_indices: np.ndarray, k: int) -> float:
    hits = 0.0
    for gt_row, ann_row in zip(gt_indices[:, :k], ann_indices[:, :k], strict=False):
        gt_set = {int(idx) for idx in gt_row.tolist() if int(idx) >= 0}
        ann_set = {int(idx) for idx in ann_row.tolist() if int(idx) >= 0}
        if not gt_set:
            continue
        hits += len(gt_set & ann_set) / len(gt_set)
    return hits / len(gt_indices)


def _build_ground_truth_top10(
    query_indices: np.ndarray,
    gt_indices: np.ndarray,
    gt_distances: np.ndarray,
    cell_ids: np.ndarray | None,
) -> list[dict[str, Any]]:
    top_k = min(10, gt_indices.shape[1])
    records: list[dict[str, Any]] = []

    for row_index, query_index in enumerate(query_indices.tolist()):
        neighbor_indices = [int(i) for i in gt_indices[row_index, :top_k].tolist()]
        records.append(
            {
                "query_index": int(query_index),
                "query_id": _cell_id_at(cell_ids, int(query_index)),
                "neighbor_indices": neighbor_indices,
                "neighbor_ids": [_cell_id_at(cell_ids, idx) for idx in neighbor_indices],
                "distances": [float(d) for d in gt_distances[row_index, :top_k].tolist()],
            }
        )

    return records


def _cell_id_at(cell_ids: np.ndarray | None, index: int) -> str | None:
    if cell_ids is None or index < 0 or index >= len(cell_ids):
        return None
    return str(cell_ids[index])


def _ensure_faiss_available() -> None:
    if faiss is None:
        raise ModuleNotFoundError(
            "faiss is required to run evaluate.py. Install faiss-cpu or use the "
            "same Python environment as the ANN service."
        ) from _FAISS_IMPORT_ERROR


if __name__ == "__main__":
    main()
