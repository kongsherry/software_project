from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

import numpy as np


DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").lower()


class DeepSeekClient:
    """Minimal OpenAI-compatible chat client for DeepSeek."""

    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL).rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", str(DEEPSEEK_MAX_TOKENS)))
        self.thinking = os.getenv("DEEPSEEK_THINKING", DEEPSEEK_THINKING).lower()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        timeout: int = 30,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法调用 DeepSeek 大模型")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "stream": False,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": self.thinking}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API 请求失败: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API 连接失败: {exc.reason}") from exc

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("DeepSeek API 返回了空 content，请重试或调整 max_tokens/prompt")
        return _extract_json_object(content)


def build_dataset_schema(searcher: Any) -> dict[str, Any]:
    metadata = searcher.metadata
    columns = [str(c) for c in metadata.columns]
    values: dict[str, list[str]] = {}
    for col in columns:
        if col == "cell_id":
            continue
        unique = metadata[col].dropna().astype(str).unique().tolist()
        if len(unique) <= 50:
            values[col] = unique[:50]
    return {
        "metadata_columns": columns,
        "metadata_values": values,
        "total_cells": int(searcher.total_vectors),
        "metric": searcher.metric,
    }


def parse_natural_query(question: str, schema: dict[str, Any]) -> dict[str, Any]:
    client = DeepSeekClient()
    system_prompt = (
        "你是单细胞 ANN 检索系统的查询规划器。"
        "只输出 json 对象，不要输出 Markdown。"
        "把中文自然语言转成结构化查询计划。"
        "可用 action: search_by_cell_id, centroid_search, metadata_filter。"
        "search_by_cell_id 需要 cell_id；centroid_search 用 query_filters 计算参考细胞群均值向量，"
        "再用 result_filters 限定返回细胞；metadata_filter 只按元数据筛选细胞。"
        "filters 的 key 必须来自 metadata_columns，value 必须使用 metadata_values 中真实存在的原值；"
        "如果用户提到的值不在 metadata_values 中，不要替换成不存在的值。"
        "k 默认 10，最大 100。"
    )
    user_prompt = json.dumps(
        {
            "question": question,
            "dataset_schema": schema,
            "required_json_shape": {
                "action": "search_by_cell_id | centroid_search | metadata_filter",
                "cell_id": "optional string",
                "k": "integer",
                "query_filters": "object for centroid_search",
                "result_filters": "object for search result filtering",
                "filters": "object for metadata_filter",
                "explain": "short Chinese explanation",
            },
            "examples": [
                {
                    "question": "找出和 cell_0044 最像的 20 个细胞",
                    "json": {
                        "action": "search_by_cell_id",
                        "cell_id": "cell_0044",
                        "k": 20,
                        "result_filters": {},
                    },
                },
                {
                    "question": "找 HCC 样本中最像 Kupffer cell 的 20 个细胞",
                    "json": {
                        "action": "centroid_search",
                        "k": 20,
                        "query_filters": {"cell_type": "Kupffer cell"},
                        "result_filters": {"disease": "HCC"},
                    },
                },
                {
                    "question": "查询 Healthy 成人样本里的 hepatocyte",
                    "json": {
                        "action": "metadata_filter",
                        "k": 10,
                        "filters": {"disease": "Healthy", "AgeGroup": "Adult", "cell_type": "hepatocyte"},
                    },
                },
            ],
        },
        ensure_ascii=False,
    )
    plan = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
    normalized = normalize_query_plan(plan)
    validate_query_plan(normalized, schema)
    return normalized


def normalize_query_plan(plan: dict[str, Any]) -> dict[str, Any]:
    action = str(plan.get("action", "")).strip()
    if action not in {"search_by_cell_id", "centroid_search", "metadata_filter"}:
        raise ValueError(f"自然语言查询 action 不受支持: {action}")

    try:
        k = int(plan.get("k", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("自然语言查询返回的 k 必须是整数") from exc
    k = max(1, min(k, 100))

    normalized: dict[str, Any] = {
        "action": action,
        "k": k,
        "explain": str(plan.get("explain", "")).strip(),
    }
    for key in ("cell_id", "query_filters", "result_filters", "filters"):
        value = plan.get(key)
        if key.endswith("filters") or key == "filters":
            normalized[key] = _clean_filters(value)
        elif value:
            normalized[key] = str(value).strip()
    return normalized


def validate_query_plan(plan: dict[str, Any], schema: dict[str, Any]) -> None:
    columns = set(schema.get("metadata_columns") or [])
    known_values = schema.get("metadata_values") or {}
    for scope in ("query_filters", "result_filters", "filters"):
        filters = plan.get(scope) or {}
        for key, value in filters.items():
            if key not in columns:
                raise ValueError(f"自然语言查询使用了不存在的元数据列: {key}")
            values = known_values.get(key)
            if values is not None and str(value) not in {str(item) for item in values}:
                preview = ", ".join(map(str, values[:12]))
                raise ValueError(
                    f"当前数据集中 {key} 不包含值 '{value}'。"
                    f"可用示例值: {preview}"
                )


def execute_query_plan(searcher: Any, plan: dict[str, Any]) -> dict[str, Any]:
    action = plan["action"]
    k = int(plan.get("k", 10))
    if action == "search_by_cell_id":
        cell_id = plan.get("cell_id")
        if not cell_id:
            raise ValueError("search_by_cell_id 查询缺少 cell_id")
        result = searcher.search_by_cell_id(
            str(cell_id),
            k=k,
            filters=plan.get("result_filters") or None,
        )
    elif action == "centroid_search":
        result = search_by_centroid(
            searcher,
            query_filters=plan.get("query_filters") or {},
            result_filters=plan.get("result_filters") or None,
            k=k,
        )
    elif action == "metadata_filter":
        result = list_by_metadata(searcher, filters=plan.get("filters") or {}, k=k)
    else:
        raise ValueError(f"未知查询计划 action: {action}")

    return {
        **result,
        "natural_language_plan": plan,
    }


def search_by_centroid(
    searcher: Any,
    *,
    query_filters: dict[str, str],
    result_filters: dict[str, str] | None,
    k: int,
) -> dict[str, Any]:
    if not query_filters:
        raise ValueError("centroid_search 需要 query_filters 来定义参考细胞群")

    indices = _metadata_indices(searcher, query_filters)
    if len(indices) == 0:
        return {
            "query": {"k": k, "metric": searcher.metric, "query_filters": query_filters, "filters": result_filters},
            "time_ms": 0.0,
            "results": [],
            "centroid_info": {"matched_reference_cells": 0},
        }

    centroid = np.asarray(searcher.vectors[indices], dtype=np.float32).mean(axis=0)
    result = searcher.search_by_vector(centroid, k=k, filters=result_filters)
    result["query"]["query_filters"] = query_filters
    result["centroid_info"] = {"matched_reference_cells": int(len(indices))}
    return result


def list_by_metadata(searcher: Any, *, filters: dict[str, str], k: int) -> dict[str, Any]:
    indices = _metadata_indices(searcher, filters)
    results = []
    for rank, idx in enumerate(indices[:k].tolist(), start=1):
        meta = searcher.metadata.iloc[int(idx)].to_dict()
        results.append(
            {
                "rank": rank,
                "cell_id": str(searcher.cell_ids[int(idx)]),
                "distance": None,
                "metadata": {key: _sanitize(value) for key, value in meta.items()},
            }
        )
    return {
        "query": {"k": k, "metric": searcher.metric, "filters": filters, "mode": "metadata_filter"},
        "time_ms": 0.0,
        "results": results,
        "filter_info": {
            "filtered_count": int(len(indices)),
            "strategy": "metadata_filter",
            "filters": filters,
        },
    }


def analyze_search_result(result: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    stats = summarize_result(result)
    client = DeepSeekClient()
    if not client.configured:
        return {
            "provider": "local",
            "model": None,
            "summary": _local_summary(stats),
            "stats": stats,
            "suggestions": _local_suggestions(stats),
        }

    system_prompt = (
        "你是单细胞数据分析助手。只输出 json 对象，不要输出 Markdown。"
        "基于 ANN 检索结果的聚合统计解释结果，不要编造不存在的基因、通路或临床结论。"
        "输出中文，包含 summary、key_findings、quality_notes、suggestions。"
    )
    user_prompt = json.dumps(
        {
            "dataset": {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "n_obs": dataset.get("n_obs"),
            },
            "query": result.get("query"),
            "stats": stats,
        },
        ensure_ascii=False,
    )
    analysis = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
    return {
        "provider": "deepseek",
        "model": client.model,
        "summary": str(analysis.get("summary", "")),
        "key_findings": analysis.get("key_findings", []),
        "quality_notes": analysis.get("quality_notes", []),
        "suggestions": analysis.get("suggestions", []),
        "stats": stats,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in result.get("results", []) if row.get("metadata")]
    distances = [
        float(row["distance"])
        for row in rows
        if row.get("distance") is not None
    ]

    # 动态从结果中收集所有元数据字段
    fields_set: set[str] = set()
    for row in rows:
        meta = row.get("metadata", {})
        if isinstance(meta, dict):
            fields_set.update(meta.keys())
    fields = sorted(fields_set)
    distributions = {}
    for field in fields:
        counter = Counter(
            str(row["metadata"].get(field))
            for row in rows
            if row.get("metadata", {}).get(field) not in (None, "")
        )
        if counter:
            total = sum(counter.values())
            distributions[field] = [
                {"value": value, "count": count, "ratio": round(count / total, 3)}
                for value, count in counter.most_common(8)
            ]

    return {
        "result_count": len(rows),
        "query": result.get("query", {}),
        "distance": {
            "min": round(min(distances), 6) if distances else None,
            "max": round(max(distances), 6) if distances else None,
            "mean": round(sum(distances) / len(distances), 6) if distances else None,
            "std": round(float(np.std(distances)), 6) if distances else None,
        },
        "distributions": distributions,
        "top_cells": [
            {
                "rank": row.get("rank"),
                "cell_id": row.get("cell_id"),
                "distance": row.get("distance"),
                "metadata": {
                    key: row.get("metadata", {}).get(key)
                    for key in fields
                    if key in row.get("metadata", {})
                },
            }
            for row in rows[:10]
        ],
    }


def _metadata_indices(searcher: Any, filters: dict[str, str]) -> np.ndarray:
    metadata = searcher.metadata
    mask = np.ones(len(metadata), dtype=bool)
    for key, value in filters.items():
        if key not in metadata.columns:
            raise ValueError(f"元数据中不存在过滤列: '{key}'")
        mask &= metadata[key].astype(str).to_numpy() == str(value)
    return np.where(mask)[0]


def _clean_filters(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(val).strip()
        for key, val in value.items()
        if str(key).strip() and str(val).strip()
    }


def _extract_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise ValueError(f"模型未返回 JSON 对象: {content[:200]}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("模型返回值必须是 JSON 对象")
    return parsed


def _local_summary(stats: dict[str, Any]) -> str:
    result_count = stats.get("result_count", 0)
    parts = [f"共得到 {result_count} 个结果。"]
    for field, dist in stats.get("distributions", {}).items():
        if dist:
            top = dist[0]
            parts.append(f"{field} 以 {top['value']} 为主，占比约 {top['ratio']:.1%}。")
    return "".join(parts)


def _local_suggestions(stats: dict[str, Any]) -> list[str]:
    suggestions = ["可尝试扩大 Top-K 或增加疾病/细胞类型过滤条件进行对照。"]
    if stats.get("distance", {}).get("std") is not None:
        suggestions.append("如果距离分布较分散，建议检查查询细胞是否处于过渡状态或批次混杂。")
    return suggestions


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
