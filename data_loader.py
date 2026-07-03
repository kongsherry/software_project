import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:

	norms = np.linalg.norm(x, axis=1, keepdims=True)
	return x / np.maximum(norms, eps)


def _split_csv(value: str) -> list[str]:
	return [x.strip() for x in value.split(",") if x.strip()]


def _safe_json(obj: Any) -> Any:
	if isinstance(obj, (str, int, float, bool)) or obj is None:
		return obj
	if isinstance(obj, (list, tuple)):
		return [_safe_json(x) for x in obj]
	if isinstance(obj, dict):
		return {str(k): _safe_json(v) for k, v in obj.items()}
	return str(obj)


def export_h5ad(
	input_path: str | Path,
	outdir: str | Path,
	embedding: str = "X_pca",
	dims: int = 30,
	obs_cols: str | list[str] = "cell_type,disease,AgeGroup,sex,Treatment,Phase,seurat_clusters,donor_age",
	l2: bool = True,
) -> dict[str, Any]:
	"""读取 .h5ad 并导出 ANN 检索所需的向量、细胞 ID 和元数据。"""
	input_path = Path(input_path)
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	# 使用 backed='r' 模式：只读取元数据，避免将庞大的基因表达矩阵 X 加载到内存
	adata = sc.read_h5ad(str(input_path), backed="r")

	emb_key = str(embedding)
	if emb_key not in adata.obsm_keys():
		available = list(map(str, adata.obsm_keys()))
		raise KeyError(f"数据中未找到 obsm['{emb_key}']，可用：{available}")

	vectors = np.asarray(adata.obsm[emb_key], dtype=np.float32)
	if vectors.ndim != 2:
		raise ValueError(f"obsm['{emb_key}'] 应为二维矩阵，当前 shape={vectors.shape}")

	dims = int(dims)
	if dims != -1:
		if dims <= 0 or dims > vectors.shape[1]:
			raise ValueError(f"--dims 需为 -1 或 1~{vectors.shape[1]} 之间")
		vectors = vectors[:, :dims]

	if l2:
		vectors = l2_normalize_rows(vectors).astype(np.float32, copy=False)

	if not np.isfinite(vectors).all():
		raise ValueError("向量中存在 NaN/Inf，无法用于索引构建")

	cell_ids = np.asarray(list(adata.obs_names), dtype=object)

	requested_cols = _split_csv(obs_cols) if isinstance(obs_cols, str) else list(obs_cols)
	cols = [c for c in requested_cols if c in adata.obs.columns]
	metadata = adata.obs.loc[:, cols].copy() if cols else adata.obs.iloc[:, 0:0].copy()
	metadata.insert(0, "cell_id", adata.obs_names)

	vectors_path = outdir / "vectors.npy"
	cell_ids_path = outdir / "cell_ids.npy"
	metadata_path = outdir / "obs_metadata.csv"
	summary_path = outdir / "summary.json"

	np.save(str(vectors_path), vectors)
	np.save(str(cell_ids_path), cell_ids)
	metadata.to_csv(str(metadata_path), index=False, encoding="utf-8")

	summary: dict[str, Any] = {
		"input": str(input_path),
		"n_obs": int(adata.n_obs),
		"n_vars": int(adata.n_vars),
		"embedding": {"key": emb_key, "shape": list(map(int, vectors.shape)), "l2": bool(l2)},
		"export": {
			"vectors": str(vectors_path),
			"cell_ids": str(cell_ids_path),
			"metadata": str(metadata_path),
			"metadata_cols": cols,
		},
	}
	if "cell_type" in adata.obs:
		vc = adata.obs["cell_type"].value_counts(dropna=False)
		summary["cell_type_counts_top10"] = _safe_json(vc.head(10).to_dict())

	summary_path.write_text(json.dumps(_safe_json(summary), ensure_ascii=False, indent=2), encoding="utf-8")
	return summary


def main() -> None:

	p = argparse.ArgumentParser(description="单细胞数据读取 + 向量化导出（中期提交）")
	p.add_argument("--input", default="data/liver.h5ad", help="输入 .h5ad 路径")
	p.add_argument("--outdir", default="results", help="输出目录")
	p.add_argument("--embedding", default="X_pca", help="使用 obsm 里的哪个 embedding（默认 X_pca）")
	p.add_argument("--dims", type=int, default=30, help="取前多少维（默认 30；-1 表示不截断）")
	p.add_argument(
		"--obs-cols",
		default="cell_type,disease,AgeGroup,sex,Treatment,Phase,seurat_clusters,donor_age",
		help="导出到 metadata 的 obs 列（逗号分隔，不存在会跳过）",
	)
	p.set_defaults(l2=True)
	p.add_argument("--l2", dest="l2", action="store_true", help="对每个向量做 L2 归一化（默认开启）")
	p.add_argument("--no-l2", dest="l2", action="store_false", help="关闭 L2 归一化")
	args = p.parse_args()

	summary = export_h5ad(
		input_path=args.input,
		outdir=args.outdir,
		embedding=args.embedding,
		dims=args.dims,
		obs_cols=args.obs_cols,
		l2=args.l2,
	)

	print("导出完成")
	print(f"- cells={summary['n_obs']}, genes={summary['n_vars']}")
	print(f"- embedding={summary['embedding']['key']}, vectors shape={summary['embedding']['shape']}")
	print(f"- outdir={args.outdir}")


if __name__ == "__main__":
	main()
