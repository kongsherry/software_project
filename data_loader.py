import argparse
from pathlib import Path

import numpy as np
import scanpy as sc


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:

	norms = np.linalg.norm(x, axis=1, keepdims=True)
	return x / np.maximum(norms, eps)


def main() -> None:

	p = argparse.ArgumentParser(description="单细胞数据读取 + PCA 向量导出（中期提交）")
	p.add_argument("--input", default="data/liver.h5ad", help="输入 .h5ad 路径")
	p.add_argument("--outdir", default="results", help="输出目录")
	p.add_argument("--dims", type=int, default=30, help="取 PCA 前多少维（默认 30）")
	p.set_defaults(l2=True)
	p.add_argument("--l2", dest="l2", action="store_true", help="对每个向量做 L2 归一化（默认开启）")
	p.add_argument("--no-l2", dest="l2", action="store_false", help="关闭 L2 归一化")
	args = p.parse_args()

	input_path = Path(args.input)
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	adata = sc.read_h5ad(str(input_path))
	if "X_pca" not in adata.obsm_keys():
		raise KeyError("数据中未找到 obsm['X_pca']（PPT 说明应已提供 PCA 结果）")

	vectors = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
	if vectors.ndim != 2:
		raise ValueError(f"X_pca 应为二维矩阵，当前 shape={vectors.shape}")
	if args.dims <= 0 or args.dims > vectors.shape[1]:
		raise ValueError(f"--dims 需在 1~{vectors.shape[1]} 之间")
	vectors = vectors[:, : args.dims]

	if args.l2:
		vectors = l2_normalize_rows(vectors).astype(np.float32, copy=False)

	cell_ids = np.asarray(list(adata.obs_names), dtype=object)

	# 导出少量常用元数据列（不存在则跳过）
	cols = [c for c in ["cell_type", "disease", "AgeGroup"] if c in adata.obs.columns]
	metadata = adata.obs.loc[:, cols].copy()
	metadata.insert(0, "cell_id", adata.obs_names)

	np.save(str(outdir / "vectors.npy"), vectors)
	np.save(str(outdir / "cell_ids.npy"), cell_ids)
	metadata.to_csv(str(outdir / "obs_metadata.csv"), index=False, encoding="utf-8")

	print("导出完成")
	print(f"- cells={adata.n_obs}, genes={adata.n_vars}")
	print(f"- vectors shape={vectors.shape}, dtype={vectors.dtype}")
	print(f"- outdir={outdir}")


if __name__ == "__main__":
	main()
