"""生成测试用的合成单细胞 h5ad 数据文件。"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ── 参数 ──────────────────────────────────────────────
N_CELLS = 500          # 细胞数
N_GENES = 2000         # 基因数
N_PCA = 50             # PCA 维度
OUTPUT = Path("data/test_data.h5ad")
SEED = 42

rng = np.random.default_rng(SEED)

# ── 合成基因表达矩阵 ─────────────────────────────────
# 模拟几个细胞类型的基因表达模式
cell_types = ["T-cell", "B-cell", "Monocyte", "NK-cell", "Hepatocyte"]
diseases = ["Healthy", "Cirrhosis", "HCC"]
age_groups = ["Young", "Middle", "Senior"]

n_types = len(cell_types)
# 每种细胞类型有一个"特征基因"模式（稀疏 + 类型特异）
base_means = rng.exponential(0.5, size=(n_types, N_GENES))
# 让每种类型有 50 个高表达标记基因
for i in range(n_types):
    marker_start = i * (N_GENES // n_types)
    marker_end = marker_start + 50
    base_means[i, marker_start:marker_end] = rng.exponential(3.0, size=50)

# 分配细胞类型并按类型生成表达
cell_type_labels = rng.choice(cell_types, size=N_CELLS)
X = np.zeros((N_CELLS, N_GENES), dtype=np.float32)
for i, ct in enumerate(cell_type_labels):
    ct_idx = cell_types.index(ct)
    # 基础表达 + 噪声
    X[i] = rng.poisson(base_means[ct_idx] + 0.1).astype(np.float32)

# 加一点随机 dropout（模拟单细胞数据的稀疏性）
dropout_mask = rng.random((N_CELLS, N_GENES)) < 0.3
X[dropout_mask] = 0.0

# ── PCA 嵌入 ─────────────────────────────────────────
pca = PCA(n_components=N_PCA, random_state=SEED)
X_pca = pca.fit_transform(X).astype(np.float32)

# ── 细胞 ID ──────────────────────────────────────────
cell_ids = [f"cell_{i:04d}" for i in range(N_CELLS)]

# ── 元数据 ───────────────────────────────────────────
disease_labels = rng.choice(diseases, size=N_CELLS)
age_labels = rng.choice(age_groups, size=N_CELLS)

obs = pd.DataFrame(
    {
        "cell_type": pd.Categorical(cell_type_labels),
        "disease": pd.Categorical(disease_labels),
        "AgeGroup": pd.Categorical(age_labels),
        # 额外列，测试系统对多余列的兼容性
        "n_counts": X.sum(axis=1).astype(np.int32),
        "n_genes": (X > 0).sum(axis=1).astype(np.int32),
    },
    index=cell_ids,
)

# ── var（基因名）───────────────────────────────────────
gene_ids = [f"gene_{j:04d}" for j in range(N_GENES)]
var = pd.DataFrame(index=gene_ids)

# ── 组装 AnnData ─────────────────────────────────────
adata = ad.AnnData(
    X=X,
    obs=obs,
    var=var,
    dtype=np.float32,
)
adata.obs_names = cell_ids
adata.var_names = gene_ids
adata.obsm["X_pca"] = X_pca

# ── 保存 ─────────────────────────────────────────────
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(str(OUTPUT), compression="gzip")

print(f"[OK] Test data generated: {OUTPUT}")
print(f"     Cells: {adata.n_obs}")
print(f"     Genes: {adata.n_vars}")
print(f"     PCA shape: {X_pca.shape}")
print(f"     obs columns: {list(adata.obs.columns)}")
print(f"     obsm keys: {list(adata.obsm_keys())}")
print(f"     Cell type distribution:\n{adata.obs['cell_type'].value_counts().to_string()}")
print(f"     Disease distribution:\n{adata.obs['disease'].value_counts().to_string()}")
print(f"     Age group distribution:\n{adata.obs['AgeGroup'].value_counts().to_string()}")
