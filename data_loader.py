import scanpy as sc

# ---------- 1. 读取数据 ----------

adata = sc.read_h5ad("data/liver.h5ad")

print(adata)

# 查看细胞和基因数量

print("细胞数:", adata.n_obs)

print("基因数:", adata.n_vars)

# PCA降维结果在obsm下的X_pca字段
X_pca = adata.obsm["X_pca"]

print(X_pca)
print(X_pca.shape)

# 查看细胞类型

print(adata.obs["cell_type"].value_counts())


# UMAP

sc.pl.umap(adata,color=["cell_type", "disease", "AgeGroup"])
