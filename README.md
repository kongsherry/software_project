# software_project

单细胞高维向量数据的 ANN（近似最近邻）检索系统。

## 中期：数据读取 + 向量化导出

数据已提供 PCA 结果：`obsm['X_pca']`。

运行（项目根目录）：

```bash
python data_loader.py --input data/liver.h5ad --outdir results
```

可选参数（保持简单）：

```bash
# 选择 embedding（来自 obsm，默认 X_pca）
python data_loader.py --embedding X_pca

# 取前 50 维；或用 -1 导出全部维度
python data_loader.py --dims 50
python data_loader.py --dims -1

# 关闭 L2 归一化
python data_loader.py --no-l2

# 导出更多元数据列
python data_loader.py --obs-cols cell_type,disease,AgeGroup,sex,donor_id
```

输出（默认到 `results/`）：
- `vectors.npy`：用于 ANN 建索引的向量（默认取 PCA 前 30 维，可用 `--dims` 调整）
- `cell_ids.npy`：与 vectors 行对应的 cell_id
- `obs_metadata.csv`：cell_id + 常用元数据列（cell_type/disease/AgeGroup，存在才导出）
- `summary.json`：数据规模、embedding 形状、导出列等摘要信息（便于展示/排查）

## 中期：ANN 索引构建
承接数据输出的高维向量矩阵，构建基于 FAISS 的空间图索引，支持持久化存取。

运行（项目根目录，依赖上述 vectors.npy）：
```bash
python index_builder.py --input results/vectors.npy --outdir indices
```

可选参数（保持简单）：

```bash
### 选择构建的索引类型（支持 hnsw 或 flat，默认 hnsw）
python index_builder.py --type flat

### 调优 HNSW 结构参数：最大连接数（默认 32）
python index_builder.py --M 64

### 调优 HNSW 结构参数：构建候选集大小（默认 200）
python index_builder.py --ef 300
```

输出（默认到 indices/）：

hnsw_M32_ef200.index：用于核心检索加速的 HNSW 图结构索引。

flat.index：用于基准对比的精确暴力检索索引（仅在使用 --type flat 时生成）。

注：核心逻辑已封装为 AnnIndexBuilder 类，供下游检索逻辑与 Flask API 直接 import 并在启动时加载到内存调用。

## 中期：ANN 检索逻辑与结果格式化

封装 `AnnSearcher` 类，支持按细胞 ID 或原始向量进行 ANN 检索，并将结果关联细胞元数据。

运行（项目根目录，依赖 P1 导出文件与 P2 索引文件）：

```bash
# 按向量检索（默认取索引 0 的向量作为查询）
python search.py

# 按细胞 ID 检索
python search.py --cell-id "AAACCTGAGCAGGTCA-1_2"

# 指定返回 Top-K 数量
python search.py --cell-id "AAACCTGAGCAGGTCA-1_2" --k 20
```

可选参数：

```bash
# 自定义索引/数据文件路径
python search.py --index indices/hnsw_M32_ef200.index \
                 --vectors results/vectors.npy \
                 --metadata results/obs_metadata.csv \
                 --cell-ids results/cell_ids.npy
```

返回格式（JSON 友好）：

```json
{
  "query": {"cell_id": "AAAC...", "k": 10, "metric": "l2"},
  "time_ms": 0.134,
  "results": [
    {
      "rank": 1,
      "cell_id": "AAACCTGAGCAGGTCA-1_2",
      "distance": 0.0,
      "metadata": {"cell_type": "hepatocyte", "disease": "normal", "AgeGroup": "Ped"}
    }
  ]
}
```

核心接口一览：

| 方法 | 说明 |
| --- | --- |
| `AnnSearcher(index_path, vectors_path, metadata_path, cell_ids_path)` | 加载索引、向量、细胞 ID 及元数据 |
| `.search_by_cell_id(cell_id, k=10)` | 按细胞 ID 检索，无效 ID 抛出 `KeyError` |
| `.search_by_vector(vector, k=10)` | 按原始向量检索，维度不匹配抛出 `ValueError` |

检索耗时覆盖 FAISS 底层 `index.search()` 调用，结果自动将 numpy/Pandas 类型转为 Python 原生类型，可直接 JSON 序列化供 Flask API 使用。