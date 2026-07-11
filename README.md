# 单细胞高维向量 ANN 检索系统

本项目是一个面向单细胞数据的近似最近邻检索系统，支持 `.h5ad` 数据集上传、向量导出、FAISS 索引构建、条件过滤检索、二维散点图可视化、点击细胞反向查询、联合检索、实时性能评估、AI 自然语言查询与结果解释，以及用户认证与管理员权限管理。

历史开发进度已迁移到 [PROGRESS.md](PROGRESS.md)。

> **⚠️ 重要提示：** 本项目 Git 仓库不包含数据文件（`.h5ad`、`vectors.npy`、`.index` 等）。克隆后需要先生成测试数据或准备你自己的 `.h5ad` 数据集才能运行。详见下方快速启动说明。

## 功能概览

- 数据集管理：管理员上传、切换、删除 `.h5ad` 数据集。
- 向量化导出：读取 `obsm["X_pca"]`，导出 `vectors.npy`、`cell_ids.npy`、`obs_metadata.csv`。
- ANN 检索：基于 FAISS HNSW、IVF+HNSW 或 Flat 索引，支持按 Cell ID 或原始向量查询 Top-K 近邻。
- 条件过滤：支持按 `cell_type`、`disease`、`AgeGroup` 等元数据字段过滤；支持多选 OR 过滤和数值范围过滤。
- 可视化：读取 `obsm["X_umap"]` 或 `obsm["X_tsne"]`，提供散点图数据接口，前端支持点击细胞后反向查询近邻。
- 检索精度控制：前端可调精度滑块，后端动态设置 HNSW `efSearch` 或 IVF `nprobe`。
- 联合检索：可加载多个数据集，在多个索引中并行搜索并按距离合并排序。
- 性能评估：结合实时查询指标与离线评估报告，展示 Recall@K、平均延迟、P95、QPS、索引大小。
- 用户系统：支持注册、登录、退出、管理员用户管理、角色变更和密码重置。
- AI 辅助分析：基于 DeepSeek 大模型支持自然语言查询细胞、检索结果解释和后续分析建议。

## 目录结构

```text
.
├── app.py                  # Flask Web 服务与 API 路由
├── data_loader.py          # .h5ad 数据读取与向量导出
├── index_builder.py        # FAISS HNSW/IVF+HNSW/Flat 索引构建
├── search.py               # ANN 检索与条件过滤逻辑
├── multi_search.py         # 多数据集联合检索
├── ai_analyzer.py          # DeepSeek 自然语言查询与结果分析
├── dataset_manager.py      # 数据集上传、切换、删除与 manifest 管理
├── visualize.py            # UMAP/t-SNE 散点图数据接口
├── evaluate.py             # Recall/QPS/延迟评估脚本
├── user_manager.py         # 用户认证与权限管理
├── generate_test_h5ad.py   # 生成带 X_pca 和 X_umap 的测试数据
├── templates/              # 前端页面模板
├── static/                 # CSS 与前端静态资源
├── indices/                # FAISS 索引文件
└── evaluation_report.json  # 性能评估报告
```

运行时会生成或使用以下数据目录：

```text
data/
results/
indices/
```

其中上传数据集默认会保存到：

```text
data/datasets/<dataset_id>/
results/datasets/<dataset_id>/
indices/datasets/<dataset_id>/
results/datasets/manifest.json
```

## 环境依赖

### 方式一：使用 requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方式二：手动安装

建议使用已有课程环境或 Conda 环境。核心 Python 依赖包括：

```text
flask>=3.0
numpy>=1.26
pandas>=2.0
scanpy>=1.10
anndata>=0.10
scikit-learn>=1.3
faiss-cpu>=1.7
werkzeug>=3.0
```

如果需要手动安装，可参考：

```bash
pip install flask numpy pandas scanpy anndata scikit-learn faiss-cpu werkzeug
```

### Conda 环境（如使用课程环境）

如果使用 Anaconda/ Miniconda，建议先激活对应环境：

```bash
conda activate <你的环境名>
```

验证依赖是否安装成功：

```bash
python -c "import flask, numpy, pandas, scanpy, anndata, sklearn, faiss; print('所有依赖已就绪')"
```

### DeepSeek 配置

AI 辅助功能默认使用 DeepSeek OpenAI-compatible 接口，模型名为：

```text
deepseek-v4-flash
```

启动服务前配置 API Key：

```bash
set DEEPSEEK_API_KEY=你的DeepSeek API Key
```

可选配置：

```bash
set DEEPSEEK_MODEL=deepseek-v4-flash
set DEEPSEEK_BASE_URL=https://api.deepseek.com
set DEEPSEEK_THINKING=disabled
set DEEPSEEK_MAX_TOKENS=4096
```

如果未配置 `DEEPSEEK_API_KEY`，自然语言查询接口会返回配置错误；“AI 分析当前结果”会退化为本地统计摘要。

说明：`deepseek-v4-flash` 默认开启思考模式；本项目的自然语言查询需要稳定返回 JSON，因此默认通过 `thinking={"type":"disabled"}` 关闭思考模式。需要查看推理过程时可以改为 `DEEPSEEK_THINKING=enabled`。

当前 API Key 只从服务端环境变量读取，暂不支持在浏览器页面中持久配置。

## 快速启动

### Git 克隆后的初始状态

项目仓库不包含以下文件（已通过 `.gitignore` 排除）：

- `data/` 目录下的 `.h5ad` 数据集文件
- `results/` 目录下的向量和元数据导出产物
- `indices/` 目录下的 FAISS 索引文件（`*.index`）
- `evaluation_report.json` 等运行时生成文件

因此，克隆后有两种方式让系统运行起来：

---

### 方式 A：通过 Web 界面操作（推荐 ✅）

这是最简单的上手方式，所有操作在浏览器中完成。

#### 1. 生成测试数据

```bash
python generate_test_h5ad.py
```

此脚本会在 `data/` 目录下生成 `test_data.h5ad`，包含 500 个合成细胞、5 种细胞类型、2000 个基因、PCA 和 UMAP 嵌入。

#### 2. 启动 Web 服务

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

首次启动会自动创建默认管理员账号：

```text
用户名：admin
密码：admin123
```

#### 3. 上传测试数据集

登录管理员（admin / admin123）后，在首页”数据集管理”面板中上传：

```text
data/test_data.h5ad
```

上传后系统会自动完成：

```text
读取 .h5ad -> 导出向量和元数据 -> 构建 FAISS 索引 -> 设置为活动数据集
```

> **说明：** 通过 Web 界面上传的数据集会完整注册到系统，散点图可视化、条件过滤、性能评估等全部功能均可正常使用。这也是推荐的新手路径。

#### 4. 使用检索与可视化

切换到”探索”页后，可以：

- 按 Cell ID 查询 Top-K 近邻。（测试数据示例：`cell_0044`、`cell_0100`）
- 粘贴原始向量进行查询。
- 添加元数据过滤条件，包括多选和数值范围过滤。
- 通过精度滑块调整 ANN 检索参数。
- 使用自然语言查询（需配置 DeepSeek API Key）。
- 查看 UMAP 散点图并点击细胞反向查询。
- 对当前检索结果执行 AI 分析。

切换到”联合检索”页后，可以加载多个数据集进行跨数据集检索。

---

### 方式 B：通过命令行操作

适合只想在终端中测试检索功能的场景。注意：CLI 路径导出的默认数据集的散点图可视化需要 `.h5ad` 源文件匹配，因此建议仅在不需要可视化的场景下使用，或确保使用的 `.h5ad` 文件路径一致。

#### 1. 准备 .h5ad 数据

如果有真实数据集（如 `data/liver.h5ad`），直接使用。否则先生成测试数据：

```bash
python generate_test_h5ad.py
```

#### 2. 导出向量

```bash
python data_loader.py --input data/test_data.h5ad --outdir results
```

常用参数：

```bash
python data_loader.py --embedding X_pca
python data_loader.py --dims 30
python data_loader.py --dims -1
python data_loader.py --no-l2
python data_loader.py --obs-cols cell_type,disease,AgeGroup,n_counts,n_genes
```

输出：

```text
results/vectors.npy
results/cell_ids.npy
results/obs_metadata.csv
results/summary.json
```

#### 3. 构建索引

```bash
python index_builder.py --input results/vectors.npy --outdir indices
```

常用参数：

```bash
python index_builder.py --type hnsw --M 32 --ef 200
python index_builder.py --type ivf_hnsw --nlist 256 --M 32 --ef 200
python index_builder.py --type flat
```

输出：

```text
indices/hnsw_M32_ef200.index
indices/ivf_hnsw_nlist256_M32_ef200.index
indices/flat.index
```

#### 4. 命令行检索

```bash
python search.py --cell-id cell_0044 --k 10
```

#### 5. （可选）启动 Web 服务

```bash
python app.py
```

此时系统会检测到 `results/` 和 `indices/` 下已有产物，自动注册为只读的 `default` 数据集。检索和指标功能可用，但散点图功能需要 `.h5ad` 源文件与产物匹配。

---

### 测试数据说明

测试数据脚本 `generate_test_h5ad.py` 生成的合成数据包含：

| 属性 | 值 |
|------|-----|
| 细胞数 | 500 |
| 基因数 | 2000 |
| PCA 维度 | 50（默认检索使用前 30 维） |
| UMAP 维度 | 2 |
| 细胞类型 | T-cell, B-cell, Monocyte, NK-cell, Hepatocyte |
| 疾病状态 | Healthy, Cirrhosis, HCC |
| 年龄组 | Young, Middle, Senior |

测试数据中的 Cell ID 示例：

```text
cell_0000
cell_0044
cell_0100
```

## 命令行流程

以下为独立于 Web 界面的命令行操作参考。如果你已通过 Web 界面上传数据集，无需手动执行以下命令。

### 导出向量

```bash
python data_loader.py --input data/test_data.h5ad --outdir results
```

常用参数：

```bash
python data_loader.py --embedding X_pca
python data_loader.py --dims 30
python data_loader.py --dims -1
python data_loader.py --no-l2
python data_loader.py --obs-cols cell_type,disease,AgeGroup,n_counts,n_genes
```

输出：

```text
results/vectors.npy
results/cell_ids.npy
results/obs_metadata.csv
results/summary.json
```

### 构建索引

```bash
python index_builder.py --input results/vectors.npy --outdir indices
```

常用参数：

```bash
python index_builder.py --type hnsw --M 32 --ef 200
python index_builder.py --type ivf_hnsw --nlist 256 --M 32 --ef 200
python index_builder.py --type flat
```

输出：

```text
indices/hnsw_M32_ef200.index
indices/ivf_hnsw_nlist256_M32_ef200.index
indices/flat.index
```

### 命令行检索

```bash
python search.py --cell-id cell_0044 --k 10
```

自定义文件路径：

```bash
python search.py --index indices/hnsw_M32_ef200.index ^
                 --vectors results/vectors.npy ^
                 --metadata results/obs_metadata.csv ^
                 --cell-ids results/cell_ids.npy ^
                 --cell-id cell_0044 ^
                 --k 10
```

### 运行性能评估

```bash
python evaluate.py --dataset-id test_data --sample-size 200 --report evaluation_report.json
```

生成报告后，在 Web 页面“性能评估”页查看 Recall、延迟、QPS 和索引大小。

如果不传 `--dataset-id`，会评估当前活动数据集。Web 页面也会为当前活动数据集自动生成或读取对应评估报告。

## Web API

以下 API 均运行在：

```text
http://127.0.0.1:5000
```

除注册、登录接口外，其余业务接口需要登录；数据集上传、删除和用户管理需要管理员权限。当前实现中，已登录用户可以切换活动数据集。

### 认证

注册：

```bash
curl -X POST http://127.0.0.1:5000/api/auth/register ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"testuser\",\"password\":\"testpass\"}"
```

登录：

```bash
curl -X POST http://127.0.0.1:5000/api/auth/login ^
  -H "Content-Type: application/json" ^
  -d "{\"username\":\"admin\",\"password\":\"admin123\"}"
```

查看当前用户：

```bash
curl http://127.0.0.1:5000/api/auth/me
```

退出：

```bash
curl -X POST http://127.0.0.1:5000/api/auth/logout
```

### 数据集管理

列出数据集：

```bash
curl http://127.0.0.1:5000/datasets
```

上传数据集，需管理员权限：

```bash
curl -X POST http://127.0.0.1:5000/datasets ^
  -F "file=@data/test_data.h5ad" ^
  -F "name=test_data" ^
  -F "dims=30" ^
  -F "index_type=ivf_hnsw" ^
  -F "nlist=256" ^
  -F "activate=true"
```

当前上传接口使用 `X_pca` 作为 embedding，默认导出 `cell_type,disease,AgeGroup,sex,Treatment,Phase,seurat_clusters,donor_age` 等元数据列。`index_type` 支持 `hnsw`、`ivf_hnsw`、`flat`，`nlist` 仅对 `ivf_hnsw` 生效；不填写 `nlist` 时会自动估计。

切换活动数据集，需登录：

```bash
curl -X POST http://127.0.0.1:5000/datasets/test_data/activate
```

删除上传的数据集，需管理员权限：

```bash
curl -X DELETE http://127.0.0.1:5000/datasets/test_data
```

默认数据集 `default` 是兼容已有产物的只读数据集，不能删除。

### 检索

按 Cell ID 检索：

```bash
curl -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10}"
```

按向量检索：

```bash
curl -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"vector\":[0.1,0.2,0.3],\"k\":10}"
```

带元数据过滤：

```bash
curl -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10,\"filters\":{\"disease\":\"HCC\",\"cell_type\":\"NK-cell\"}}"
```

多选过滤和数值范围过滤：

```bash
curl -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10,\"filters\":{\"cell_type\":[\"NK-cell\",\"T-cell\"],\"donor_age\":{\"op\":\">=\",\"value\":50}}}"
```

指定 ANN 精度参数：

```bash
curl -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10,\"search_params\":{\"precision_pct\":80,\"ef_search\":128,\"nprobe\":16}}"
```

返回结果包含：

```text
query
time_ms
results
filter_info
search_profile
request_time_ms
metrics
```

### 可视化

获取散点图数据：

```bash
curl "http://127.0.0.1:5000/scatter_data?max_points=5000"
```

点击细胞反向查询：

```bash
curl -X POST http://127.0.0.1:5000/scatter_search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10}"
```

获取当前数据集可用过滤字段取值：

```bash
curl http://127.0.0.1:5000/filter_options
```

### 联合检索

加载一个或多个数据集到联合检索器：

```bash
curl -X POST http://127.0.0.1:5000/api/multi_load ^
  -H "Content-Type: application/json" ^
  -d "{\"dataset_ids\":[\"default\",\"test_data\"]}"
```

查看已加载数据集：

```bash
curl http://127.0.0.1:5000/api/multi_status
```

跨数据集检索：

```bash
curl -X POST http://127.0.0.1:5000/api/multi_search ^
  -H "Content-Type: application/json" ^
  -d "{\"dataset_ids\":[\"default\",\"test_data\"],\"cell_id\":\"cell_0044\",\"k\":10,\"filters\":{\"disease\":\"HCC\"}}"
```

卸载指定数据集：

```bash
curl -X DELETE http://127.0.0.1:5000/api/multi_load/test_data
```

管理员可以将已加载的多个数据集构建为合并索引：

```bash
curl -X POST http://127.0.0.1:5000/api/multi_merge ^
  -H "Content-Type: application/json" ^
  -d "{\"dataset_ids\":[\"default\",\"test_data\"],\"output_path\":\"indices/merged/merged_hnsw.index\",\"index_type\":\"hnsw\"}"
```

### 性能报告

```bash
curl http://127.0.0.1:5000/metrics
```

该接口读取 `evaluation_report.json`。如果报告不存在，请先运行：

```bash
python evaluate.py
```

### AI 辅助分析

自然语言查询并自动分析：

```bash
curl -X POST http://127.0.0.1:5000/ai/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"找 HCC 样本中最像 Kupffer cell 的 20 个细胞\"}"
```

支持的查询计划包括：

```text
search_by_cell_id   # 例如：找出和 cell_0044 最像的 10 个细胞
centroid_search     # 例如：找 HCC 样本中最像 Kupffer cell 的 20 个细胞
metadata_filter     # 例如：查询 Healthy 成人样本里的 hepatocyte
```

对已有检索结果做 AI 分析：

```bash
curl -X POST http://127.0.0.1:5000/ai/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"cell_0044\",\"k\":10}"
```

## 权限说明

| 操作 | 未登录 | 普通用户 | 管理员 |
| --- | --- | --- | --- |
| 注册 / 登录 | ✅ | ✅ | ✅ |
| 查看首页 | ❌ | ✅ | ✅ |
| 检索 | ❌ | ✅ | ✅ |
| 查看散点图 | ❌ | ✅ | ✅ |
| 查看性能报告 | ❌ | ✅ | ✅ |
| 上传数据集 | ❌ | ❌ | ✅ |
| 切换数据集 | ❌ | ✅ | ✅ |
| 删除数据集 | ❌ | ❌ | ✅ |
| 联合检索 | ❌ | ✅ | ✅ |
| AI 查询 / 分析 | ❌ | ✅ | ✅ |
| 用户管理 | ❌ | ❌ | ✅ |

## 完整测试方法

### 基础功能测试

```bash
python generate_test_h5ad.py
python app.py
```

然后在浏览器中：

```text
1. 打开 http://127.0.0.1:5000
2. 使用 admin / admin123 登录
3. 上传 data/test_data.h5ad（在”数据集管理”面板）
4. 切换到”探索”页
5. 输入 cell_0044，Top-K 设置为 10，执行检索
6. 添加过滤条件 disease = HCC 或 cell_type = NK-cell，再次检索
7. 调整精度滑块，检查结果表头中的 efSearch / nprobe / 精度信息
8. 使用自然语言查询”找 HCC 样本中最像 NK-cell 的 5 个细胞”（需配置 DeepSeek API Key）
9. 点击”AI 分析当前结果”，检查分析面板
10. 点击散点图中的细胞，检查结果表是否刷新
11. 切换到”联合检索”页，加载 test_data 后执行跨数据集检索
12. 切换到”性能评估”页，查看实时指标和离线 Recall 报告
```

### 命令行接口测试（可选）

以下命令可快速验证检索功能是否正常：

```bash
# 1. 生成测试数据
python generate_test_h5ad.py

# 2. 导出向量（500 个细胞，30 维 PCA）
python data_loader.py --input data/test_data.h5ad --outdir results

# 3. 构建 HNSW 索引
python index_builder.py --input results/vectors.npy --outdir indices

# 4. 命令行检索测试
python search.py --cell-id cell_0044 --k 10

# 预期输出：
#   [1] cell_0044, distance=0.0000  ← 自身距离为 0
#   [2] cell_XXXX, distance=0.xxxx  ← NK-cell 类型的近邻细胞
#   ...
```

### API 接口测试（curl）

```bash
# 登录并保存 Cookie
curl -c cookies.txt -X POST http://127.0.0.1:5000/api/auth/login \
  -H “Content-Type: application/json” \
  -d “{\”username\”:\”admin\”,\”password\”:\”admin123\”}”

# 按 Cell ID 检索
curl -b cookies.txt -X POST http://127.0.0.1:5000/search \
  -H “Content-Type: application/json” \
  -d “{\”cell_id\”:\”cell_0044\”,\”k\”:5}”

# 带条件过滤检索
curl -b cookies.txt -X POST http://127.0.0.1:5000/search \
  -H “Content-Type: application/json” \
  -d “{\”cell_id\”:\”cell_0044\”,\”k\”:10,\”filters\”:{\”disease\”:\”HCC\”,\”cell_type\”:\”NK-cell\”}}”

# 获取散点图数据
curl -b cookies.txt “http://127.0.0.1:5000/scatter_data?max_points=100”

# 获取可用过滤字段
curl -b cookies.txt http://127.0.0.1:5000/filter_options

# 获取系统状态
curl -b cookies.txt http://127.0.0.1:5000/status

# 获取性能指标
curl -b cookies.txt http://127.0.0.1:5000/metrics
```

### 权限测试

普通用户登录后直接访问以下接口应返回 `403`：

```text
POST /datasets
DELETE /datasets/<dataset_id>
POST /api/multi_merge
```

管理员访问：

```text
POST /datasets             # 不带文件时返回 400，说明已通过权限检查进入业务校验
POST /datasets/default/activate  # 登录用户即可返回 200
DELETE /datasets/default   # 返回 400，因为 default 数据集受保护
```

### 语法检查

```bash
python -m py_compile app.py ai_analyzer.py search.py dataset_manager.py visualize.py user_manager.py multi_search.py data_loader.py index_builder.py evaluate.py generate_test_h5ad.py
```

## 故障排查

### 启动后检索失败，提示"未找到细胞ID"

**症状：** 搜索 `cell_0044` 时返回错误，提示可用的 ID 示例为 `AAACCTGAGCAGGTCA-1_2` 等。

**原因：** 当前活动数据集使用的是旧的 liver 数据产物。可能是之前运行过 `data_loader.py` 导出了 liver 数据。

**解决：** 通过 Web 界面上传 `data/test_data.h5ad`，系统会自动完成导出、建索引和激活。

### 散点图无法显示，提示"元数据行数与 h5ad 细胞数不一致"

**症状：** 访问 `/scatter_data` 返回 400 错误，提示行数不一致（如 500 vs 69032）。

**原因：** 当前数据集引用的 `.h5ad` 源文件与 `results/` 下的向量/元数据产物不匹配。常见于：
- 用 `data_loader.py` 从 `test_data.h5ad`（500 细胞）导出，但 `source_path` 指向 `data/liver.h5ad`（69032 细胞）
- 或者反过来

**解决：**
1. 推荐：通过 Web 界面上传 `.h5ad`（系统自动保持源文件与产物一致）
2. 或：确保 CLI 导出的 `--input` 路径与后续使用的源文件一致

### 端口 5000 被占用

**症状：** 启动 Flask 时提示 `Address already in use`。

**解决：**
```bash
# Windows: 查找并关闭占用端口的进程
netstat -ano | findstr ":5000"
taskkill //PID <进程ID> //F

# 然后重新启动
python app.py
```

### 关闭调试模式（生产环境）

默认以 debug 模式启动。如果需要关闭自动重载和多进程：

```bash
# Windows (cmd)
set FLASK_DEBUG=0
python app.py

# Windows (PowerShell)
$env:FLASK_DEBUG=0
python app.py

# Linux / macOS / Git Bash
FLASK_DEBUG=0 python app.py
```

### Python 找不到依赖模块

**症状：** `ModuleNotFoundError: No module named 'anndata'` 或类似错误。

**解决：**
1. 确认已安装依赖：`pip install -r requirements.txt`
2. 如果使用 Conda，确认已激活正确的环境：`conda activate <环境名>`
3. 验证安装：`python -c "import flask, numpy, pandas, scanpy, anndata, sklearn, faiss; print('OK')"`

### AI 自然语言查询不工作

**症状：** 使用 `/ai/query` 时返回错误或无法解析。

**原因：** 未配置 DeepSeek API Key。

**解决：** 配置环境变量后重启服务：
```bash
set DEEPSEEK_API_KEY=你的DeepSeek API Key
python app.py
```
未配置 API Key 时，AI 分析功能会自动退化为本地统计摘要（仅 `/ai/analyze` 可用）。

## 注意事项

- `admin / admin123` 仅适合本地测试，正式部署应修改默认管理员密码。
- 建议设置环境变量 `ANN_SESSION_SECRET`，避免使用默认 Session Secret。
- 上传大型 `.h5ad` 时，向量导出和索引构建可能耗时较长。
- 可视化接口需要原始 `.h5ad` 中存在 `obsm["X_umap"]` 或 `obsm["X_tsne"]`。
- 当前前端通过 Plotly CDN 加载图表库，离线环境需要改成本地静态文件。
- 项目仓库不包含数据文件和索引文件（`data/`、`results/`、`indices/`、`*.index`），克隆后需按快速启动流程操作。
- 已有的 `results/` 和 `indices/` 下产物（如从真实 liver 数据集导出）会被自动识别为只读的 `default` 数据集，不能通过 Web 界面删除。
- 使用 CLI 路径（`data_loader.py` + `index_builder.py`）导出数据时，建议使用相同的数据源文件（`--input`），否则可能导致散点图等依赖 `.h5ad` 源文件的功能异常。
- 测试数据使用 `cell_0000` ~ `cell_0499` 格式的 Cell ID；真实 liver 数据集使用 `AAACCTGAGCAGGTCA-1_2` 等格式。检索时请使用与当前数据集匹配的 ID。

