# 单细胞高维向量 ANN 检索系统

本项目是一个面向单细胞数据的近似最近邻检索系统，支持 `.h5ad` 数据集上传、向量导出、FAISS 索引构建、条件过滤检索、二维散点图可视化、点击细胞反向查询、性能评估报告展示，以及用户认证与管理员权限管理。

历史开发进度已迁移到 [PROGRESS.md](PROGRESS.md)。

## 功能概览

- 数据集管理：管理员上传、切换、删除 `.h5ad` 数据集。
- 向量化导出：读取 `obsm["X_pca"]`，导出 `vectors.npy`、`cell_ids.npy`、`obs_metadata.csv`。
- ANN 检索：基于 FAISS HNSW/Flat 索引，支持按 Cell ID 或原始向量查询 Top-K 近邻。
- 条件过滤：支持按 `cell_type`、`disease`、`AgeGroup` 等元数据字段过滤检索结果。
- 可视化：读取 `obsm["X_umap"]` 或 `obsm["X_tsne"]`，提供散点图数据接口，前端支持点击细胞后反向查询近邻。
- 性能评估：计算 HNSW 相对 Flat Ground Truth 的 Recall@K、平均延迟、P95、QPS、索引大小。
- 用户系统：支持注册、登录、退出、管理员用户管理、角色变更和密码重置。
- AI 辅助分析：基于 DeepSeek 大模型支持自然语言查询细胞、检索结果解释和后续分析建议。

## 目录结构

```text
.
├── app.py                  # Flask Web 服务与 API 路由
├── data_loader.py          # .h5ad 数据读取与向量导出
├── index_builder.py        # FAISS HNSW/Flat 索引构建
├── search.py               # ANN 检索与条件过滤逻辑
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

建议使用已有课程环境或 Conda 环境。核心 Python 依赖包括：

```text
flask
numpy
pandas
scanpy
anndata
scikit-learn
faiss-cpu
werkzeug
```

如果需要手动安装，可参考：

```bash
pip install flask numpy pandas scanpy anndata scikit-learn faiss-cpu werkzeug
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

## 快速启动

### 1. 生成测试数据

测试数据脚本会生成 `data/test_data.h5ad`，并写入：

```text
obsm["X_pca"]   # 用于向量检索
obsm["X_umap"]  # 用于散点图可视化
```

运行：

```bash
python generate_test_h5ad.py
```

### 2. 启动 Web 服务

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

管理员账号用于上传、切换、删除数据集，以及管理用户。

### 3. 上传测试数据集

登录管理员后，在首页“数据集管理”页上传：

```text
data/test_data.h5ad
```

上传后系统会自动完成：

```text
读取 .h5ad -> 导出向量和元数据 -> 构建 FAISS 索引 -> 设置为活动数据集
```

### 4. 使用检索与可视化

切换到“探索”页后，可以：

- 按 Cell ID 查询 Top-K 近邻。
- 粘贴原始向量进行查询。
- 添加元数据过滤条件。
- 查看 UMAP 散点图。
- 点击散点图中的细胞，自动反向查询该细胞的 Top-K 近邻。

测试数据中的 Cell ID 示例：

```text
cell_0000
cell_0044
cell_0100
```

## 命令行流程

如果不通过 Web 上传，也可以手动完成数据导出和索引构建。

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
python index_builder.py --type flat
```

输出：

```text
indices/hnsw_M32_ef200.index
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
python evaluate.py --sample-size 200 --report evaluation_report.json
```

生成报告后，在 Web 页面“性能评估”页查看 Recall、延迟、QPS 和索引大小。

## Web API

以下 API 均运行在：

```text
http://127.0.0.1:5000
```

除注册、登录接口外，其余业务接口需要登录；数据集上传、切换、删除需要管理员权限。

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
  -F "embedding=X_pca" ^
  -F "dims=30" ^
  -F "index_type=hnsw" ^
  -F "M=32" ^
  -F "ef=200" ^
  -F "activate=true"
```

切换活动数据集，需管理员权限：

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

返回结果包含：

```text
query
time_ms
results
filter_info
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
| 注册 / 登录 | 允许 | 允许 | 允许 |
| 查看首页 | 不允许 | 允许 | 允许 |
| 检索 | 不允许 | 允许 | 允许 |
| 查看散点图 | 不允许 | 允许 | 允许 |
| 查看性能报告 | 不允许 | 允许 | 允许 |
| 上传数据集 | 不允许 | 不允许 | 允许 |
| 切换数据集 | 不允许 | 不允许 | 允许 |
| 删除数据集 | 不允许 | 不允许 | 允许 |
| 用户管理 | 不允许 | 不允许 | 允许 |

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
3. 上传 data/test_data.h5ad
4. 切换到“探索”页
5. 输入 cell_0044，Top-K 设置为 10，执行检索
6. 添加过滤条件 disease = HCC 或 cell_type = NK-cell，再次检索
7. 点击散点图中的细胞，检查结果表是否刷新
8. 切换到“性能评估”页，查看 evaluation_report.json 内容
```

### 权限测试

普通用户登录后直接访问以下接口应返回 `403`：

```text
POST /datasets
POST /datasets/<dataset_id>/activate
DELETE /datasets/<dataset_id>
```

管理员访问：

```text
POST /datasets             # 不带文件时返回 400，说明已通过权限检查进入业务校验
POST /datasets/default/activate  # 返回 200
DELETE /datasets/default   # 返回 400，因为 default 数据集受保护
```

### 语法检查

```bash
python -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['app.py','data_loader.py','index_builder.py','search.py','dataset_manager.py','visualize.py','user_manager.py','evaluate.py','generate_test_h5ad.py']]; print('syntax ok')"
```

## 注意事项

- `admin / admin123` 仅适合本地测试，正式部署应修改默认管理员密码。
- 建议设置环境变量 `ANN_SESSION_SECRET`，避免使用默认 Session Secret。
- 上传大型 `.h5ad` 时，向量导出和索引构建可能耗时较长。
- 可视化接口需要原始 `.h5ad` 中存在 `obsm["X_umap"]` 或 `obsm["X_tsne"]`。
- 当前前端通过 Plotly CDN 加载图表库，离线环境需要改成本地静态文件。

