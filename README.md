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

## 中期：Flask 后端 API 测试

先确保前面的导出和索引文件已经生成，然后启动 Flask 服务：

```bash
python app.py
```

服务启动后，可以用下面的命令做接口测试：

```bash
# 首页健康检查
curl.exe http://127.0.0.1:5000/

# 按 cell_id 查询
curl.exe -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"cell_id\":\"AAACCTGAGCAGGTCA-1_2\",\"k\":5}"

# 按向量查询（示例向量需替换成实际维度一致的值）
curl.exe -X POST http://127.0.0.1:5000/search ^
  -H "Content-Type: application/json" ^
  -d "{\"vector\":[0.1,0.2,0.3,0.4],\"k\":5}"
```

也可以先用 Flask 的测试客户端做快速烟测：

```bash
python -c "from app import app; client = app.test_client(); print(client.get('/').status_code); print(client.post('/search', json={'cell_id':'AAACCTGAGCAGGTCA-1_2','k':5}).status_code)"
```

## 中期：前端 Web 界面与交互

基于 Flask + HTML/CSS/JavaScript 实现 ANN 检索系统的前端页面，支持用户通过浏览器进行交互式查询，并动态展示检索结果。

运行（项目根目录）：

```bash
python app.py
```

浏览器访问：

```text
http://127.0.0.1:5000
```

功能说明：

```text
1. 支持按 Cell ID 查询
2. 支持按原始向量查询
3. 支持自定义 Top-K 返回数量
4. 动态显示 ANN 检索耗时
5. 表格化展示检索结果与元数据
6. 前后端通过 /search API 完成交互
```

前端目录结构：

```text
templates/
└── index.html

static/
└── style.css
```

页面主要实现内容：

```text
- HTML 页面结构设计
- CSS 页面美化与响应式布局
- JavaScript Fetch API 异步请求
- 查询结果动态渲染
- 错误状态提示与异常处理
- Flask 模板页面集成
```

说明：

运行前需确保已完成前面模块的数据导出与索引构建，并存在以下文件：

```text
results/vectors.npy
results/cell_ids.npy
results/obs_metadata.csv
indices/hnsw_M32_ef200.index
```

否则前端页面无法正常完成 ANN 检索请求。


---
## 数据集管理（上传 / 删除 / 切换）

完成数据集管理模块，新增 `dataset_manager.py`，并在现有 Flask 后端与前端页面中接入上传、删除、切换活动数据集的能力。上传完成后会自动触发向量导出和索引构建：

```text
data_loader.export_h5ad() -> 导出 vectors/cell_ids/metadata/summary
index_builder.build_index_from_vectors() -> 构建 FAISS 索引
```

新增/修改文件：

```text
新增 dataset_manager.py
修改 app.py
修改 data_loader.py
修改 index_builder.py
修改 templates/index.html
修改 static/style.css
补充 README.md
```

默认产物目录：

```text
data/datasets/<dataset_id>/        # 上传的 .h5ad 源文件
results/datasets/<dataset_id>/     # vectors.npy、cell_ids.npy、obs_metadata.csv、summary.json
indices/datasets/<dataset_id>/     # hnsw_M*_ef*.index 或 flat.index
results/datasets/manifest.json     # 数据集清单与当前活动数据集
```

兼容已有默认数据集：

```text
results/vectors.npy
results/cell_ids.npy
results/obs_metadata.csv
indices/hnsw_M32_ef200.index
```

如果上述文件存在，系统会自动登记为只读的 `default` 数据集。默认数据集可以切换使用，但不能通过管理接口删除。

### Web 页面操作

启动服务后访问首页：

```bash
python app.py
```

```text
http://127.0.0.1:5000
```

页面顶部的“数据集管理”区域支持：

```text
1. 查看当前活动数据集
2. 上传 .h5ad 文件并自动导出向量、构建索引
3. 切换活动数据集，后续 /search 自动使用新数据集
4. 删除上传的数据集并清理对应 data/results/indices 文件
```

### 数据集管理 API

列出数据集：

```bash
curl.exe http://127.0.0.1:5000/datasets
```

上传 `.h5ad` 并自动激活：

```bash
curl.exe -X POST http://127.0.0.1:5000/datasets ^
  -F "file=@data/liver.h5ad" ^
  -F "name=liver-demo" ^
  -F "embedding=X_pca" ^
  -F "dims=30" ^
  -F "index_type=hnsw" ^
  -F "M=32" ^
  -F "ef=200" ^
  -F "activate=true"
```

切换活动数据集：

```bash
curl.exe -X POST http://127.0.0.1:5000/datasets/liver-demo/activate
```

删除上传的数据集：

```bash
curl.exe -X DELETE http://127.0.0.1:5000/datasets/liver-demo
```

查看服务状态：

```bash
curl.exe http://127.0.0.1:5000/status
```

活动数据集切换后，`app.py` 会清空当前 `AnnSearcher` 缓存；下一次 `/search` 请求会按新的 `index_path / vectors_path / metadata_path / cell_ids_path` 重新加载。


这是按照你提供的模板格式，为你梳理和缩减的两个功能模块的实现记录：

## 条件检索（元数据过滤与混合策略）

完成条件检索模块，在现有 `AnnSearcher` 中接入按细胞类型/疾病等元数据条件过滤后再返回 Top-K 的能力。采用动态混合过滤策略，以保证不同数据规模下的检索精度与性能：

```text
接收带 filters 的检索请求 -> 判断过滤后的候选细胞数量
若候选数量 < 1000 -> 预过滤：构建 FAISS IDMap 子集进行精确搜索
若候选数量 ≥ 1000 -> 后过滤：先扩大搜索范围至 max(k*3, 200)，再按条件筛选补齐至 K 个

```

新增/修改文件：

```text
修改 search.py
修改 app.py

```

返回数据结构更新：

```text
响应 JSON 新增 filter_info 字段，包含：
- filtered_count: 过滤后的细胞总数
- strategy: 当前触发的策略 (pre_filter / post_filter)
- filters: 当前生效的过滤条件

```

### 检索 API (支持条件过滤)

带条件检索（通过 `filters` 字段传入元数据条件）：

```bash
curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "cell_id": "AAACCTGAGCAGGTCA-1_2",
    "k": 10,
    "filters": {"disease": "healthy", "cell_type": "hepatocyte"}
  }'

```

---

## 交互式可视化数据接口

主要实现单细胞数据的二维可视化接口，为前端散点图展示和点击查询提供后端支持。

### 主要功能

1. 新增 `visualize.py`，用于读取当前活动数据集中的 `.h5ad` 文件。
2. 优先读取 `adata.obsm["X_umap"]` 作为二维坐标；如果不存在，则尝试读取 `adata.obsm["X_tsne"]`。
3. 将细胞 ID、二维坐标、细胞类型和元数据整合为 JSON，供前端绘制散点图。
4. 支持前端点击散点图中的细胞后，通过 `cell_id` 反向查询该细胞的 Top-K 近邻。

### 涉及文件

```text
新增：visualize.py
修改：app.py
```

### 新增接口

#### 获取散点图数据

```text
GET /scatter_data
```

可选参数：

```text
max_points：限制返回点数，用于前端降采样显示
fields：指定返回的元数据字段
```

请求示例：

```text
http://127.0.0.1:5000/scatter_data?max_points=5
```

返回内容包括：

```text
basis：使用的坐标类型，如 X_umap 或 X_tsne
dataset：当前数据集信息
points：散点图数据点
returned：实际返回点数
total：数据集总细胞数
```

单个点的数据格式示例：

```json
{
  "cell_id": "cell_0044",
  "cell_type": "NK-cell",
  "x": 21.4587,
  "y": -2.2080,
  "metadata": {
    "cell_type": "NK-cell",
    "disease": "HCC",
    "AgeGroup": "Senior"
  }
}
```

#### 点击细胞反向查询

```text
POST /scatter_search
```

请求示例：

```json
{
  "cell_id": "cell_0044",
  "k": 10
}
```

该接口复用已有的 `AnnSearcher.search_by_cell_id()` 方法，根据点击的细胞 ID 返回 Top-K 近邻结果。

返回结果包括：

```text
rank：排名
cell_id：近邻细胞 ID
distance：距离
metadata：近邻细胞元数据
```

### 测试结果

已使用测试数据集 `test_data` 完成功能验证：

```text
1. /scatter_data?max_points=5 可以正常返回 X_umap 坐标数据；
2. 返回数据包含 cell_id、x、y、cell_type 和 metadata；
3. /scatter_search 可以根据 cell_id 返回 Top-K 近邻；
4. 查询 cell_0044 时，返回 results: Array(10)，且 rank=1 为自身，distance=0。
```

说明交互式可视化数据接口已完成，可以为前端散点图渲染和点击查询功能提供支持。

## 用户认证与权限管理系统

完成完整的用户认证与授权系统，新增 `user_manager.py`，并在现有 Flask 后端与前端页面中接入用户注册、登录、会话管理及管理员功能。所有核心业务路由均添加了强制登录限制（`@login_required`）：

```text
用户注册/登录 -> PBKDF2-SHA256 密码哈希校验 -> 签发 HTTPOnly Session
访问受保护路由 -> 鉴权装饰器校验 Session 与 Role -> 渲染页面或执行 API

```

新增/修改文件：

```text
新增 user_manager.py
新增 templates/login.html
新增 templates/register.html
新增 templates/admin.html
新增 static/auth.css
修改 app.py
修改 templates/index.html

```

默认产物目录：

```text
data/users.json    # 用户数据持久化存储文件 (可通过 ANN_USERS_PATH 环境变量自定义)

```

兼容与安全策略：

```text
首次启动会自动在 data/users.json 中创建默认管理员账号 (admin / admin123)。
在执行删除用户或降级角色操作时，系统会自动拦截并保护“最后一个管理员”，防止出现权限锁定情况。

```

### Web 页面操作

启动服务后访问相关路由：

```text
1. 访问 /login 或 /register 进行登录与注册，支持前端校验与异步错误反馈。
2. 登录成功后，首页顶部展示用户信息栏，包含当前角色标签与退出登录按钮。
3. 管理员可进入 /admin 面板，支持：
   - 浏览所有注册用户信息及活跃状态
   - 动态提升普通用户为管理员或降级
   - 强制重置任意用户的密码
   - 删除违规或闲置用户

```

### 认证与管理 API

用户注册（公开）：

```bash
curl -X POST http://127.0.0.1:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass"}'

```

用户登录（公开，登录后客户端需保存返回的 Cookie）：

```bash
curl -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "password": "testpass"}'

```

获取当前登录用户信息（需登录）：

```bash
curl http://127.0.0.1:5000/api/auth/me

```

修改用户角色（需管理员权限）：

```bash
curl -X POST http://127.0.0.1:5000/api/admin/users/testuser/role \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'

```
