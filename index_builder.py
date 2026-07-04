import argparse
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

def suggest_nlist(n_vectors: int) -> int:
    """根据数据规模自动估计 IVF 聚类中心数量。"""
    if n_vectors <= 0:
        return 32
    return int(min(4096, max(32, round(np.sqrt(n_vectors) * 4))))


def ensure_faiss_vectors(vectors: np.ndarray) -> np.ndarray:
    """确保向量满足 FAISS 需要的 float32 + C 连续内存。"""
    vectors = np.asarray(vectors, dtype=np.float32)
    return np.ascontiguousarray(vectors)

class AnnIndexBuilder:
    """
    单细胞向量 ANN 索引构建器 (P2模块)
    封装了 FAISS 的 HNSW 和基础 Flat 索引的构建、保存与加载逻辑。
    """
    def __init__(self, dim: int = None, metric: str = 'l2'):
        """
        初始化索引构建器。
        :param dim: 向量维度（如果在 load_index 前未知，可传 None）
        :param metric: 距离度量方式，'l2' 或 'ip' (内积/余弦相似度)
        """
        self.dim = dim
        self.metric = faiss.METRIC_INNER_PRODUCT if metric.lower() == 'ip' else faiss.METRIC_L2
        self.index = None

    def build_hnsw_index(self, vectors: np.ndarray, M: int = 32, efConstruction: int = 200) -> None:
        """
        构建 HNSW (Hierarchical Navigable Small World) 索引
        :param vectors: numpy 向量矩阵 (float32)
        :param M: 每个节点在图中的最大连接数 (控制内存和精度，默认32)
        :param efConstruction: 构建时的邻居候选集大小 (控制构建耗时和精度，默认200)
        """
        if self.dim is None:
            self.dim = vectors.shape[1]
            
        print(f"[*] 开始构建 FAISS HNSW 索引 (M={M}, efConstruction={efConstruction})...")
        start_time = time.time()
        
        # 初始化 HNSW 索引
        self.index = faiss.IndexHNSWFlat(self.dim, M, self.metric)
        self.index.hnsw.efConstruction = efConstruction
        
        # 添加数据并建树
        self.index.add(vectors)
        
        print(f"[+] HNSW 索引构建完成! 耗时: {time.time() - start_time:.3f} 秒")
        print(f"    当前索引包含总向量数: {self.index.ntotal}")

    def build_ivf_hnsw_index(
        self,
        vectors: np.ndarray,
        nlist: Optional[int] = None,
        hnsw_m: int = 32,
        efConstruction: int = 200,
        train_size: int = 20000,
    ) -> None:
        """
        构建 IVF + HNSW 混合索引。

        结构：
        - coarse quantizer 使用 HNSW
        - inverted lists 使用 IVF Flat

        适合中大规模数据集，兼顾速度和召回率。
        """
        vectors = ensure_faiss_vectors(vectors)

        if self.dim is None:
            self.dim = vectors.shape[1]

        if nlist is None:
            nlist = suggest_nlist(len(vectors))

        nlist = max(1, min(int(nlist), int(len(vectors))))

        print(f"[*] 开始构建 IVF+HNSW 索引 (nlist={nlist}, M={hnsw_m}, efConstruction={efConstruction})...")
        start_time = time.time()

        quantizer = faiss.IndexHNSWFlat(self.dim, hnsw_m, self.metric)
        quantizer.hnsw.efConstruction = efConstruction

        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dim,
            int(nlist),
            self.metric,
        )

        if not self.index.is_trained:
            if len(vectors) > train_size:
                sample_idx = np.random.choice(len(vectors), size=train_size, replace=False)
                train_vectors = vectors[sample_idx]
            else:
                train_vectors = vectors

            train_vectors = ensure_faiss_vectors(train_vectors)
            self.index.train(train_vectors)

        self.index.add(vectors)

        # 默认查询参数，后续 search.py 会根据前端滑块动态覆盖
        self.index.nprobe = min(max(1, round(np.sqrt(nlist))), min(int(nlist), 64))

        print(f"[+] IVF+HNSW 索引构建完成! 耗时: {time.time() - start_time:.3f} 秒")
        print(f"    当前索引包含总向量数: {self.index.ntotal}")
        print(f"    默认 nprobe: {self.index.nprobe}")

    def build_flat_index(self, vectors: np.ndarray) -> None:
        """
        构建暴力精确检索索引 (IndexFlatL2/IP)
        作为基线方法 (Baseline)，供系统对比召回率和速度。
        """
        if self.dim is None:
            self.dim = vectors.shape[1]
            
        print("[*] 开始构建暴力检索 Flat 索引...")
        start_time = time.time()
        
        if self.metric == faiss.METRIC_L2:
            self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            
        self.index.add(vectors)
        print(f"[+] Flat 索引构建完成! 耗时: {time.time() - start_time:.3f} 秒")

    def save_index(self, filepath: str) -> None:
        """持久化保存索引到磁盘"""
        if self.index is None:
            raise ValueError("索引为空，请先 build_index 或 load_index")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(filepath))
        print(f"[*] 索引已保存至: {filepath}")

    def load_index(self, filepath: str) -> None:
        """从磁盘加载已有索引"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到索引文件: {filepath}")
            
        print(f"[*] 正在加载索引: {filepath} ...")
        start_time = time.time()
        self.index = faiss.read_index(filepath)
        self.dim = self.index.d
        self.metric = getattr(self.index, "metric_type", self.metric)
        print(f"[+] 索引加载成功! 耗时: {time.time() - start_time:.3f} 秒, 维度: {self.dim}, 总数: {self.index.ntotal}")

    def get_faiss_index(self):
        """提供给 P3 (检索逻辑) 直接调用的底层 FAISS index 对象"""
        return self.index


def build_index_from_vectors(
    vectors_path: str,
    output_path: str,
    index_type: str = "hnsw",
    metric: str = "l2",
    M: int = 32,
    efConstruction: int = 200,
    nlist: Optional[int] = None,
) -> dict[str, object]:
    """从 .npy 向量矩阵构建索引并保存，供 CLI 和数据集管理模块复用。"""
    vectors = ensure_faiss_vectors(np.load(vectors_path))
    if vectors.ndim != 2:
        raise ValueError(f"向量矩阵应为二维，当前 shape={vectors.shape}")

    started = time.time()
    builder = AnnIndexBuilder(dim=vectors.shape[1], metric=metric)
    if index_type == "hnsw":
        builder.build_hnsw_index(vectors, M=M, efConstruction=efConstruction)
    elif index_type == "flat":
        builder.build_flat_index(vectors)
    elif index_type == "ivf_hnsw":
        builder.build_ivf_hnsw_index(
            vectors,
            nlist=nlist,
            hnsw_m=M,
            efConstruction=efConstruction,
        )
    else:
        raise ValueError("index_type 仅支持 hnsw、flat 或 ivf_hnsw")

    builder.save_index(output_path)
    elapsed = time.time() - started
    return {
        "index_path": output_path,
        "index_type": index_type,
        "metric": metric,
        "vectors_path": vectors_path,
        "shape": [int(vectors.shape[0]), int(vectors.shape[1])],
        "build_seconds": round(elapsed, 3),
        "M": int(M),
        "efConstruction": int(efConstruction),
        "nlist": int(getattr(builder.index, "nlist", 0)) if hasattr(builder.index, "nlist") else None,
        "nprobe": int(getattr(builder.index, "nprobe", 0)) if hasattr(builder.index, "nprobe") else None,
    }


def main():
    """
    命令行测试入口 (P2 自己独立测试用)
    """
    parser = argparse.ArgumentParser(description="P2: ANN 索引构建与测试")
    parser.add_argument("--input", required=True, help="P1 提供的 numpy 向量矩阵文件 (.npy)")
    parser.add_argument("--outdir", default="indices", help="索引保存目录")
    parser.add_argument("--type", choices=['hnsw', 'flat', 'ivf_hnsw'], default='hnsw', help="要构建的索引类型")
    parser.add_argument("--M", type=int, default=32, help="HNSW: 最大连接数")
    parser.add_argument("--ef", type=int, default=200, help="HNSW: 构建候选集大小")
    parser.add_argument("--nlist", type=int, default=None, help="IVF: 聚类中心数量，不填则自动估计")
    
    args = parser.parse_args()

    if args.type == 'hnsw':
        save_name = f"hnsw_M{args.M}_ef{args.ef}.index"
    elif args.type == 'ivf_hnsw':
        nlist_tag = args.nlist if args.nlist else "auto"
        save_name = f"ivf_hnsw_nlist{nlist_tag}_M{args.M}_ef{args.ef}.index"
    else:
        save_name = "flat.index"

    save_path = os.path.join(args.outdir, save_name)
    print(f"[*] 正在读取向量数据: {args.input}")
    summary = build_index_from_vectors(
        vectors_path=args.input,
        output_path=save_path,
        index_type=args.type,
        M=args.M,
        efConstruction=args.ef,
        nlist=args.nlist,
    )
    print(f"    向量形状: {tuple(summary['shape'])}")

    builder = AnnIndexBuilder()
    builder.load_index(save_path)


if __name__ == "__main__":
    main()
