import argparse
import time
import os
from pathlib import Path

import numpy as np
import faiss

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
        print(f"[+] 索引加载成功! 耗时: {time.time() - start_time:.3f} 秒, 维度: {self.dim}, 总数: {self.index.ntotal}")

def get_faiss_index(self):
        """提供给 P3 (检索逻辑) 直接调用的底层 FAISS index 对象"""
        return self.index

