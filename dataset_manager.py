from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from index_builder import build_index_from_vectors


DEFAULT_DATASET_ID = "default"


@dataclass(frozen=True)
class DatasetPaths:
    source: Path
    artifacts: Path
    index: Path


class DatasetManager:
    """管理上传数据集、导出产物、索引产物和当前活动数据集。"""

    def __init__(
        self,
        data_root: str | Path = "data/datasets",
        results_root: str | Path = "results/datasets",
        indices_root: str | Path = "indices/datasets",
        legacy_vectors: str | Path = "results/vectors.npy",
        legacy_metadata: str | Path = "results/obs_metadata.csv",
        legacy_cell_ids: str | Path = "results/cell_ids.npy",
        legacy_index: str | Path = "indices/hnsw_M32_ef200.index",
    ) -> None:
        self.data_root = Path(data_root)
        self.results_root = Path(results_root)
        self.indices_root = Path(indices_root)
        self.manifest_path = self.results_root / "manifest.json"
        self.legacy_paths = {
            "vectors": Path(legacy_vectors),
            "metadata": Path(legacy_metadata),
            "cell_ids": Path(legacy_cell_ids),
            "index": Path(legacy_index),
        }

    def list_datasets(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        return {
            "active_dataset_id": manifest.get("active_dataset_id"),
            "datasets": list(manifest["datasets"].values()),
        }

    def get_active_dataset(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        dataset_id = manifest.get("active_dataset_id")
        if not dataset_id or dataset_id not in manifest["datasets"]:
            raise FileNotFoundError("当前没有可用的数据集，请先上传或生成默认数据产物")
        return manifest["datasets"][dataset_id]

    def activate_dataset(self, dataset_id: str) -> dict[str, Any]:
        manifest = self._load_manifest()
        if dataset_id not in manifest["datasets"]:
            raise KeyError(f"数据集不存在: {dataset_id}")
        manifest["active_dataset_id"] = dataset_id
        self._write_manifest(manifest)
        return manifest["datasets"][dataset_id]

    def delete_dataset(self, dataset_id: str) -> dict[str, Any]:
        if dataset_id == DEFAULT_DATASET_ID:
            raise ValueError("默认数据集来自兼容路径，不能通过管理接口删除")

        manifest = self._load_manifest()
        dataset = manifest["datasets"].get(dataset_id)
        if dataset is None:
            raise KeyError(f"数据集不存在: {dataset_id}")

        for root in (self.data_root, self.results_root, self.indices_root):
            target = root / dataset_id
            self._delete_tree_inside_root(target, root)

        del manifest["datasets"][dataset_id]
        if manifest.get("active_dataset_id") == dataset_id:
            manifest["active_dataset_id"] = (
                DEFAULT_DATASET_ID if DEFAULT_DATASET_ID in manifest["datasets"] else None
            )
        self._write_manifest(manifest)
        return {"deleted_dataset_id": dataset_id, "active_dataset_id": manifest["active_dataset_id"]}

    def create_from_upload(
        self,
        uploaded_file: FileStorage,
        name: str | None = None,
        embedding: str = "X_pca",
        dims: int = 30,
        obs_cols: str = "cell_type,disease,AgeGroup,sex,Treatment,Phase,seurat_clusters,donor_age",
        l2: bool = True,
        index_type: str = "hnsw",
        metric: str = "l2",
        M: int = 32,
        ef: int = 200,
        nlist: int | None = None,
        activate: bool = True,
    ) -> dict[str, Any]:
        if not uploaded_file or not uploaded_file.filename:
            raise ValueError("请上传 .h5ad 文件")
        if not uploaded_file.filename.lower().endswith(".h5ad"):
            raise ValueError("仅支持上传 .h5ad 文件")

        display_name = (name or Path(uploaded_file.filename).stem).strip()
        if not display_name:
            display_name = "dataset"
        dataset_id = self._unique_dataset_id(display_name)
        paths = self._paths_for(dataset_id)

        for path in (paths.source, paths.artifacts, paths.index):
            path.mkdir(parents=True, exist_ok=True)

        filename = secure_filename(uploaded_file.filename) or f"{dataset_id}.h5ad"
        source_path = paths.source / filename
        uploaded_file.save(str(source_path))

        from data_loader import export_h5ad

        export_summary = export_h5ad(
            input_path=source_path,
            outdir=paths.artifacts,
            embedding=embedding,
            dims=dims,
            obs_cols=obs_cols,
            l2=l2,
        )

        if index_type == "hnsw":
            index_name = f"hnsw_M{M}_ef{ef}.index"
        elif index_type == "ivf_hnsw":
            nlist_tag = nlist if nlist else "auto"
            index_name = f"ivf_hnsw_nlist{nlist_tag}_M{M}_ef{ef}.index"
        else:
            index_name = "flat.index"
        index_path = paths.index / index_name
        index_summary = build_index_from_vectors(
            vectors_path=str(paths.artifacts / "vectors.npy"),
            output_path=str(index_path),
            index_type=index_type,
            metric=metric,
            M=M,
            efConstruction=ef,
            nlist=nlist,
        )

        manifest = self._load_manifest()
        dataset = {
            "id": dataset_id,
            "name": display_name,
            "source_path": str(source_path),
            "vectors_path": str(paths.artifacts / "vectors.npy"),
            "metadata_path": str(paths.artifacts / "obs_metadata.csv"),
            "cell_ids_path": str(paths.artifacts / "cell_ids.npy"),
            "summary_path": str(paths.artifacts / "summary.json"),
            "index_path": str(index_path),
            "index_type": index_type,
            "metric": metric,
            "embedding": export_summary.get("embedding"),
            "n_obs": export_summary.get("n_obs"),
            "n_vars": export_summary.get("n_vars"),
            "metadata_cols": export_summary.get("export", {}).get("metadata_cols", []),
            "created_at": int(time.time()),
            "index_build": index_summary,
        }
        manifest["datasets"][dataset_id] = dataset
        if activate:
            manifest["active_dataset_id"] = dataset_id
        self._write_manifest(manifest)
        return dataset

    def _paths_for(self, dataset_id: str) -> DatasetPaths:
        return DatasetPaths(
            source=self.data_root / dataset_id,
            artifacts=self.results_root / dataset_id,
            index=self.indices_root / dataset_id,
        )

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {"active_dataset_id": None, "datasets": {}}

        if self._legacy_ready():
            manifest["datasets"].setdefault(
                DEFAULT_DATASET_ID,
                {
                    "id": DEFAULT_DATASET_ID,
                    "name": "默认 liver 数据集",
                    "source_path": "data/liver.h5ad",
                    "vectors_path": str(self.legacy_paths["vectors"]),
                    "metadata_path": str(self.legacy_paths["metadata"]),
                    "cell_ids_path": str(self.legacy_paths["cell_ids"]),
                    "summary_path": "results/summary.json",
                    "index_path": str(self.legacy_paths["index"]),
                    "index_type": "hnsw",
                    "metric": "l2",
                    "created_at": None,
                    "readonly": True,
                },
            )

        active_id = manifest.get("active_dataset_id")
        if not active_id or active_id not in manifest["datasets"]:
            manifest["active_dataset_id"] = (
                DEFAULT_DATASET_ID if DEFAULT_DATASET_ID in manifest["datasets"] else None
            )
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _legacy_ready(self) -> bool:
        return all(path.exists() for path in self.legacy_paths.values())

    def _unique_dataset_id(self, name: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.lower()).strip("-") or "dataset"
        manifest = self._load_manifest()
        dataset_id = base
        suffix = 2
        while dataset_id in manifest["datasets"]:
            dataset_id = f"{base}-{suffix}"
            suffix += 1
        return dataset_id

    @staticmethod
    def _delete_tree_inside_root(target: Path, root: Path) -> None:
        resolved_root = root.resolve()
        resolved_target = target.resolve()
        if resolved_root == resolved_target or resolved_root not in resolved_target.parents:
            raise ValueError(f"拒绝删除非数据集目录: {target}")
        if resolved_target.exists():
            shutil.rmtree(resolved_target)
