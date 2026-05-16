"""
ODVS Dataset Registry — Central metadata registry for all managed datasets.

The registry is a flat JSON store (suitable for local dev and small deployments).
In production, this would be backed by a PostgreSQL catalog or a REST catalog service
(Apache Polaris, Project Nessie, etc.).

Schema:
{
  "datasets": {
    "<dataset_name>": {
      "name": str,
      "description": str,
      "tags": List[str],
      "created_at": ISO8601,
      "updated_at": ISO8601,
      "versions": [
        {
          "version_tag": str,
          "snapshot_id": int | null,
          "row_count": int,
          "created_at": ISO8601,
          "checksum": str,
          "lineage_node_id": str | null,
        }
      ],
      "latest_version": str,
      "schema": {col: dtype, ...},
      "source_uri": str,
      "table_path": str,
      "extra": dict
    }
  }
}
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from odvs.config import RegistryConfig, get_config
from odvs.logger import get_logger

logger = get_logger(__name__)

_LOCK = threading.RLock()


class DatasetNotFoundError(KeyError):
    """Raised when a dataset is not found in the registry."""


class VersionNotFoundError(KeyError):
    """Raised when a version is not found in a dataset's version history."""


class DatasetRegistry:
    """
    Thread-safe local dataset registry.

    The registry file is lazily created on first write. All operations
    use a reentrant lock to ensure consistency in multi-threaded contexts.
    """

    def __init__(self, config: Optional[RegistryConfig] = None) -> None:
        self._cfg = config or get_config().registry
        self._path = self._cfg.registry_filepath
        self._path.parent.mkdir(parents=True, exist_ok=True)


    def register(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        source_uri: str = "",
        table_path: str = "",
        schema: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a new dataset or update an existing one's metadata.
        Does NOT overwrite version history.
        """
        with _LOCK:
            registry = self._load()
            now = datetime.now(timezone.utc).isoformat()

            if name in registry["datasets"]:
                existing = registry["datasets"][name]
                existing["description"] = description or existing.get("description", "")
                existing["tags"] = tags or existing.get("tags", [])
                existing["updated_at"] = now
                existing["source_uri"] = source_uri or existing.get("source_uri", "")
                existing["table_path"] = table_path or existing.get("table_path", "")
                if schema:
                    existing["schema"] = schema
                if extra:
                    existing["extra"] = {**existing.get("extra", {}), **extra}
                registry["datasets"][name] = existing
            else:
                registry["datasets"][name] = {
                    "name": name,
                    "description": description,
                    "tags": tags or [],
                    "created_at": now,
                    "updated_at": now,
                    "versions": [],
                    "latest_version": None,
                    "schema": schema or {},
                    "source_uri": source_uri,
                    "table_path": table_path,
                    "extra": extra or {},
                }

            self._save(registry)
            logger.info(f"Registered dataset: {name}")
            return registry["datasets"][name]

    def add_version(
        self,
        dataset_name: str,
        version_tag: str,
        row_count: int,
        snapshot_id: Optional[int] = None,
        checksum: str = "",
        schema: Optional[Dict[str, str]] = None,
        lineage_node_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Append a new version record to an existing dataset.
        Creates the dataset registration if it doesn't exist.
        """
        with _LOCK:
            registry = self._load()

            if dataset_name not in registry["datasets"]:
                logger.warning(f"Dataset '{dataset_name}' not pre-registered — auto-registering")
                self.register(name=dataset_name)
                registry = self._load()

            version_record = {
                "version_tag": version_tag,
                "snapshot_id": snapshot_id,
                "row_count": row_count,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checksum": checksum,
                "lineage_node_id": lineage_node_id,
                "extra": extra or {},
            }

            dataset = registry["datasets"][dataset_name]
            dataset["versions"].append(version_record)
            dataset["latest_version"] = version_tag
            dataset["updated_at"] = datetime.now(timezone.utc).isoformat()
            if schema:
                dataset["schema"] = schema

            self._save(registry)
            logger.info(f"Added version {version_tag} to dataset {dataset_name} (rows={row_count:,})")
            return version_record

    def get_dataset(self, name: str) -> Dict[str, Any]:
        registry = self._load()
        if name not in registry["datasets"]:
            raise DatasetNotFoundError(f"Dataset not found in registry: '{name}'")
        return registry["datasets"][name]

    def get_version(self, dataset_name: str, version_tag: str) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_name)
        for version in dataset["versions"]:
            if version["version_tag"] == version_tag:
                return version
        raise VersionNotFoundError(
            f"Version '{version_tag}' not found in dataset '{dataset_name}'"
        )

    def get_latest_version(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        dataset = self.get_dataset(dataset_name)
        if not dataset["versions"]:
            return None
        return dataset["versions"][-1]

    def list_datasets(self) -> List[str]:
        return list(self._load()["datasets"].keys())

    def list_versions(self, dataset_name: str) -> List[str]:
        return [v["version_tag"] for v in self.get_dataset(dataset_name)["versions"]]

    def delete_dataset(self, dataset_name: str) -> None:
        with _LOCK:
            registry = self._load()
            if dataset_name not in registry["datasets"]:
                raise DatasetNotFoundError(dataset_name)
            del registry["datasets"][dataset_name]
            self._save(registry)
            logger.warning(f"Deleted dataset from registry: {dataset_name}")

    def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search datasets by name/description substring or tags.
        Case-insensitive matching.
        """
        registry = self._load()
        results = []

        for dataset in registry["datasets"].values():
            if query:
                if (
                    query.lower() not in dataset["name"].lower()
                    and query.lower() not in dataset.get("description", "").lower()
                ):
                    continue
            if tags:
                dataset_tags = set(dataset.get("tags", []))
                if not set(tags).issubset(dataset_tags):
                    continue
            results.append(dataset)

        return results

    def get_all_tags(self) -> List[str]:
        registry = self._load()
        all_tags: set = set()
        for ds in registry["datasets"].values():
            all_tags.update(ds.get("tags", []))
        return sorted(all_tags)

    def get_stats(self) -> Dict[str, Any]:
        registry = self._load()
        datasets = registry["datasets"]
        total_versions = sum(len(ds["versions"]) for ds in datasets.values())
        total_rows = sum(
            sum(v["row_count"] for v in ds["versions"])
            for ds in datasets.values()
        )
        return {
            "total_datasets": len(datasets),
            "total_versions": total_versions,
            "total_rows_across_versions": total_rows,
        }


    def _load(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {"datasets": {}, "_meta": {"version": "1.0"}}
        with self._path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, registry: Dict[str, Any]) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)
        tmp_path.replace(self._path)