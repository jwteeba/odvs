"""
ODVS Lineage — Dataset lineage tracking and DAG construction.

Records provenance metadata for every write operation:
- source URIs
- transformations applied
- Iceberg snapshot IDs
- schema at write time
- parent-child version relationships

Lineage graph is stored as JSON in the registry and can be visualized
as a directed acyclic graph.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from odvs.config import RegistryConfig, get_config
from odvs.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LineageNode:
    """A single node in the lineage DAG representing one version of a dataset."""
    node_id: str
    dataset_name: str
    version_tag: str
    snapshot_id: Optional[int]
    created_at: str  # ISO 8601
    source_uris: List[str]
    parent_node_ids: List[str]
    transforms_applied: List[str]
    schema_snapshot: Dict[str, str]
    row_count: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        return cls(**data)


class LineageTracker:
    """
    Tracks and persists dataset lineage as a DAG.

    Storage: JSON file per dataset in the registry directory.
    Format: {node_id: LineageNode} flat map (traversal is done in memory).

    Scalability note: For production deployments with thousands of datasets,
    this would be backed by a graph database (Neptune, TigerGraph) or
    a lineage-specific service (DataHub, OpenLineage).
    """

    def __init__(self, config: Optional[RegistryConfig] = None) -> None:
        self._cfg = config or get_config().registry
        self._base_path = Path(self._cfg.registry_path) / "lineage"
        self._base_path.mkdir(parents=True, exist_ok=True)

  

    def record(
        self,
        dataset_name: str,
        version_tag: str,
        source_uris: List[str],
        transforms_applied: List[str],
        schema_snapshot: Dict[str, str],
        row_count: int,
        snapshot_id: Optional[int] = None,
        parent_node_ids: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """
        Record a new lineage node for a dataset version.

        Args:
            dataset_name: Logical dataset name.
            version_tag: Version identifier (e.g. "v1.0.0", snapshot ID as string).
            source_uris: Input data sources for this version.
            transforms_applied: List of transform names applied.
            schema_snapshot: Column → dtype mapping at write time.
            row_count: Number of rows written.
            snapshot_id: Iceberg snapshot ID.
            parent_node_ids: Lineage DAG parent node IDs.
            extra: Arbitrary extra metadata.

        Returns:
            The created LineageNode.
        """
        node = LineageNode(
            node_id=str(uuid.uuid4()),
            dataset_name=dataset_name,
            version_tag=version_tag,
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_uris=source_uris,
            parent_node_ids=parent_node_ids or [],
            transforms_applied=transforms_applied,
            schema_snapshot=schema_snapshot,
            row_count=row_count,
            extra=extra or {},
        )

        self._persist_node(dataset_name, node)
        logger.info(
            f"Lineage recorded: dataset={dataset_name}, version={version_tag}, "
            f"node_id={node.node_id}, parent_count={len(node.parent_node_ids)}"
        )
        return node


    def get_lineage(self, dataset_name: str) -> List[LineageNode]:
        """Return all lineage nodes for a dataset, ordered by creation time."""
        graph = self._load_graph(dataset_name)
        nodes = list(graph.values())
        nodes.sort(key=lambda n: n.created_at)
        return nodes

    def get_node(self, dataset_name: str, node_id: str) -> Optional[LineageNode]:
        graph = self._load_graph(dataset_name)
        return graph.get(node_id)

    def get_latest_node(self, dataset_name: str) -> Optional[LineageNode]:
        nodes = self.get_lineage(dataset_name)
        return nodes[-1] if nodes else None

    def get_ancestors(self, dataset_name: str, node_id: str) -> List[LineageNode]:
        """Walk the DAG upward from node_id and return all ancestors."""
        graph = self._load_graph(dataset_name)
        visited: List[LineageNode] = []
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)
            node = graph.get(current_id)
            if node and node not in visited:
                visited.append(node)
                queue.extend(node.parent_node_ids)

        return visited

    def get_full_provenance(self, dataset_name: str) -> Dict[str, Any]:
        """
        Return the complete provenance report for a dataset:
        all nodes, version chain, source URIs, and transforms.
        """
        nodes = self.get_lineage(dataset_name)

        all_sources: List[str] = []
        all_transforms: List[str] = []
        for n in nodes:
            all_sources.extend(n.source_uris)
            all_transforms.extend(n.transforms_applied)

        return {
            "dataset_name": dataset_name,
            "version_count": len(nodes),
            "first_seen": nodes[0].created_at if nodes else None,
            "last_seen": nodes[-1].created_at if nodes else None,
            "versions": [n.version_tag for n in nodes],
            "all_source_uris": list(dict.fromkeys(all_sources)),  # deduped, ordered
            "all_transforms": list(dict.fromkeys(all_transforms)),
            "lineage_nodes": [n.to_dict() for n in nodes],
        }

    def list_datasets(self) -> List[str]:
        """Return all dataset names that have lineage records."""
        return [p.stem for p in self._base_path.glob("*.json")]


    def _lineage_path(self, dataset_name: str) -> Path:
        safe_name = dataset_name.replace("/", "__")
        return self._base_path / f"{safe_name}.json"

    def _load_graph(self, dataset_name: str) -> Dict[str, LineageNode]:
        path = self._lineage_path(dataset_name)
        if not path.exists():
            return {}
        with path.open("r") as f:
            raw = json.load(f)
        return {node_id: LineageNode.from_dict(node_data) for node_id, node_data in raw.items()}

    def _persist_node(self, dataset_name: str, node: LineageNode) -> None:
        graph = self._load_graph(dataset_name)
        graph[node.node_id] = node
        path = self._lineage_path(dataset_name)
        with path.open("w") as f:
            json.dump({nid: n.to_dict() for nid, n in graph.items()}, f, indent=2, default=str)