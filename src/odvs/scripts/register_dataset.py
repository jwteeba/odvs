#!/usr/bin/env python3
"""
ODVS register_dataset.py — Dataset registry management CLI.

Query, update, and inspect the ODVS dataset registry without running
a full ingestion pipeline. Useful for:
- Inspecting registered datasets and their version history
- Manually registering external datasets
- Generating HF Hub dataset cards
- Viewing lineage graphs
- Exporting registry metadata

Usage:
    # List all datasets:
    python scripts/register_dataset.py list

    # Show dataset info:
    python scripts/register_dataset.py info --dataset ecommerce_events

    # Show version history:
    python scripts/register_dataset.py versions --dataset ecommerce_events

    # Show lineage:
    python scripts/register_dataset.py lineage --dataset ecommerce_events

    # Generate dataset card (HF Hub style):
    python scripts/register_dataset.py card --dataset ecommerce_events

    # Manually register a dataset:
    python scripts/register_dataset.py register \\
        --dataset my_dataset \\
        --description "My custom dataset" \\
        --tags nlp,text \\
        --source-uri s3://my-bucket/data.parquet

    # Export registry to JSON:
    python scripts/register_dataset.py export --output registry_export.json

    # Search by tag:
    python scripts/register_dataset.py search --tag ecommerce

    # Registry stats:
    python scripts/register_dataset.py stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from odvs.analytics.lineage import LineageTracker
from odvs.hf.simulator import HFSimulator
from odvs.logger import configure_root_logger, get_logger
from odvs.registry.dataset_registry import DatasetNotFoundError, DatasetRegistry

configure_root_logger()
logger = get_logger(__name__)


def cmd_list(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    datasets = registry.list_datasets()
    if not datasets:
        print("No datasets registered.")
        return 0

    print(f"\n  {'Dataset':<40} {'Latest Version':<16} {'Versions'}")
    print(f"  {'─'*40} {'─'*16} {'─'*8}")
    for name in sorted(datasets):
        try:
            ds = registry.get_dataset(name)
            latest = ds.get("latest_version") or "—"
            version_count = len(ds.get("versions", []))
            print(f"  {name:<40} {latest:<16} {version_count}")
        except Exception:
            print(f"  {name:<40} [error reading]")

    print(f"\n  {len(datasets)} dataset(s) registered.\n")
    return 0


def cmd_info(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    try:
        ds = registry.get_dataset(args.dataset)
    except DatasetNotFoundError:
        print(f"ERROR: Dataset not found: '{args.dataset}'", file=sys.stderr)
        return 1

    print(f"\n{'═' * 60}")
    print(f"  Dataset: {ds['name']}")
    print(f"{'═' * 60}")
    print(f"  Description : {ds.get('description') or '—'}")
    print(f"  Tags        : {', '.join(ds.get('tags', [])) or '—'}")
    print(f"  Source URI  : {ds.get('source_uri') or '—'}")
    print(f"  Table Path  : {ds.get('table_path') or '—'}")
    print(f"  Created     : {ds.get('created_at', '')[:19]}")
    print(f"  Updated     : {ds.get('updated_at', '')[:19]}")
    print(f"  Latest Ver  : {ds.get('latest_version') or '—'}")
    print(f"  Versions    : {len(ds.get('versions', []))}")

    schema = ds.get("schema", {})
    if schema:
        print(f"\n  Schema ({len(schema)} columns):")
        for col, dtype in schema.items():
            print(f"    {col:<35} {dtype}")

    extra = ds.get("extra", {})
    if extra:
        print(f"\n  Extra Metadata:")
        for k, v in extra.items():
            print(f"    {k}: {v}")

    print(f"{'═' * 60}\n")
    return 0


def cmd_versions(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    try:
        versions = registry.list_versions(args.dataset)
        ds = registry.get_dataset(args.dataset)
    except DatasetNotFoundError:
        print(f"ERROR: Dataset not found: '{args.dataset}'", file=sys.stderr)
        return 1

    if not versions:
        print(f"No versions registered for dataset: {args.dataset}")
        return 0

    print(f"\n  Versions for: {args.dataset}")
    print(f"  {'─'*70}")
    print(f"  {'Version':<16} {'Snapshot ID':<20} {'Rows':>10} {'Created':<22} {'Checksum'}")
    print(f"  {'─'*70}")
    for v in ds["versions"]:
        snap = str(v.get("snapshot_id") or "—")
        rows = f"{v.get('row_count', 0):,}"
        created = v.get("created_at", "")[:19]
        checksum = v.get("checksum", "—")[:12]
        tag = v["version_tag"]
        latest_marker = " ← latest" if tag == ds.get("latest_version") else ""
        print(f"  {tag:<16} {snap:<20} {rows:>10} {created:<22} {checksum}{latest_marker}")
    print(f"  {'─'*70}\n")
    return 0


def cmd_lineage(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    tracker = LineageTracker()
    provenance = tracker.get_full_provenance(args.dataset)

    if not provenance["lineage_nodes"]:
        print(f"No lineage recorded for dataset: {args.dataset}")
        return 0

    print(f"\n  Lineage: {args.dataset}")
    print(f"  Versions: {provenance['version_count']}")
    print(f"  First seen: {provenance['first_seen']}")
    print(f"  Last seen: {provenance['last_seen']}")
    print(f"\n  All source URIs:")
    for uri in provenance["all_source_uris"]:
        print(f"    → {uri}")

    print(f"\n  Transforms applied (across all versions):")
    for t in provenance["all_transforms"]:
        print(f"    • {t}")

    print(f"\n  Version chain:")
    for node in provenance["lineage_nodes"]:
        parents = ", ".join(node["parent_node_ids"]) or "root"
        print(f"    [{node['version_tag']}] node={node['node_id'][:8]}… parents=[{parents[:30]}]")
        print(f"           rows={node['row_count']:,} | snapshot={node.get('snapshot_id') or '—'}")

    if args.json:
        print(f"\n  JSON:")
        print(json.dumps(provenance, indent=2, default=str))

    return 0


def cmd_card(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    """Generate a Hugging Face-style dataset card."""
    hf = HFSimulator(registry=registry)

    try:
        card = hf.generate_dataset_card(
            dataset_name=args.dataset,
            language=args.language,
            license=args.license,
        )
    except DatasetNotFoundError:
        print(f"ERROR: Dataset not found: '{args.dataset}'", file=sys.stderr)
        return 1

    markdown = card.to_markdown()

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"✓ Dataset card written to: {args.output}")
    else:
        print(markdown)

    return 0


def cmd_register(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    ds = registry.register(
        name=args.dataset,
        description=args.description or "",
        tags=tags,
        source_uri=args.source_uri or "",
        table_path=args.table_path or "",
    )
    print(f"✓ Registered dataset: {ds['name']}")
    return 0


def cmd_search(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    tags = [t.strip() for t in args.tag.split(",")] if args.tag else None
    results = registry.search(query=args.query or "", tags=tags)

    if not results:
        print("No matching datasets found.")
        return 0

    print(f"\n  Search results ({len(results)}):")
    for ds in results:
        tags_str = ", ".join(ds.get("tags", [])) or "—"
        print(f"  • {ds['name']:<40} tags: {tags_str}")
    print()
    return 0


def cmd_export(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    all_datasets = {}
    for name in registry.list_datasets():
        try:
            all_datasets[name] = registry.get_dataset(name)
        except Exception as exc:
            logger.warning(f"Could not export dataset {name}: {exc}")

    export = {
        "odvs_registry_export": True,
        "exported_at": __import__("datetime").datetime.utcnow().isoformat(),
        "dataset_count": len(all_datasets),
        "datasets": all_datasets,
    }

    output_str = json.dumps(export, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output_str, encoding="utf-8")
        print(f"✓ Registry exported to: {args.output} ({len(all_datasets)} datasets)")
    else:
        print(output_str)

    return 0


def cmd_stats(args: argparse.Namespace, registry: DatasetRegistry) -> int:
    stats = registry.get_stats()
    tags = registry.get_all_tags()

    print(f"\n  ODVS Registry Statistics")
    print(f"  {'─'*35}")
    print(f"  Total datasets        : {stats['total_datasets']}")
    print(f"  Total versions        : {stats['total_versions']}")
    print(f"  Total rows (all vers) : {stats['total_rows_across_versions']:,}")
    print(f"  Unique tags           : {len(tags)}")
    if tags:
        print(f"  Tags                  : {', '.join(tags)}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ODVS — Dataset Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered datasets")

    # info
    p_info = sub.add_parser("info", help="Show dataset metadata")
    p_info.add_argument("--dataset", required=True)

    # versions
    p_vers = sub.add_parser("versions", help="Show version history for a dataset")
    p_vers.add_argument("--dataset", required=True)

    # lineage
    p_lin = sub.add_parser("lineage", help="Show lineage graph for a dataset")
    p_lin.add_argument("--dataset", required=True)
    p_lin.add_argument("--json", action="store_true", help="Also output raw JSON")

    # card
    p_card = sub.add_parser("card", help="Generate a Hugging Face-style dataset card")
    p_card.add_argument("--dataset", required=True)
    p_card.add_argument("--output", help="Write markdown to FILE instead of stdout")
    p_card.add_argument("--language", default=None)
    p_card.add_argument("--license", default="apache-2.0")

    # register
    p_reg = sub.add_parser("register", help="Manually register a dataset")
    p_reg.add_argument("--dataset", required=True)
    p_reg.add_argument("--description", default="")
    p_reg.add_argument("--tags", default="", help="Comma-separated tags")
    p_reg.add_argument("--source-uri", default="")
    p_reg.add_argument("--table-path", default="")

    # search
    p_search = sub.add_parser("search", help="Search datasets by name or tag")
    p_search.add_argument("--query", default="", help="Search query string")
    p_search.add_argument("--tag", default="", help="Comma-separated tag filter")

    # export
    p_export = sub.add_parser("export", help="Export full registry as JSON")
    p_export.add_argument("--output", help="Output file path (stdout if omitted)")

    # stats
    sub.add_parser("stats", help="Show registry statistics")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    registry = DatasetRegistry()

    commands = {
        "list": cmd_list,
        "info": cmd_info,
        "versions": cmd_versions,
        "lineage": cmd_lineage,
        "card": cmd_card,
        "register": cmd_register,
        "search": cmd_search,
        "export": cmd_export,
        "stats": cmd_stats,
    }

    handler = commands.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    return handler(args, registry)


if __name__ == "__main__":
    sys.exit(main())