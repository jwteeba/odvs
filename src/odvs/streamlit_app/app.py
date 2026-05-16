"""
ODVS Streamlit Dataset Explorer

Streamlit UI for exploring, versioning, and monitoring
datasets managed by the ODVS platform.

Run: streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from odvs.analytics.diff_engine import DiffEngine
from odvs.analytics.lineage import LineageTracker
from odvs.config import get_config
from odvs.hf.simulator import HFSimulator
from odvs.logger import get_logger
from odvs.processing.ingestion.ingest import DatasetIngester
from odvs.processing.optimization.compression import CompressionBenchmarker
from odvs.processing.transforms.normalize import TransformPipeline
from odvs.processing.deduplication.hash_dedup import HashDeduplicator
from odvs.registry.dataset_registry import DatasetRegistry, DatasetNotFoundError

logger = get_logger(__name__)


st.set_page_config(
    page_title="ODVS · Dataset Explorer",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/odvs/odvs",
        "Report a bug": "https://github.com/odvs/odvs/issues",
        "About": "ODVS — Open Dataset Versioning System. Apache Iceberg + Spark + S3.",
    },
)


st.markdown("""
<style>
  /* ── Font imports ─────────────────────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* ── Root variables ───────────────────────────────────────────── */
  :root {
    --bg-primary:      #0d0f14;
    --bg-secondary:    #131720;
    --bg-card:         #181d28;
    --bg-card-hover:   #1e2535;
    --border:          #252d3d;
    --border-accent:   #2a7aff;
    --text-primary:    #e8ecf4;
    --text-secondary:  #7a8599;
    --text-muted:      #4a5568;
    --accent-blue:     #2a7aff;
    --accent-cyan:     #00d4e8;
    --accent-green:    #00c98d;
    --accent-orange:   #ff7a2a;
    --accent-red:      #ff4d6d;
    --accent-purple:   #9d5cff;
    --font-mono:       'IBM Plex Mono', monospace;
    --font-sans:       'IBM Plex Sans', sans-serif;
  }

  /* ── Global overrides ─────────────────────────────────────────── */
  html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
  }

  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
  }

  /* ── Sidebar ──────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
  }

  [data-testid="stSidebar"] .css-1d391kg {
    padding-top: 1rem;
  }

  /* ── Headers ──────────────────────────────────────────────────── */
  h1, h2, h3 {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }

  /* ── Metric cards ─────────────────────────────────────────────── */
  [data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
  }

  [data-testid="metric-container"] > div {
    font-family: var(--font-sans) !important;
  }

  [data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
  }

  [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
  }

  /* ── Dataframes ───────────────────────────────────────────────── */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
  }

  /* ── Select boxes ─────────────────────────────────────────────── */
  [data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
  }

  /* ── Buttons ──────────────────────────────────────────────────── */
  [data-testid="baseButton-primary"] {
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
  }

  [data-testid="baseButton-secondary"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
  }

  /* ── Expanders ────────────────────────────────────────────────── */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
  }

  /* ── Code blocks ──────────────────────────────────────────────── */
  code, pre {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    background: #0a0c10 !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
  }

  /* ── Tabs ─────────────────────────────────────────────────────── */
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
  }

  [data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
  }

  /* ── Dividers ─────────────────────────────────────────────────── */
  hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
  }

  /* ── Info / warning / error boxes ────────────────────────────── */
  [data-testid="stAlert"] {
    border-radius: 6px !important;
    border-left-width: 3px !important;
    font-size: 0.85rem !important;
  }

  /* ── Custom component classes ─────────────────────────────────── */
  .odvs-logo {
    font-family: var(--font-mono);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent-blue);
    letter-spacing: 0.15em;
  }

  .odvs-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }

  .badge-version {
    background: #1a2a4a;
    color: var(--accent-blue);
    border: 1px solid #2a3a5a;
  }

  .badge-tag {
    background: #1a2a2a;
    color: var(--accent-cyan);
    border: 1px solid #1a3a3a;
  }

  .badge-ok {
    background: #0a2a1a;
    color: var(--accent-green);
    border: 1px solid #0a3a2a;
  }

  .badge-warn {
    background: #2a1a0a;
    color: var(--accent-orange);
    border: 1px solid #3a2a0a;
  }

  .badge-error {
    background: #2a0a10;
    color: var(--accent-red);
    border: 1px solid #3a0a18;
  }

  .schema-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.8rem;
  }

  .schema-col-name { color: var(--text-primary); }
  .schema-col-type { color: var(--accent-cyan); }

  .version-row {
    padding: 0.6rem 0.75rem;
    border-left: 2px solid var(--border);
    margin-bottom: 0.5rem;
    background: var(--bg-card);
    border-radius: 0 6px 6px 0;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    transition: border-color 0.15s;
  }

  .version-row:hover { border-left-color: var(--accent-blue); }
  .version-row.latest { border-left-color: var(--accent-green); }

  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.35rem 0;
    font-size: 0.82rem;
    border-bottom: 1px solid var(--border);
  }

  .stat-label { color: var(--text-secondary); }
  .stat-value { color: var(--text-primary); font-family: var(--font-mono); }

  .section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin: 1.2rem 0 0.5rem 0;
    font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)



# Singletons (cached across sessions)

def get_registry() -> DatasetRegistry:
    return DatasetRegistry()


@st.cache_resource
def get_lineage_tracker() -> LineageTracker:
    return LineageTracker()


def get_hf_simulator() -> HFSimulator:
    return HFSimulator(registry=get_registry())


@st.cache_resource
def get_diff_engine() -> DiffEngine:
    return DiffEngine()


@st.cache_resource
def get_config_cached():
    return get_config()


# Data helpers

@st.cache_data(ttl=5)
def fetch_all_datasets() -> List[Dict[str, Any]]:
    registry = get_registry()
    results = []
    for name in registry.list_datasets():
        try:
            results.append(registry.get_dataset(name))
        except Exception:
            pass
    return results


@st.cache_data(ttl=5)
def fetch_dataset(name: str) -> Optional[Dict[str, Any]]:
    try:
        return get_registry().get_dataset(name)
    except DatasetNotFoundError:
        return None


def load_demo_csv(path: str) -> Optional[pd.DataFrame]:
    """Load a local CSV for preview (no Spark required)."""
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


# Plotly theme

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans", color="#7a8599", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(
        gridcolor="#1e2535",
        linecolor="#252d3d",
        tickfont=dict(color="#7a8599", size=10),
    ),
    yaxis=dict(
        gridcolor="#1e2535",
        linecolor="#252d3d",
        tickfont=dict(color="#7a8599", size=10),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a8599", size=10),
    ),
)

COLORS = {
    "blue":   "#2a7aff",
    "cyan":   "#00d4e8",
    "green":  "#00c98d",
    "orange": "#ff7a2a",
    "red":    "#ff4d6d",
    "purple": "#9d5cff",
    "muted":  "#4a5568",
}

COLOR_SEQUENCE = [
    COLORS["blue"], COLORS["cyan"], COLORS["green"],
    COLORS["orange"], COLORS["purple"], COLORS["red"],
]


# UI Components

def render_tags(tags: List[str]) -> str:
    if not tags:
        return "—"
    return " ".join(
        f'<span class="odvs-badge badge-tag">{t}</span>' for t in tags
    )


def render_version_badge(v: str) -> str:
    return f'<span class="odvs-badge badge-version">{v}</span>'


def render_status_badge(ok: bool, ok_label: str = "OK", fail_label: str = "ERROR") -> str:
    cls = "badge-ok" if ok else "badge-error"
    label = ok_label if ok else fail_label
    return f'<span class="odvs-badge {cls}">{label}</span>'


def stat_row(label: str, value: str) -> str:
    return f"""
    <div class="stat-row">
      <span class="stat-label">{label}</span>
      <span class="stat-value">{value}</span>
    </div>"""


def section_header(text: str) -> None:
    st.markdown(f'<p class="section-header">{text}</p>', unsafe_allow_html=True)


# Pages

def page_overview() -> None:
    """Registry overview — dataset catalog at-a-glance."""
    st.markdown("## Dataset Catalog")

    datasets = fetch_all_datasets()

    if not datasets:
        st.info(
            "No datasets registered yet. Run `python scripts/run_pipeline.py` to ingest your first dataset.",
            icon="ℹ️",
        )
        _render_quickstart()
        return

    # Top-level stats
    total_versions = sum(len(ds.get("versions", [])) for ds in datasets)
    total_rows = sum(
        sum(v.get("row_count", 0) for v in ds.get("versions", []))
        for ds in datasets
    )
    all_tags = sorted({tag for ds in datasets for tag in ds.get("tags", [])})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Datasets", len(datasets))
    col2.metric("Total Versions", total_versions)
    col3.metric("Total Rows (all versions)", f"{total_rows:,}")
    col4.metric("Unique Tags", len(all_tags))

    st.divider()

    # Filters
    col_search, col_tag = st.columns([3, 1])
    with col_search:
        query = st.text_input("🔍 Search datasets", placeholder="Filter by name or description…", label_visibility="collapsed")
    with col_tag:
        tag_filter = st.selectbox("Tag", ["All tags"] + all_tags, label_visibility="collapsed")

    registry = get_registry()
    filtered = registry.search(
        query=query,
        tags=[tag_filter] if tag_filter != "All tags" else None,
    )

    st.markdown(f"<p style='color:#7a8599;font-size:0.8rem;margin-bottom:0.75rem'>{len(filtered)} dataset(s) found</p>", unsafe_allow_html=True)

    # Dataset cards
    for ds in filtered:
        versions = ds.get("versions", [])
        latest_v = ds.get("latest_version") or "—"
        latest_rows = versions[-1].get("row_count", 0) if versions else 0
        tags_html = render_tags(ds.get("tags", []))

        with st.expander(
            f"**{ds['name']}**  ·  {latest_v}  ·  {latest_rows:,} rows",
            expanded=False,
        ):
            c1, c2 = st.columns([2, 1])

            with c1:
                st.markdown(f"_{ds.get('description') or 'No description provided.'}_")
                st.markdown(f"**Tags:** {tags_html}  &nbsp; **Version:** {render_version_badge(latest_v)}", unsafe_allow_html=True)
                schema = ds.get("schema", {})
                if schema:
                    section_header("Schema")
                    schema_df = pd.DataFrame(
                        [{"Column": k, "Type": v} for k, v in schema.items()]
                    )
                    st.dataframe(schema_df, hide_index=True, use_container_width=True, height=min(200, 35 * len(schema) + 38))

            with c2:
                st.markdown("".join([
                    stat_row("Source URI", f"`{ds.get('source_uri', '—')[:40]}…`" if len(ds.get('source_uri', '')) > 40 else f"`{ds.get('source_uri', '—')}`"),
                    stat_row("Table", f"`{ds.get('table_path', '—')}`"),
                    stat_row("Created", ds.get("created_at", "—")[:10]),
                    stat_row("Updated", ds.get("updated_at", "—")[:10]),
                    stat_row("Versions", str(len(versions))),
                ]), unsafe_allow_html=True)

                if st.button("Explore →", key=f"explore_{ds['name']}"):
                    st.session_state["selected_dataset"] = ds["name"]
                    st.session_state["page"] = "Dataset Detail"
                    st.rerun()


def page_dataset_detail(dataset_name: str) -> None:
    """Per-dataset deep-dive: schema, versions, snapshots, lineage, HF card."""
    ds = fetch_dataset(dataset_name)
    if ds is None:
        st.error(f"Dataset not found: `{dataset_name}`")
        return

    versions = ds.get("versions", [])
    latest = ds.get("latest_version")
    tags_html = render_tags(ds.get("tags", []))

    # Header
    col_title, col_actions = st.columns([4, 1])
    with col_title:
        st.markdown(f"## {ds['name']}")
        st.markdown(f"{tags_html}", unsafe_allow_html=True)
        if ds.get("description"):
            st.markdown(f"_{ds['description']}_")

    with col_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Catalog", use_container_width=True):
            st.session_state.pop("selected_dataset", None)
            st.session_state["page"] = "Catalog"
            st.rerun()

    # Summary metrics
    total_rows = sum(v.get("row_count", 0) for v in versions)
    latest_rows = versions[-1].get("row_count", 0) if versions else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Versions", len(versions))
    c2.metric("Latest Version", latest or "—")
    c3.metric("Rows (latest)", f"{latest_rows:,}")
    c4.metric("Total Rows (all)", f"{total_rows:,}")

    st.divider()

    # Tabs
    tab_schema, tab_versions, tab_lineage, tab_card, tab_diff = st.tabs([
        "Schema", "Version History", "Lineage", "Dataset Card", "Diff Viewer"
    ])

    # Schema tab
    with tab_schema:
        schema = ds.get("schema", {})
        if not schema:
            st.info("No schema recorded for this dataset.")
        else:
            col_schema, col_stats = st.columns([1, 1])
            with col_schema:
                section_header("Column definitions")
                schema_rows = [{"Column": k, "Type": v, "Nullable": "yes"} for k, v in schema.items()]
                st.dataframe(
                    pd.DataFrame(schema_rows),
                    hide_index=True,
                    use_container_width=True,
                    height=min(400, 35 * len(schema_rows) + 38),
                )

            with col_stats:
                section_header("Type distribution")
                type_counts = {}
                for dtype in schema.values():
                    base = dtype.split("[")[0].split("(")[0]
                    type_counts[base] = type_counts.get(base, 0) + 1

                fig = px.pie(
                    names=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    color_discrete_sequence=COLOR_SEQUENCE,
                    hole=0.55,
                )
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_traces(
                    textfont=dict(color="#e8ecf4", size=11),
                    marker=dict(line=dict(color="#0d0f14", width=2)),
                )
                st.plotly_chart(fig, use_container_width=True)

    # Versions tab
    with tab_versions:
        if not versions:
            st.info("No versions recorded yet.")
        else:
            section_header(f"{len(versions)} version(s)")

            # Row count over versions chart
            version_df = pd.DataFrame([
                {
                    "Version": v["version_tag"],
                    "Rows": v.get("row_count", 0),
                    "Created": v.get("created_at", "")[:10],
                    "Snapshot ID": str(v.get("snapshot_id") or "—"),
                    "Checksum": v.get("checksum", "—"),
                }
                for v in versions
            ])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=version_df["Version"],
                y=version_df["Rows"],
                marker_color=COLORS["blue"],
                marker_line_color=COLORS["cyan"],
                marker_line_width=0.5,
                opacity=0.85,
                hovertemplate="<b>%{x}</b><br>Rows: %{y:,}<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Row Count by Version", font=dict(color="#e8ecf4", size=13), x=0),
                xaxis_title=None,
                yaxis_title="Rows",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                version_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rows": st.column_config.NumberColumn(format="%d"),
                },
            )

    # Lineage tab
    with tab_lineage:
        tracker = get_lineage_tracker()
        nodes = tracker.get_lineage(dataset_name)

        if not nodes:
            st.info("No lineage recorded. Run the pipeline to generate lineage data.")
        else:
            provenance = tracker.get_full_provenance(dataset_name)

            c1, c2, c3 = st.columns(3)
            c1.metric("Lineage Nodes", provenance["version_count"])
            c2.metric("Unique Sources", len(provenance["all_source_uris"]))
            c3.metric("Transforms Applied", len(provenance["all_transforms"]))

            section_header("Source URIs")
            for uri in provenance["all_source_uris"]:
                st.code(uri, language=None)

            section_header("Transforms pipeline")
            transform_df = pd.DataFrame(
                [{"Step": i + 1, "Transform": t} for i, t in enumerate(provenance["all_transforms"])]
            )
            st.dataframe(transform_df, hide_index=True, use_container_width=True)

            section_header("Lineage nodes")
            for node in nodes:
                is_latest = node.version_tag == latest
                parent_str = " ← ".join(node.parent_node_ids[:2]) if node.parent_node_ids else "root"
                st.markdown(
                    f"""<div class="version-row {'latest' if is_latest else ''}">
                      <b>{node.version_tag}</b>
                      {'<span class="odvs-badge badge-ok">LATEST</span>' if is_latest else ''}
                      &nbsp;·&nbsp; node: <code>{node.node_id[:12]}…</code>
                      &nbsp;·&nbsp; rows: <code>{node.row_count:,}</code>
                      &nbsp;·&nbsp; {node.created_at[:19]}
                      <br><span style='color:#4a5568;font-size:0.75rem'>parent → {parent_str[:40]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # Dataset Card tab
    with tab_card:
        hf = get_hf_simulator()
        try:
            card = hf.generate_dataset_card(dataset_name)
            col_md, col_meta = st.columns([3, 1])

            with col_md:
                section_header("Dataset Card Preview (Hugging Face format)")
                st.code(card.to_markdown(), language="markdown")

            with col_meta:
                section_header("Hub entry")
                hub_entry = hf.dataset_info(dataset_name)
                st.json(hub_entry)

                st.markdown("<br>", unsafe_allow_html=True)
                hub_url = hf.get_download_url(dataset_name)
                st.markdown(f"**Simulated Hub URL:**")
                st.code(hub_url, language=None)

        except Exception as exc:
            st.error(f"Could not generate dataset card: {exc}")

    # Diff Viewer tab
    with tab_diff:
        _render_diff_viewer(dataset_name, versions)


def _render_diff_viewer(dataset_name: str, versions: List[Dict]) -> None:
    """Inline diff viewer comparing two CSV/Parquet files for quick diff."""
    if len(versions) < 2:
        st.info("Need at least 2 versions to compute a diff. Upload two files below for an ad-hoc comparison.")

    st.markdown("#### Ad-hoc Dataset Diff")
    st.markdown(
        "<p style='color:#7a8599;font-size:0.83rem'>Upload two versions of a CSV dataset to compare schema and row-level differences.</p>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Version A (before)**")
        file_a = st.file_uploader("Upload CSV — Version A", type=["csv"], key="diff_a", label_visibility="collapsed")
    with col_b:
        st.markdown("**Version B (after)**")
        file_b = st.file_uploader("Upload CSV — Version B", type=["csv"], key="diff_b", label_visibility="collapsed")

    if file_a and file_b:
        with st.spinner("Computing diff…"):
            try:
                df_a = pd.read_csv(file_a, low_memory=False)
                df_b = pd.read_csv(file_b, low_memory=False)

                engine = get_diff_engine()
                diff = engine.diff(
                    df_before=df_a,
                    df_after=df_b,
                    dataset_name=dataset_name,
                    version_before=f"A ({len(df_a):,} rows)",
                    version_after=f"B ({len(df_b):,} rows)",
                )

                # Schema diff
                section_header("Schema diff")
                s = diff.schema_diff
                if not s.has_changes:
                    st.success("No schema changes detected.", icon="✓")
                else:
                    if s.added_columns:
                        st.markdown(f"**Added columns:** `{'`, `'.join(s.added_columns)}`")
                    if s.removed_columns:
                        st.markdown(f"**Removed columns:** `{'`, `'.join(s.removed_columns)}`")
                    if s.type_changed:
                        for col, (old, new) in s.type_changed.items():
                            st.markdown(f"**Type change** `{col}`: `{old}` → `{new}`")

                # Row diff
                section_header("Row diff")
                r = diff.row_diff
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Rows Added", f"+{r.rows_added:,}", delta=f"+{r.rows_added:,}", delta_color="normal")
                rc2.metric("Rows Removed", f"-{r.rows_removed:,}", delta=f"-{r.rows_removed:,}", delta_color="inverse")
                rc3.metric("Unchanged", f"{r.rows_unchanged:,}")
                rc4.metric("Change Rate", f"{r.change_rate:.1%}")

                # Statistical diffs — show significant ones
                significant = [s for s in diff.statistical_diffs if s.is_significant]
                if significant:
                    section_header(f"Significant column changes ({len(significant)})")
                    sig_rows = []
                    for s in significant:
                        sig_rows.append({
                            "Column": s.column,
                            "Null Δ": f"{s.null_fraction_delta:+.1%}" if s.null_fraction_delta else "—",
                            "Mean Δ": f"{s.mean_delta:+.3f}" if s.mean_delta is not None else "—",
                            "Unique Δ": f"{s.unique_count_delta:+,}" if s.unique_count_delta is not None else "—",
                        })
                    st.dataframe(pd.DataFrame(sig_rows), hide_index=True, use_container_width=True)

                if diff.has_breaking_changes:
                    st.error("⚠️ Breaking schema changes detected — downstream consumers may be affected.", icon="⚠️")

            except Exception as exc:
                st.error(f"Diff failed: {exc}")


def page_run_pipeline() -> None:
    """Interactive pipeline runner — ingest and preview without needing CLI."""
    st.markdown("## Run Pipeline")
    st.markdown(
        "<p style='color:#7a8599'>Ingest a CSV dataset, apply normalization and deduplication, "
        "then preview results. Full Iceberg write requires the CLI.</p>",
        unsafe_allow_html=True,
    )

    col_form, col_preview = st.columns([1, 2])

    with col_form:
        section_header("Pipeline configuration")
        source_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        dataset_name = st.text_input("Dataset Name", placeholder="e.g. ecommerce_events")
        version_tag = st.text_input("Version Tag", placeholder="e.g. v1.0.0")
        tags_input = st.text_input("Tags (comma-separated)", placeholder="e.g. ecommerce,events")
        description = st.text_area("Description", height=80)
        compression = st.selectbox("Compression", ["zstd", "snappy", "gzip", "none"])
        run_dedup = st.toggle("Run deduplication", value=True)
        run_compression_bench = st.toggle("Run compression benchmark", value=True)

        run_btn = st.button("▶  Run Pipeline (preview)", type="primary", use_container_width=True)

    with col_preview:
        if run_btn and source_file:
            if not dataset_name or not version_tag:
                st.error("Dataset Name and Version Tag are required.")
                return

            with st.spinner("Running pipeline…"):
                try:
                    raw_df = pd.read_csv(source_file, low_memory=False)
                    st.session_state["pipeline_raw"] = raw_df

                    # Transform
                    pipeline = TransformPipeline.standard()
                    df = pipeline.run(raw_df.copy())

                    # Dedup
                    dedup_result = None
                    if run_dedup:
                        deduplicator = HashDeduplicator()
                        df, dedup_result = deduplicator.deduplicate(df)

                    st.session_state["pipeline_result"] = df
                    st.session_state["pipeline_dedup"] = dedup_result

                    # Compression benchmark
                    if run_compression_bench:
                        benchmarker = CompressionBenchmarker()
                        comp_report = benchmarker.benchmark(df, dataset_name=dataset_name, sample_n=5000)
                        st.session_state["comp_report"] = comp_report

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

        # Render results if available
        if "pipeline_result" in st.session_state:
            df = st.session_state["pipeline_result"]
            raw = st.session_state.get("pipeline_raw", df)
            dedup_r = st.session_state.get("pipeline_dedup")

            section_header("Pipeline output")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Raw Rows", f"{len(raw):,}")
            mc2.metric("After Transform + Dedup", f"{len(df):,}", delta=f"-{len(raw) - len(df):,}" if len(raw) != len(df) else None, delta_color="off")
            mc3.metric("Columns", len(df.columns))

            if dedup_r:
                st.markdown(
                    f"<p style='color:#00c98d;font-size:0.82rem'>✓ Dedup removed {dedup_r.duplicates_removed:,} rows "
                    f"({dedup_r.deduplication_rate:.1%} duplicate rate)</p>",
                    unsafe_allow_html=True,
                )

            tab_data, tab_schema, tab_bench = st.tabs(["Data Preview", "Schema", "Compression Benchmark"])

            with tab_data:
                st.dataframe(df.head(100), use_container_width=True, height=320)

            with tab_schema:
                schema_df = pd.DataFrame([
                    {"Column": c, "Type": str(t), "Null %": f"{df[c].isnull().mean():.1%}", "Unique": df[c].nunique()}
                    for c, t in df.dtypes.items()
                ])
                st.dataframe(schema_df, hide_index=True, use_container_width=True)

            with tab_bench:
                if "comp_report" in st.session_state:
                    comp_report = st.session_state["comp_report"]
                    bench_df = comp_report.to_dataframe().sort_values("compression_ratio", ascending=False)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Compression Ratio",
                        x=bench_df["codec"],
                        y=bench_df["compression_ratio"],
                        marker_color=COLORS["cyan"],
                        yaxis="y",
                        offsetgroup=1,
                    ))
                    fig.add_trace(go.Bar(
                        name="Write ms",
                        x=bench_df["codec"],
                        y=bench_df["write_ms"],
                        marker_color=COLORS["orange"],
                        yaxis="y2",
                        offsetgroup=2,
                    ))
                    fig.update_layout(
                        **PLOTLY_LAYOUT,
                        barmode="group",
                        yaxis_title="Compression Ratio",
                        yaxis2=dict(title="Write (ms)", overlaying="y", side="right", gridcolor="#1e2535"),
                        title=dict(text="Codec Comparison", font=dict(color="#e8ecf4", size=13), x=0),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.success(f"✓ Recommended codec: **{comp_report.recommendation}**")
                    st.dataframe(bench_df, hide_index=True, use_container_width=True)
                else:
                    st.info("Enable 'Run compression benchmark' to see codec comparison.")

            # CLI command
            section_header("CLI command to write to Iceberg")
            tags_str = tags_input if "tags_input" in dir() else ""
            cmd = (
                f"python scripts/run_pipeline.py \\\n"
                f"  --source <your_file.csv> \\\n"
                f"  --dataset-name {dataset_name or 'your_dataset'} \\\n"
                f"  --version-tag {version_tag or 'v1.0.0'} \\\n"
                f"  --compression {compression}"
            )
            st.code(cmd, language="bash")

        elif run_btn and not source_file:
            st.warning("Please upload a CSV file first.")


def page_compression_benchmark() -> None:
    """Standalone compression benchmark page."""
    st.markdown("## Compression Benchmark")
    st.markdown(
        "<p style='color:#7a8599'>Upload a CSV to benchmark all Parquet compression codecs. "
        "Use results to choose the optimal codec for your dataset characteristics.</p>",
        unsafe_allow_html=True,
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        sample_n = st.number_input("Sample size (rows)", min_value=1000, max_value=100000, value=10000, step=1000)
    with col_opts2:
        codecs = st.multiselect(
            "Codecs to benchmark",
            ["snappy", "gzip", "zstd", "brotli", "lz4", "none"],
            default=["snappy", "gzip", "zstd", "none"],
        )

    if st.button("▶  Run Benchmark", type="primary") and file:
        with st.spinner("Benchmarking codecs…"):
            try:
                df = pd.read_csv(file, low_memory=False)
                benchmarker = CompressionBenchmarker(codecs=codecs)
                report = benchmarker.benchmark(df, dataset_name=file.name, sample_n=int(sample_n))

                bench_df = report.to_dataframe().sort_values("compression_ratio", ascending=False)

                st.success(f"**Recommended codec: `{report.recommendation}`**")

                c1, c2 = st.columns(2)
                with c1:
                    section_header("Compression ratio (higher = smaller files)")
                    fig1 = px.bar(
                        bench_df,
                        x="codec",
                        y="compression_ratio",
                        color="codec",
                        color_discrete_sequence=COLOR_SEQUENCE,
                        labels={"compression_ratio": "Ratio", "codec": "Codec"},
                    )
                    fig1.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)

                with c2:
                    section_header("Write + Read latency (ms)")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(name="Write (ms)", x=bench_df["codec"], y=bench_df["write_ms"], marker_color=COLORS["blue"]))
                    fig2.add_trace(go.Bar(name="Read (ms)", x=bench_df["codec"], y=bench_df["read_ms"], marker_color=COLORS["cyan"]))
                    fig2.update_layout(**PLOTLY_LAYOUT, barmode="group")
                    st.plotly_chart(fig2, use_container_width=True)

                section_header("Throughput (MB/s)")
                fig3 = px.bar(
                    bench_df,
                    x="codec",
                    y="throughput_mb_s",
                    color="codec",
                    color_discrete_sequence=COLOR_SEQUENCE,
                    labels={"throughput_mb_s": "MB/s", "codec": "Codec"},
                )
                fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

                section_header("Full results table")
                st.dataframe(bench_df, hide_index=True, use_container_width=True)

            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")

    elif not file:
        st.info("Upload a CSV file to run the benchmark.", icon="ℹ️")


def page_settings() -> None:
    """Configuration and environment info."""
    st.markdown("## Settings & Configuration")
    cfg = get_config_cached()

    col1, col2 = st.columns(2)

    with col1:
        section_header("S3 Configuration")
        st.markdown("".join([
            stat_row("Endpoint URL", f"`{cfg.s3.endpoint_url}`"),
            stat_row("Bucket", f"`{cfg.s3.bucket}`"),
            stat_row("Region", f"`{cfg.s3.region}`"),
            stat_row("Warehouse Path", f"`{cfg.s3.warehouse_path}`"),
        ]), unsafe_allow_html=True)

        section_header("Iceberg Configuration")
        st.markdown("".join([
            stat_row("Catalog Name", f"`{cfg.iceberg.catalog_name}`"),
            stat_row("Database", f"`{cfg.iceberg.database}`"),
            stat_row("Default Compression", f"`{cfg.iceberg.default_compression}`"),
            stat_row("Target File Size", f"`{cfg.iceberg.target_file_size_bytes // (1024*1024)} MB`"),
        ]), unsafe_allow_html=True)

    with col2:
        section_header("Spark Configuration")
        st.markdown("".join([
            stat_row("App Name", f"`{cfg.spark.app_name}`"),
            stat_row("Master", f"`{cfg.spark.master}`"),
            stat_row("Driver Memory", f"`{cfg.spark.driver_memory}`"),
            stat_row("Iceberg Version", f"`{cfg.spark.iceberg_version}`"),
        ]), unsafe_allow_html=True)

        section_header("Registry")
        st.markdown("".join([
            stat_row("Registry Path", f"`{cfg.registry.registry_path}`"),
            stat_row("Registry File", f"`{cfg.registry.registry_file}`"),
        ]), unsafe_allow_html=True)

        section_header("Environment")
        st.markdown("".join([
            stat_row("ODVS_ENV", f"`{cfg.environment}`"),
            stat_row("Structured Logs", f"`{cfg.logging.structured}`"),
            stat_row("Log Level", f"`{cfg.logging.level}`"),
        ]), unsafe_allow_html=True)

    st.divider()
    section_header("Environment overrides")
    env_vars = {
        "S3_ENDPOINT_URL": "S3/MinIO endpoint",
        "AWS_ACCESS_KEY_ID": "S3 access key",
        "AWS_SECRET_ACCESS_KEY": "S3 secret key",
        "ODVS_S3_BUCKET": "S3 bucket name",
        "ODVS_ICEBERG_DB": "Iceberg database name",
        "ODVS_REGISTRY_PATH": "Registry storage path",
        "ODVS_ENV": "Environment (development | production)",
        "ODVS_LOG_LEVEL": "Log level (DEBUG | INFO | WARNING)",
        "ODVS_STRUCTURED_LOGS": "JSON log output (true | false)",
        "SPARK_MASTER": "Spark master URL",
        "SPARK_DRIVER_MEMORY": "Spark driver memory",
    }
    env_df = pd.DataFrame([
        {
            "Variable": k,
            "Description": v,
            "Current": os.environ.get(k, "(default)"),
        }
        for k, v in env_vars.items()
    ])
    st.dataframe(env_df, hide_index=True, use_container_width=True)


def _render_quickstart() -> None:
    """Show a quickstart guide when no datasets are registered."""
    st.divider()
    st.markdown("### Quickstart")
    st.code("""
# 1. Start MinIO (local S3)
docker compose -f docker/docker-compose.yml up -d minio

# 2. Ingest your first dataset
python scripts/run_pipeline.py \\
    --source examples/sample_dataset.csv \\
    --dataset-name ecommerce_events \\
    --version-tag v1.0.0 \\
    --description "E-commerce events dataset" \\
    --tags ecommerce,events \\
    --compression zstd

# 3. Refresh this page to see your dataset
""", language="bash")


# Sidebar + Routing

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown('<p class="odvs-logo">◈ ODVS</p>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#4a5568;font-size:0.72rem;margin-top:-0.25rem;margin-bottom:1.5rem'>"
            "Open Dataset Versioning System</p>",
            unsafe_allow_html=True,
        )

        pages = {
            "Catalog": "🗂",
            "Run Pipeline": "▶",
            "Compression Benchmark": "📊",
            "Settings": "⚙",
        }

        default_page = st.session_state.get("page", "Catalog")
        radio_default = default_page if default_page in pages else "Catalog"
        selected = st.radio(
            "Navigation",
            list(pages.keys()),
            format_func=lambda p: f"{pages[p]}  {p}",
            index=list(pages.keys()).index(radio_default),
            label_visibility="collapsed",
        )

        # If session state says Dataset Detail, preserve it over the radio selection
        if st.session_state.get("page") == "Dataset Detail" and st.session_state.get("selected_dataset"):
            selected = "Dataset Detail"
        else:
            st.session_state["page"] = selected
            st.session_state.pop("selected_dataset", None)

        # Dataset quick-select (only if datasets exist)
        registry = get_registry()
        dataset_names = registry.list_datasets()
        if dataset_names:
            st.divider()
            st.markdown('<p class="section-header">Quick navigate</p>', unsafe_allow_html=True)
            chosen = st.selectbox(
                "Jump to dataset",
                ["—"] + sorted(dataset_names),
                label_visibility="collapsed",
            )
            if chosen != "—":
                st.session_state["selected_dataset"] = chosen
                selected = "Dataset Detail"

        st.divider()

        # Registry stats
        stats = registry.get_stats()
        st.markdown("".join([
            '<p class="section-header">Registry</p>',
            stat_row("Datasets", str(stats["total_datasets"])),
            stat_row("Versions", str(stats["total_versions"])),
            stat_row("Total Rows", f"{stats['total_rows_across_versions']:,}"),
        ]), unsafe_allow_html=True)

        st.divider()
        st.markdown(
            "<p style='color:#2a3a4a;font-size:0.7rem;text-align:center'>"
            "Apache Iceberg · Spark · S3<br>"
            "ODVS v1.0.0</p>",
            unsafe_allow_html=True,
        )

    return selected


def main() -> None:
    page = render_sidebar()

    if page == "Catalog":
        st.session_state.pop("selected_dataset", None)
        page_overview()

    elif page == "Dataset Detail":
        selected = st.session_state.get("selected_dataset")
        if not selected:
            st.info("Select a dataset from the sidebar or catalog.")
        else:
            page_dataset_detail(selected)

    elif page == "Run Pipeline":
        page_run_pipeline()

    elif page == "Compression Benchmark":
        page_compression_benchmark()

    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()