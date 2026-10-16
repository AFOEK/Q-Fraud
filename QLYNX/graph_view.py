"""Live QLYNX graph rendering. Aggregate diagrams never invent entity identities.

Real account<->resource edges can be built only from an explicit relation upload.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go

COLOR = {"account": "#233654", "device": "#d92d43", "network": "#ed8a23",
         "payment": "#219c67", "aggregate": "#6c7fa2"}


def context_for_account(row: pd.Series, features: pd.DataFrame | None = None):
    """Supply whitelisted aggregate counts from current view or feature upload."""
    data = row.to_dict()
    if features is not None and "account_id" in features:
        found = features.loc[features.account_id.astype(str) == str(row.account_id)]
        if not found.empty:
            for col, val in found.iloc[0].items():
                if col not in data or pd.isna(data[col]):
                    data[col] = val
    return data


def aggregate_graph(account_id: str, context: dict):
    """A dynamically built star of *summary categories*, not actual linked entities."""
    graph = nx.Graph()
    center = f"Account {account_id}"
    graph.add_node(center, group="account", detail="Selected account (known account ID)", size=25)
    fields = [
        ("Devices", "distinct_devices", "device"),
        ("Shared-device indicators", "shared_device_count", "device"),
        ("Device account degree", "max_device_account_degree", "device"),
        ("Networks", "distinct_networks", "network"),
        ("Regions", "distinct_regions", "network"),
        ("Payment tokens", "distinct_payment_tokens", "payment"),
        ("Payment events", "payment_event_count", "payment"),
        ("Sessions", "session_count", "aggregate"),
    ]
    available = 0
    for label, key, group in fields:
        value = context.get(key)
        try:
            num = float(value)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(num):
            continue
        node = f"{label} · {num:,.0f}"
        graph.add_node(node, group=group, detail=f"Observed aggregate: {key}={num:g}. Not a specific entity ID.", size=12)
        graph.add_edge(center, node, relation="aggregate")
        available += 1
    return graph, available


def parse_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Strict, sanitized relationship input: account_id,entity_type,entity_id."""
    needed = {"account_id", "entity_type", "entity_id"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError("Edge CSV requires account_id, entity_type, entity_id (missing: " + ", ".join(sorted(missing)) + ")")
    if len(frame) > 100000:
        raise ValueError("Upload at most 100,000 relationship records for this local demonstration.")
    edge = frame[["account_id", "entity_type", "entity_id"]].dropna().astype(str).copy()
    for col in edge.columns:
        edge[col] = edge[col].str.strip()
    edge["entity_type"] = edge["entity_type"].str.lower().replace({
        "payment_token": "payment", "token": "payment", "ip": "network", "ip_address": "network"
    })
    if not edge.entity_type.isin(("device", "network", "payment")).all():
        raise ValueError("entity_type must be device, network, or payment.")
    if (edge[["account_id", "entity_type", "entity_id"]] == "").any().any():
        raise ValueError("Empty account or resource IDs are not allowed.")
    if edge.account_id.str.len().max() > 128 or edge.entity_id.str.len().max() > 128:
        raise ValueError("Identifiers must be under 129 characters.")
    return edge.drop_duplicates()


def linked_graph(account_id: str, edges: pd.DataFrame, max_entities=24, max_accounts=24):
    """Observed two-hop ego graph from explicit uploaded relationship rows."""
    selected = edges.loc[edges.account_id == str(account_id)]
    if selected.empty:
        return nx.Graph(), 0
    selected = selected.head(max_entities)
    entities = set(zip(selected.entity_type, selected.entity_id))
    matches = edges.loc[[ (a, b) in entities for a, b in zip(edges.entity_type, edges.entity_id) ]]
    second_accounts = [a for a in matches.account_id.unique() if a != str(account_id)][:max_accounts]
    subset = matches.loc[matches.account_id.isin([str(account_id)] + second_accounts)]
    graph = nx.Graph()
    main = "Account " + str(account_id)
    graph.add_node(main, group="account", detail="Selected account", size=28)
    for r in subset.itertuples(index=False):
        acct = "Account " + r.account_id
        entity = f"{r.entity_type}: {r.entity_id}"
        graph.add_node(acct, group="account", detail=f"Observed account {r.account_id}", size=25 if acct == main else 15)
        graph.add_node(entity, group=r.entity_type, detail=f"Observed {r.entity_type} identifier: {r.entity_id}", size=14)
        graph.add_edge(acct, entity, relation="observed")
    return graph, len(entities)


def render_graph(graph: nx.Graph, *, aggregate: bool = False):
    if not graph.nodes:
        return go.Figure()
    pos = nx.spring_layout(graph, seed=42, iterations=90, k=1.4 / (len(graph.nodes) ** .3))
    fig = go.Figure()
    for group in ("account", "device", "network", "payment", "aggregate"):
        nodes = [node for node, attr in graph.nodes(data=True) if attr["group"] == group]
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text", name=group.title(),
            text=[n if group == "account" or aggregate else "" for n in nodes],
            textposition="top center", textfont=dict(size=11, color="#26364f"),
            customdata=[graph.nodes[n]["detail"] for n in nodes],
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(size=[graph.nodes[n].get("size", 14) for n in nodes],
                        color=COLOR[group], line=dict(color="white", width=1.6))))
    for u, v, attr in graph.edges(data=True):
        style = "dot" if attr.get("relation") == "aggregate" else "solid"
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]],
            mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color="#c7d2e2", width=1.4, dash=style)))
    fig.update_layout(height=520, margin=dict(l=15, r=15, t=25, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-.08),
                      hoverlabel=dict(bgcolor="white", font_color="#18233c"),
                      dragmode="pan")
    return fig
