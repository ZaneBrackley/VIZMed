# data/graph/build_graph.py
import os
import torch
from torch_geometric.data import Data
import pandas as pd
import ast
from collections import defaultdict

GRAPH_CACHE = "data/graph/ontology_graph.pt"

def build_graph(vocab_path: str, add_sibling_edges: bool = True, include_ancestors: bool = True):
    """
    Build ontology graph with dense connections (parent/descendant/ancestor/sibling).
    Saves node labels and semantic type colors.
    """

    # --- Load vocab ---
    df = pd.read_csv(vocab_path)
    df["parents"] = df["parents"].apply(ast.literal_eval)
    df["descendants"] = df["descendants"].apply(ast.literal_eval)

    # Ancestors may not exist in dataset-specific vocab (check metadata if needed)
    if "ancestors" in df.columns:
        df["ancestors"] = df["ancestors"].apply(ast.literal_eval)
    else:
        df["ancestors"] = [[] for _ in range(len(df))]

    # --- Map CUI to index ---
    cui_to_idx = {cui: idx for idx, cui in enumerate(df["cui"])}

    # --- Build edges (parent, descendant, ancestor) ---
    edges = []

    for idx, row in df.iterrows():
        # Parent edges
        for parent in row["parents"]:
            parent_cui = parent.split(", ")[-1].strip("}")
            if parent_cui in cui_to_idx:
                edges.append((cui_to_idx[parent_cui], idx))

        # Ancestor edges
        if include_ancestors:
            for anc in row["ancestors"]:
                anc_cui = anc.split(", ")[-1].strip("}")
                if anc_cui in cui_to_idx:
                    edges.append((cui_to_idx[anc_cui], idx))

        # Descendant edges
        for desc in row["descendants"]:
            desc_cui = desc.split(", ")[-1].strip("}")
            if desc_cui in cui_to_idx:
                edges.append((idx, cui_to_idx[desc_cui]))

    # --- Add sibling edges ---
    if add_sibling_edges:
        parent_to_children = defaultdict(list)
        for idx, row in df.iterrows():
            for parent in row["parents"]:
                parent_cui = parent.split(", ")[-1].strip("}")
                if parent_cui in cui_to_idx:
                    parent_to_children[parent_cui].append(idx)

        for siblings in parent_to_children.values():
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    edges.append((siblings[i], siblings[j]))
                    edges.append((siblings[j], siblings[i]))

    if not edges:
        raise ValueError("No edges built from vocab file.")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Placeholder node features
    x = torch.arange(len(df), dtype=torch.float).unsqueeze(1)

    # Node labels
    node_labels = [f"{row['name']} ({row['cui']})" for _, row in df.iterrows()]

    # Semantic type colors
    semantic_types = df["semantic_type"].fillna("Unknown").unique().tolist()
    color_map = {stype: i for i, stype in enumerate(semantic_types)}
    node_colors = [color_map[row["semantic_type"]] for _, row in df.iterrows()]

    # Build graph object
    graph = Data(
        x=x,
        edge_index=edge_index,
        node_labels=node_labels,
        node_colors=torch.tensor(node_colors, dtype=torch.long),
    )

    torch.save(graph, GRAPH_CACHE)
    print(f"[✓] Saved ontology graph with dense edges to {GRAPH_CACHE}")

    return graph
