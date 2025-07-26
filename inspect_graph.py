import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_PATH = "data/graph/ontology_graph.pt"

def main():
    # Allowlist PyG objects for torch.load (required in PyTorch 2.6+)
    torch.serialization.add_safe_globals([Data, DataEdgeAttr])

    # Load saved graph
    graph = torch.load(GRAPH_PATH, weights_only=False)

    # Convert to NetworkX for visualization
    nx_graph = nx.Graph()

    # --- Add nodes with labels and colors ---
    node_labels = getattr(graph, "node_labels", [str(i) for i in range(graph.num_nodes)])
    node_colors = getattr(graph, "node_colors", torch.zeros(graph.num_nodes, dtype=torch.long))

    for i in range(graph.num_nodes):
        nx_graph.add_node(i, label=node_labels[i], color=node_colors[i].item())

    # --- Add edges ---
    for src, dst in graph.edge_index.t().tolist():
        nx_graph.add_edge(src, dst)

    # --- Layout ---
    pos = nx.spring_layout(nx_graph, seed=42, k=0.3)  # `k` controls spacing

    # --- Draw nodes with semantic-type colors ---
    colors = [data["color"] for _, data in nx_graph.nodes(data=True)]
    nx.draw(
        nx_graph, pos,
        with_labels=False,
        node_color=colors,
        cmap=plt.cm.tab20,  # 20-color discrete palette
        node_size=500,
        edge_color="gray",
        alpha=0.8
    )

    # --- Draw labels separately ---
    labels = nx.get_node_attributes(nx_graph, "label")
    nx.draw_networkx_labels(nx_graph, pos, labels, font_size=6)

    plt.title("Ontology Graph (CUI + Name, Colored by Semantic Type)")
    plt.axis("off")
    plt.show()
