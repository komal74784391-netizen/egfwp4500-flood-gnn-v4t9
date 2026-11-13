"""Build a directed river-network graph from connectivity tables."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RiverGraph:
    node_ids: List[str]
    adjacency: Dict[str, List[str]]
    distances: np.ndarray


def build_graph(edges: List[dict]) -> RiverGraph:
    adjacency: Dict[str, List[str]] = {}
    node_ids = sorted({edge["upstream"] for edge in edges} | {edge["downstream"] for edge in edges})
    for node in node_ids:
        adjacency[node] = []
    for edge in edges:
        adjacency[edge["upstream"]].append(edge["downstream"])
    matrix = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    index = {node: i for i, node in enumerate(node_ids)}
    for edge in edges:
        matrix[index[edge["upstream"]], index[edge["downstream"]]] = 1.0
    distances = np.where(matrix > 0, 1.0, np.inf)
    np.fill_diagonal(distances, 0.0)
    return RiverGraph(node_ids=node_ids, adjacency=adjacency, distances=distances)
