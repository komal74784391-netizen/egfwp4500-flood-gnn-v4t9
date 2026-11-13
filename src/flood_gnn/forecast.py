"""Message-passing forecaster for short-range streamflow prediction."""

from dataclasses import dataclass

import numpy as np

from .graph import RiverGraph


@dataclass
class ForecastConfig:
    hidden_dim: int = 32
    rounds: int = 3


class GraphForecaster:
    def __init__(self, graph: RiverGraph, config: ForecastConfig, seed: int = 0):
        self.graph = graph
        self.config = config
        rng = np.random.default_rng(seed)
        n = len(graph.node_ids)
        self.node_weights = rng.normal(size=(n, config.hidden_dim))
        self.message_weights = rng.normal(size=(config.hidden_dim, config.hidden_dim))
        self.readout = rng.normal(size=(config.hidden_dim, 1))

    def propagate(self, states: np.ndarray) -> np.ndarray:
        hidden = states @ self.node_weights
        for _ in range(self.config.rounds):
            messages = np.zeros_like(hidden)
            for i, node in enumerate(self.graph.node_ids):
                for downstream in self.graph.adjacency[node]:
                    j = self.graph.node_ids.index(downstream)
                    messages[j] += hidden[i]
            hidden = np.tanh(messages @ self.message_weights + hidden)
        return hidden @ self.readout

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.propagate(states).ravel()
