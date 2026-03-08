from __future__ import annotations

from collections import defaultdict

import pandas as pd

from weather_markov.markov.graph import TransitionGraph


class LayeredTransitionGraph(TransitionGraph):
    """
    Weighted directed transition graph between states.
    Works as a bipartite graph (methods 1, 3) and as a shared graph (method 2).

    The key method predict_distribution() allows chaining graphs
    without extra code in MarkovChain.
    """

    def __init__(self):
        super().__init__()
        self._layers = list()

    # --- Building ---
    def add_layer(self, states: list[str]) -> None:
        self._layers.append(states)

    def add_layers(self, layers: list[list[str]]) -> None:
        self._layers.extend(layers)

    @classmethod
    def from_pairs(
        cls, pairs: list[tuple[str, str]], layers: list[list[str]]
    ) -> "LayeredTransitionGraph":
        g = cls()
        g.add_transitions_from(pairs)
        g.add_layers(layers)
        return g

    # --- Properties ---

    @property
    def layers(self) -> list[tuple[str, str]]:
        return sorted(self._layers)
