from __future__ import annotations

from weather_markov.markov.graph import TransitionGraph


class MarkovChain:
    """
    A chain of TransitionGraphs for multi-step prediction.

    Method 1 -> MarkovChain([graph_feb_may])
    Method 2 -> MarkovChain([shared_graph] * n_steps)   ← from_single_graph()
    Method 3 -> MarkovChain([g1, g2, g3, ..., gN])
    """

    def __init__(self, graphs: list[TransitionGraph]):
        if not graphs:
            raise ValueError("At least one graph is required")
        self.graphs = graphs

    @classmethod
    def from_single_graph(cls, graph: TransitionGraph, steps: int) -> "MarkovChain":
        """Method 2: a single graph applied steps times"""
        return cls([graph] * steps)

    def predict(self, initial_state: str) -> dict[str, float]:
        """Chains predictions through all graphs"""
        return self.predict_from_distribution({initial_state: 1.0})

    def predict_from_distribution(
        self, initial_dist: dict[str, float]
    ) -> dict[str, float]:
        """Allows starting from a distribution rather than a single state"""
        current = initial_dist
        for graph in self.graphs:
            current = graph.predict_distribution(current)
        return current

    def sub_chain(self, start: int, end: int | None = None) -> "MarkovChain":
        """Chain slice — useful for predicting from an intermediate step"""
        return MarkovChain(self.graphs[start:end])

    @staticmethod
    def most_likely(distribution: dict[str, float]) -> str:
        return max(distribution, key=distribution.get)

    @property
    def n_steps(self) -> int:
        return len(self.graphs)
