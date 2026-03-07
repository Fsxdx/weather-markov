from __future__ import annotations

from collections import defaultdict

import pandas as pd


class TransitionGraph:
    """
    Weighted directed transition graph between states.
    Works as a bipartite graph (methods 1, 3) and as a shared graph (method 2).

    The key method predict_distribution() allows chaining graphs
    without extra code in MarkovChain.
    """

    def __init__(self):
        # counts[from_state][to_state] = int
        self._counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._from_states: set[str] = set()
        self._to_states: set[str] = set()

    # --- Building ---

    def add_transition(self, from_state: str, to_state: str) -> None:
        self._counts[from_state][to_state] += 1
        self._from_states.add(from_state)
        self._to_states.add(to_state)

    def add_transitions_from(self, pairs: list[tuple[str, str]]) -> None:
        for from_state, to_state in pairs:
            self.add_transition(from_state, to_state)

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str]]) -> "TransitionGraph":
        g = cls()
        g.add_transitions_from(pairs)
        return g

    # --- Properties ---

    @property
    def from_states(self) -> list[str]:
        return sorted(self._from_states)

    @property
    def to_states(self) -> list[str]:
        return sorted(self._to_states)

    # --- Matrices ---

    def get_count_matrix(self) -> pd.DataFrame:
        matrix = pd.DataFrame(0, index=self.from_states, columns=self.to_states)
        for fs, transitions in self._counts.items():
            for ts, cnt in transitions.items():
                matrix.loc[fs, ts] = cnt
        return matrix

    def get_probability_matrix(self) -> pd.DataFrame:
        """Row-normalised transition probability matrix"""
        counts = self.get_count_matrix().astype(float)
        row_sums = counts.sum(axis=1)
        return counts.div(row_sums, axis=0).fillna(0.0)

    # --- Prediction ---

    def predict(self, from_state: str) -> dict[str, float]:
        """Probability distribution for a single given state"""
        if from_state not in self._counts:
            # Uniform distribution for an unknown state
            n = len(self._to_states)
            return {s: 1.0 / n for s in self._to_states}
        total = sum(self._counts[from_state].values())
        return {ts: cnt / total for ts, cnt in self._counts[from_state].items()}

    def predict_distribution(self, from_dist: dict[str, float]) -> dict[str, float]:
        """
        Accepts a probability distribution over from_states,
        returns a distribution over to_states.
        This method eliminates duplication when chaining graphs.
        """
        result: defaultdict[str, float] = defaultdict(float)
        for from_state, prob in from_dist.items():
            if prob == 0.0:
                continue
            for to_state, trans_prob in self.predict(from_state).items():
                result[to_state] += prob * trans_prob
        return dict(result)
