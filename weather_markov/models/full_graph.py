from itertools import chain as iterchain

import pandas as pd

from weather_markov.markov.chain import MarkovChain
from weather_markov.markov.graph import TransitionGraph
from weather_markov.models.decade_base import DecadeBasedPredictor


class FullGraphMarkovPredictor(DecadeBasedPredictor):
    """
    Method 2: a single shared transition graph trained on all pairs
    of adjacent decades (February–May), applied n_steps times.

    Assumes stationarity: transition probabilities do not depend
    on the specific month or decade.
    """

    def __init__(self, discretizer, months=None):
        super().__init__(discretizer, months)
        self.shared_graph: TransitionGraph | None = None
        self.n_steps: int = 0

    def fit(self, data: pd.DataFrame) -> "FullGraphMarkovPredictor":
        pairs_per_transition = self._build_transition_pairs(data)

        all_pairs = list(iterchain.from_iterable(pairs_per_transition.values()))
        self.shared_graph = TransitionGraph.from_pairs(all_pairs)
        self.n_steps = len(
            pairs_per_transition
        )  # number of transitions from Feb to May
        self._is_fitted = True
        return self

    def predict(self, state: str) -> dict[str, float]:
        """
        current_decades: decade data for the current year
                         (year, month, decade, avg_temperature)
        Predicts the average May temperature.
        """
        self._check_fitted()

        chain = MarkovChain.from_single_graph(self.shared_graph, self.n_steps)
        return chain.predict(state)
