from itertools import chain as iterchain

import pandas as pd

from weather_markov.markov.chain import MarkovChain
from weather_markov.markov.layered_graph import LayeredTransitionGraph
from weather_markov.models.decade_base import DecadeBasedPredictor


class TwoLayerGraphMarkovPredictor(DecadeBasedPredictor):
    """
    Method 2: a single shared transition graph trained on all pairs
    of adjacent decades (February–May), applied n_steps times.

    Assumes stationarity: transition probabilities do not depend
    on the specific month or decade.
    """

    def __init__(self, discretizer, months=None):
        super().__init__(discretizer, [2, 5])
        self.shared_graph: LayeredTransitionGraph | None = None
        self.n_steps: int = 0

    def _build_transition_pairs(
        self, data: pd.DataFrame
    ) -> dict[tuple[tuple, tuple], list[tuple[str, str]]]:
        """
        Builds transition pairs grouped by transition type.
        Key:   ((from_month, from_decade), (to_month, to_decade))
        Value: list of (from_state, to_state) across all years

        Method 2 merges all pairs into a single set.
        Method 3 uses the pairs from each key separately.
        """
        ordered = self._get_ordered_decades(data)
        labels = [label for label in self._get_decade_labels(data) if label[1] == 1]
        pairs_per_transition: dict = {}
        layers = list()

        for i in range(len(labels) - 1):
            from_label, to_label = labels[i], labels[i + 1]
            from_m, from_d = from_label
            to_m, to_d = to_label

            from_data = ordered[
                (ordered["month"] == from_m) & (ordered["decade"] == from_d)
            ].set_index("year")["avg_temperature"]
            to_data = ordered[
                (ordered["month"] == to_m) & (ordered["decade"] == to_d)
            ].set_index("year")["avg_temperature"]

            common_years = from_data.index.intersection(to_data.index)
            from_states = f"{from_m}" + self.discretizer.transform(
                from_data[common_years]
            )
            to_states = f"{to_m}" + self.discretizer.transform(to_data[common_years])

            layers.append(from_states.to_list())
            layers.append(to_states.to_list())

            pairs_per_transition[(from_label, to_label)] = list(
                zip(from_states, to_states)
            )

        return pairs_per_transition, layers

    def fit(self, data: pd.DataFrame) -> "TwoLayerGraphMarkovPredictor":
        pairs_per_transition, layers = self._build_transition_pairs(data)

        all_pairs = list(iterchain.from_iterable(pairs_per_transition.values()))
        self.shared_graph = LayeredTransitionGraph.from_pairs(all_pairs, layers)
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
