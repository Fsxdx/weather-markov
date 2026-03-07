from abc import ABC

import pandas as pd

from weather_markov.models.base import BaseWeatherPredictor


class DecadeBasedPredictor(BaseWeatherPredictor, ABC):
    """Shared logic for methods 2 and 3, which operate on decade sequences"""

    def __init__(self, discretizer, months: list[int] | None = None):
        super().__init__(discretizer)
        self.months = months or [2, 3, 4, 5]

    def _get_ordered_decades(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filters the required months and sorts by (year, month, decade)"""
        return (
            data[data["month"].isin(self.months)]
            .sort_values(["year", "month", "decade"])
            .reset_index(drop=True)
        )

    def _get_decade_labels(self, data: pd.DataFrame) -> list[tuple[int, int]]:
        """Returns unique (month, decade) pairs in chronological order"""
        ordered = self._get_ordered_decades(data)
        return ordered.groupby(["month", "decade"], sort=False).first().index.tolist()

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
        labels = self._get_decade_labels(data)
        pairs_per_transition: dict = {}

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
            from_states = self.discretizer.transform(from_data[common_years])
            to_states = self.discretizer.transform(to_data[common_years])

            pairs_per_transition[(from_label, to_label)] = list(
                zip(from_states, to_states)
            )

        return pairs_per_transition
