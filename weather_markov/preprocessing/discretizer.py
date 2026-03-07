import numpy as np
import pandas as pd


class TemperatureDiscretizer:
    """
    Converts continuous temperatures into discrete range labels.
    Supports uniform, quantile-based and arbitrary bin splits.
    """

    def __init__(self, bins: list[float], labels: list[str] | None = None):
        self.bins = bins
        self.labels = labels or self._auto_labels(bins)
        self._fitted = False

    # --- Factory methods ---

    @classmethod
    def from_manual(
        cls, bins: list[float], labels: list[str] | None = None
    ) -> "TemperatureDiscretizer":
        return cls(bins, labels)

    @classmethod
    def from_equal_width(
        cls, n_bins: int, t_min: float, t_max: float
    ) -> "TemperatureDiscretizer":
        bins = list(np.linspace(t_min, t_max, n_bins + 1))
        return cls(bins)

    @classmethod
    def from_quantiles(cls, n_bins: int) -> "TemperatureDiscretizer":
        """Bin boundaries are computed during fit() from data quantiles"""
        instance = cls.__new__(cls)
        instance.n_bins = n_bins
        instance.bins = []
        instance.labels = []
        instance._fitted = False
        instance._use_quantiles = True
        return instance

    # --- Core methods ---

    def fit(self, temperatures: pd.Series | np.ndarray) -> "TemperatureDiscretizer":
        if getattr(self, "_use_quantiles", False):
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            self.bins = list(np.percentile(temperatures, quantiles))
            self.labels = self._auto_labels(self.bins)
        self._fitted = True
        return self

    def transform(self, temperatures: pd.Series | np.ndarray) -> pd.Series:
        series = (
            pd.Series(temperatures)
            if not isinstance(temperatures, pd.Series)
            else temperatures
        )
        return pd.cut(
            series, bins=self.bins, labels=self.labels, include_lowest=True
        ).astype(str)

    def fit_transform(self, temperatures) -> pd.Series:
        return self.fit(temperatures).transform(temperatures)

    @staticmethod
    def _auto_labels(bins: list[float]) -> list[str]:
        return [f"({bins[i]:.1f}, {bins[i+1]:.1f}]" for i in range(len(bins) - 1)]
