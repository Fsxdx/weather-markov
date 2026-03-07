from abc import ABC, abstractmethod

import pandas as pd

from weather_markov.preprocessing.discretizer import TemperatureDiscretizer


class BaseWeatherPredictor(ABC):

    def __init__(self, discretizer: TemperatureDiscretizer):
        self.discretizer = discretizer
        self._is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseWeatherPredictor":
        """
        data: DataFrame with columns [year, month, decade, avg_temperature]
        """

    @abstractmethod
    def predict(self, input_data) -> dict[str, float]:
        """Returns a probability distribution over temperature range labels"""

    def predict_label(self, input_data) -> str:
        dist = self.predict(input_data)
        return max(dist, key=dist.get)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
