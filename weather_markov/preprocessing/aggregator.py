import pandas as pd


class DecadeAggregator:
    """
    Aggregates daily temperatures into decade averages.
    Decade: days 1-10 (№1), 11-20 (№2), 21-end of month (№3).
    """

    @staticmethod
    def get_decade_number(day: int) -> int:
        if day <= 10:
            return 1
        if day <= 20:
            return 2
        return 3

    def aggregate(
        self, df: pd.DataFrame, date_col: str = "date", temp_col: str = "temperature"
    ) -> pd.DataFrame:
        """
        Input:  date, temperature
        Output: year, month, decade, avg_temperature
        """
        df = df.copy()
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["decade"] = df[date_col].dt.day.map(self.get_decade_number)

        return (
            df.groupby(["year", "month", "decade"])[temp_col]
            .mean()
            .reset_index()
            .rename(columns={temp_col: "avg_temperature"})
        )
