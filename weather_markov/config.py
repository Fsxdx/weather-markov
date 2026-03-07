from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiscretizerConfig:
    bins: list[float]  # Temperature range boundaries
    labels: list[str] | None = None  # Range labels; auto-generated if None


@dataclass
class ParserConfig:
    base_url: str
    location: str
    year_start: int
    year_end: int
    months: list[int] = field(default_factory=lambda: [2, 3, 4, 5])


@dataclass
class PredictionConfig:
    months: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    target_month: int = 5


@dataclass
class AppConfig:
    parser: ParserConfig
    discretizer: DiscretizerConfig
    prediction: PredictionConfig
    data_dir: Path = Path("data")


def load_config(path: str = "config.yaml") -> AppConfig: ...
