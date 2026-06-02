# Central config — all magic numbers live here, nowhere else
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data
    TICKER: str = "AAPL"
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2024-12-31"
    LOOKBACK: int = 60          # days of history per sample
    TARGET_COL: str = "close"

    # Features (computed by DataPipeline) - must match scaler's feature_names_in_
    # Scaler was trained with: ['SMA_20' 'SMA_50' 'EMA_12'] (in that order)
    FEATURES: List[str] = field(default_factory=lambda: [
        "SMA_20", "SMA_50", "EMA_12"
    ])

    # Train/val/test split (chronological — never shuffle time series)
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    # test = remaining 15%

    # Model
    HIDDEN_SIZE_1: int = 128
    HIDDEN_SIZE_2: int = 64
    DROPOUT: float = 0.2
    NUM_LAYERS: int = 2

    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    LR: float = 0.001
    EARLY_STOP_PATIENCE: int = 10
    LR_PATIENCE: int = 5
    LR_FACTOR: float = 0.5

    # Paths
    MODEL_PATH: str = "models/lstm_model.pt"
    SCALER_PATH: str = "models/scaler.pkl"


CONFIG = Config()
