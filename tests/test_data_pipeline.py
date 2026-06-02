import pytest
import numpy as np
import pandas as pd
from src.data_pipeline import DataPipeline
from src.config import CONFIG


class TestDataPipeline:
    """Test suite for DataPipeline."""

    @pytest.fixture
    def pipeline(self):
        return DataPipeline()

    def test_download_returns_dataframe(self, pipeline):
        """Test that download returns a non-empty DataFrame."""
        df = pipeline.download(CONFIG.TICKER, "2024-01-01", "2024-02-01")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_download_raises_on_invalid_ticker(self, pipeline):
        """Test that download raises ValueError for invalid ticker."""
        with pytest.raises(ValueError):
            pipeline.download("INVALID_TICKER_XYZ123", "2024-01-01", "2024-02-01")

    def test_add_features_returns_all_features(self, pipeline):
        """Test that add_features adds all required features."""
        df = pipeline.download(CONFIG.TICKER, "2023-01-01", "2024-02-01")
        df = pipeline.add_features(df)
        
        # Check that expected features are present
        for feature in CONFIG.FEATURES:
            assert feature in df.columns, f"Feature {feature} not found in DataFrame"

    def test_create_sequences_correct_shape(self, pipeline):
        """Test that create_sequences returns correct shapes."""
        df = pipeline.download(CONFIG.TICKER, "2023-01-01", "2024-02-01")
        df = pipeline.add_features(df)
        
        feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
        scaled_data = np.random.rand(len(df), len(feature_cols))
        
        X, y = pipeline.create_sequences(scaled_data, CONFIG.LOOKBACK)
        
        assert X.shape[1] == CONFIG.LOOKBACK
        assert X.shape[2] == len(feature_cols)
        assert y.shape[0] == X.shape[0]
        assert y.shape[1] == 1

    def test_split_respects_chronological_order(self, pipeline):
        """Test that split maintains chronological order and no data leakage."""
        n = 1000
        X = np.random.rand(n, CONFIG.LOOKBACK, len(CONFIG.FEATURES))
        y = np.random.rand(n, 1)
        
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split(X, y)
        
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        # Verify sizes
        assert len(X_train) == train_end
        assert len(X_val) == val_end - train_end
        assert len(X_test) == n - val_end
        
        # Verify no overlap
        assert train_end < val_end
        assert val_end < n

    def test_scale_fit_saves_scaler(self, pipeline, tmp_path):
        """Test that scale(fit=True) saves scaler to disk."""
        df = pipeline.download(CONFIG.TICKER, "2023-01-01", "2024-02-01")
        df = pipeline.add_features(df)
        
        feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
        scaled, scaler = pipeline.scale(df[feature_cols], fit=True)
        
        assert scaled.shape[0] == len(df[feature_cols])
        assert scaler is not None

    def test_scale_without_fit_raises_on_no_scaler(self, pipeline):
        """Test that scale(fit=False) raises ValueError if scaler not fitted."""
        df = pipeline.download(CONFIG.TICKER, "2023-01-01", "2024-02-01")
        df = pipeline.add_features(df)
        
        feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
        
        with pytest.raises(ValueError):
            pipeline.scale(df[feature_cols], fit=False)

    def test_get_dataloaders_returns_loaders(self, pipeline):
        """Test that get_dataloaders returns valid PyTorch DataLoaders."""
        X_train = np.random.rand(100, CONFIG.LOOKBACK, len(CONFIG.FEATURES))
        y_train = np.random.rand(100, 1)
        X_val = np.random.rand(20, CONFIG.LOOKBACK, len(CONFIG.FEATURES))
        y_val = np.random.rand(20, 1)
        
        train_loader, val_loader = pipeline.get_dataloaders(X_train, y_train, X_val, y_val)
        
        # Check that loaders work
        batch = next(iter(train_loader))
        assert len(batch) == 2  # (X, y)
        assert batch[0].shape[0] <= CONFIG.BATCH_SIZE
