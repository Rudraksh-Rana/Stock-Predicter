import logging
from pathlib import Path
from typing import Tuple
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import joblib

from src.config import CONFIG

logger = logging.getLogger(__name__)

# Common Indian stock short-name → Yahoo Finance ticker mapping
_INDIAN_TICKER_ALIASES: dict[str, str] = {
    # Reliance
    "RIL": "RELIANCE.NS",
    # Infosys
    "INFY": "INFY.NS",
    # TCS
    "TCS": "TCS.NS",
    # HDFC Bank
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC": "HDFCBANK.NS",
    # ICICI Bank
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI": "ICICIBANK.NS",
    # State Bank of India
    "SBI": "SBIN.NS",
    "SBIN": "SBIN.NS",
    # Wipro
    "WIPRO": "WIPRO.NS",
    # HCL Technologies
    "HCLTECH": "HCLTECH.NS",
    # Bajaj Finance
    "BAJFINANCE": "BAJFINANCE.NS",
    # Bharti Airtel
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    # Asian Paints
    "ASIANPAINT": "ASIANPAINT.NS",
    # Axis Bank
    "AXISBANK": "AXISBANK.NS",
    # Kotak Mahindra Bank
    "KOTAKBANK": "KOTAKBANK.NS",
    "KOTAK": "KOTAKBANK.NS",
    # Larsen & Toubro
    "LT": "LT.NS",
    # Maruti Suzuki
    "MARUTI": "MARUTI.NS",
    # Hindustan Unilever
    "HUL": "HINDUNILVR.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    # ITC
    "ITC": "ITC.NS",
    # Sun Pharmaceutical
    "SUNPHARMA": "SUNPHARMA.NS",
    # ONGC
    "ONGC": "ONGC.NS",
    # NTPC
    "NTPC": "NTPC.NS",
    # Power Grid
    "POWERGRID": "POWERGRID.NS",
    # Adani Enterprises
    "ADANIENT": "ADANIENT.NS",
    "ADANI": "ADANIENT.NS",
    # Nifty 50 ETF
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
}


def _resolve_ticker(ticker: str) -> str:
    """
    Resolve a ticker to a valid Yahoo Finance symbol.

    Resolution order:
    1. Check the alias map for popular Indian short names.
    2. Try the ticker as-is.
    3. Try <ticker>.NS  (NSE India).
    4. Try <ticker>.BO  (BSE India).

    Returns the first symbol that actually returns data, or the
    original ticker if none succeed (letting the caller raise).
    """
    upper = ticker.upper()

    # 1. Alias map
    if upper in _INDIAN_TICKER_ALIASES:
        resolved = _INDIAN_TICKER_ALIASES[upper]
        logger.info(f"Ticker alias: {ticker!r} → {resolved!r}")
        return resolved

    # 2-4. Probe candidates
    candidates = [ticker, f"{ticker}.NS", f"{ticker}.BO"]
    for candidate in candidates:
        try:
            probe = yf.download(candidate, period="5d", auto_adjust=True, progress=False)
            if not probe.empty:
                if candidate != ticker:
                    logger.info(f"Ticker resolved: {ticker!r} → {candidate!r}")
                return candidate
        except Exception:
            continue

    # Fall back to original — download() will raise a clear error
    logger.warning(f"Could not auto-resolve ticker {ticker!r}; falling back to original symbol")
    return ticker


class DataPipeline:
    """Download, engineer features, scale, and create sequences for LSTM training."""

    def __init__(self):
        self.scaler = None

    def download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            ValueError: If download fails or returns empty data
        """
        # Resolve ticker symbol (handles Indian short names and .NS/.BO suffixes)
        ticker = _resolve_ticker(ticker)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {ticker} data from {start} to {end}")
                df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

                if df.empty:
                    raise ValueError(f"No data downloaded for {ticker}")

                # Ensure it's a DataFrame and not a Series
                if isinstance(df, pd.Series):
                    df = df.to_frame()

                # Handle MultiIndex columns (when downloading multiple tickers)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                # Lowercase column names for consistency
                df.columns = [str(col).lower() for col in df.columns]

                # Drop NaN rows in OHLCV columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in required_cols if col in df.columns]
                if available_cols:
                    df = df.dropna(subset=available_cols)
                else:
                    raise ValueError(f"Missing OHLCV columns. Got columns: {list(df.columns)}")

                if df.empty:
                    raise ValueError(f"All data for {ticker} was NaN after cleaning")

                logger.info(f"Downloaded {len(df)} rows for {ticker}")
                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {ticker} after {max_retries} attempts: {e}")
                    raise ValueError(f"Failed to download {ticker}: {e}")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using numpy/pandas (no external TA library).

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with added features
        """
        logger.info("Adding technical indicators")
        df = df.copy()

        # Simple Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Average
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['BBL'] = sma20 - (std20 * 2)
        df['BBU'] = sma20 + (std20 * 2)

        # Drop first 50 rows (NaN from indicator warm-up)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values from indicator calculations")

        return df


    def scale(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Scale data using MinMaxScaler.
        
        Args:
            df: DataFrame with features
            fit: If True, fit scaler on this data. If False, use existing scaler.
            
        Returns:
            Scaled numpy array and scaler object
        """
        if fit:
            logger.info("Fitting MinMaxScaler on training data")
            self.scaler = MinMaxScaler()
            # Select only feature columns (exclude index)
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled = self.scaler.fit_transform(df[feature_cols])
            joblib.dump(self.scaler, CONFIG.SCALER_PATH)
            logger.info(f"Scaler saved to {CONFIG.SCALER_PATH}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call scale(fit=True) first.")
            logger.info("Scaling data with existing scaler")
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled = self.scaler.transform(df[feature_cols])
        
        return scaled, self.scaler

    def create_sequences(self, scaled_data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM.
        
        Args:
            scaled_data: Scaled feature array (n_samples, n_features)
            lookback: Number of timesteps to look back
            
        Returns:
            X: (n_samples, lookback, n_features), y: (n_samples, 1)
        """
        logger.info(f"Creating sequences with lookback={lookback}")
        
        X, y = [], []
        # Close price is first feature (index 0)
        close_col_idx = 0
        
        for i in range(len(scaled_data) - lookback):
            X.append(scaled_data[i:i + lookback])
            y.append(scaled_data[i + lookback, close_col_idx])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        logger.info(f"Created {len(X)} sequences. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into train/val/test chronologically (no shuffling).
        
        Args:
            X: Feature sequences
            y: Target values
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n = len(X)
        train_end = int(n * CONFIG.TRAIN_RATIO)
        val_end = int(n * (CONFIG.TRAIN_RATIO + CONFIG.VAL_RATIO))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Verify no data leakage
        assert train_end < val_end < len(X), "Data leakage detected in split"
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_dataloaders(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            train_loader, val_loader
        """
        logger.info("Creating PyTorch DataLoaders")
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
