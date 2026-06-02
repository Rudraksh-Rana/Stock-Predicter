import logging
from pathlib import Path

import numpy as np
import torch
import joblib
import pandas as pd

from src.config import CONFIG
from src.model import LSTMPredictor
from src.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class Predictor:
    """Load model and make predictions on new data."""

    def __init__(self, model_path: str = CONFIG.MODEL_PATH, scaler_path: str = CONFIG.SCALER_PATH, device: str = "cpu"):
        """
        Initialize predictor with saved model and scaler.
        
        Args:
            model_path: Path to saved model checkpoint
            scaler_path: Path to saved scaler
            device: Device to use (cpu or cuda)
        """
        self.device = device
        
        # Load scaler first so the model can be built to match the saved feature size
        self.scaler = joblib.load(scaler_path)
        input_size = getattr(self.scaler, 'n_features_in_', len(CONFIG.FEATURES))
        if input_size == len(CONFIG.FEATURES):
            logger.warning(
                'Scaler does not expose n_features_in_; using len(CONFIG.FEATURES)=%d as fallback',
                input_size
            )

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size_1=CONFIG.HIDDEN_SIZE_1,
            hidden_size_2=CONFIG.HIDDEN_SIZE_2,
            dropout=CONFIG.DROPOUT
        ).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")

    def predict(self, ticker: str, days_ahead: int = 1) -> dict:
        """
        Make prediction for next day(s).
        
        Args:
            ticker: Stock ticker
            days_ahead: Number of days to predict ahead (currently only 1)
            
        Returns:
            Dictionary with prediction, confidence interval, and metadata
        """
        try:
            pipeline = DataPipeline()
            pipeline.scaler = self.scaler
            
            # Download recent data
            df = pipeline.download(ticker, CONFIG.START_DATE, CONFIG.END_DATE)
            df = pipeline.add_features(df)
            
            # Scale data
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled_data, _ = pipeline.scale(df[feature_cols], fit=False)
            
            # Create sequence from last CONFIG.LOOKBACK days
            X_last = scaled_data[-CONFIG.LOOKBACK:].reshape(1, CONFIG.LOOKBACK, len(feature_cols))
            
            # Make prediction
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_last).to(self.device)
                pred_scaled = self.model(X_tensor).cpu().numpy()
            
            # Inverse transform
            dummy = np.zeros((1, len(feature_cols)))
            dummy[:, 0] = pred_scaled[0, 0]
            pred_unscaled = self.scaler.inverse_transform(dummy)[0, 0]
            
            # Get last actual close price
            last_close = df[CONFIG.TARGET_COL].iloc[-1]
            
            # Simple confidence interval: ± 1.5 * MAE (estimated)
            # In production, this would be from validation metrics
            estimated_mae = abs(pred_unscaled - last_close) * 0.02  # rough estimate
            confidence_lower = pred_unscaled - 1.5 * estimated_mae
            confidence_upper = pred_unscaled + 1.5 * estimated_mae
            
            direction = "up" if pred_unscaled > last_close else "down"
            
            result = {
                'ticker': ticker,
                'predicted_close': float(pred_unscaled),
                'confidence_interval': [float(confidence_lower), float(confidence_upper)],
                'last_actual_close': float(last_close),
                'direction': direction,
                'last_update': str(df.index[-1].date())
            }
            
            logger.info(f"Prediction for {ticker}: ${pred_unscaled:.2f} ({direction})")
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            raise ValueError(f"Failed to predict for {ticker}: {e}")

    def predict_historical(self, ticker: str, lookback_days: int = 30) -> dict:
        """
        Get historical predictions for last N days.
        
        Args:
            ticker: Stock ticker
            lookback_days: Number of past days to predict
            
        Returns:
            Dictionary with historical prices and predictions
        """
        try:
            pipeline = DataPipeline()
            pipeline.scaler = self.scaler
            
            # Download data
            df = pipeline.download(ticker, CONFIG.START_DATE, CONFIG.END_DATE)
            df = pipeline.add_features(df)
            
            # Scale data
            feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
            scaled_data, _ = pipeline.scale(df[feature_cols], fit=False)
            
            # Make predictions for last lookback_days
            actuals = []
            predictions = []
            dates = []
            
            for i in range(max(CONFIG.LOOKBACK, len(df) - lookback_days), len(df) - 1):
                X = scaled_data[i-CONFIG.LOOKBACK:i].reshape(1, CONFIG.LOOKBACK, len(feature_cols))
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    pred_scaled = self.model(X_tensor).cpu().numpy()
                
                # Inverse transform
                dummy = np.zeros((1, len(feature_cols)))
                dummy[:, 0] = pred_scaled[0, 0]
                pred_unscaled = self.scaler.inverse_transform(dummy)[0, 0]
                
                actuals.append(float(df[CONFIG.TARGET_COL].iloc[i]))
                predictions.append(float(pred_unscaled))
                dates.append(str(df.index[i].date()))
            
            return {
                'ticker': ticker,
                'dates': dates,
                'actual': actuals,
                'predictions': predictions
            }
        
        except Exception as e:
            logger.error(f"Historical prediction failed for {ticker}: {e}")
            raise ValueError(f"Failed to get historical predictions for {ticker}: {e}")
