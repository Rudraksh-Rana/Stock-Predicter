import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation metrics and visualization."""

    @staticmethod
    def inverse_transform(predictions: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale.
        
        Args:
            predictions: Scaled predictions (n_samples, 1)
            scaler: Fitted MinMaxScaler
            
        Returns:
            Unscaled predictions
        """
        # Create dummy array with same shape as training data (n_samples, n_features)
        n_features = scaler.n_features_in_
        dummy = np.zeros((predictions.shape[0], n_features))
        dummy[:, 0] = predictions.flatten()  # Close price is first feature
        
        unscaled = scaler.inverse_transform(dummy)
        return unscaled[:, 0].reshape(-1, 1)

    @staticmethod
    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with MAE, RMSE, MAPE, directional accuracy
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE with protection against division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # Directional accuracy: % of days where predicted direction == actual direction
        actual_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)
        
        directional_acc = np.mean((actual_direction * pred_direction) > 0) * 100
        
        metrics_dict = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_acc
        }
        
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)
        logger.info(f"MAE:                   {mae:.6f}")
        logger.info(f"RMSE:                  {rmse:.6f}")
        logger.info(f"MAPE:                  {mape:.2f}%")
        logger.info(f"Directional Accuracy:  {directional_acc:.2f}%")
        logger.info("=" * 60)
        
        if mape > 5:
            logger.warning("MAPE > 5%: Consider checking for data leakage or NaN values")
        
        return metrics_dict

    @staticmethod
    def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, risk_free: float = 0.0) -> float:
        """
        Calculate Sharpe ratio based on trading strategy.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            risk_free: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Trading strategy: buy if pred > true (expect up), else sell (expect down)
        signal = (y_pred > y_true).astype(int) * 2 - 1  # -1 or 1
        
        # Daily returns
        actual_returns = np.diff(y_true) / y_true[:-1]
        strategy_returns = signal[:-1] * actual_returns
        
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns) + 1e-8
        
        sharpe = (mean_return - risk_free) / std_return * np.sqrt(252)
        
        logger.info(f"Sharpe Ratio (252-day):  {sharpe:.4f}")
        
        return sharpe

    @staticmethod
    def plot_results(y_true: np.ndarray, y_pred: np.ndarray, ticker: str, save_path: str = "models/plots") -> None:
        """
        Plot actual vs predicted and residuals.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            ticker: Stock ticker
            save_path: Path to save plots
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Calculate MAPE for title
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        # Plot 1: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true, label='Actual', linewidth=2)
        ax.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Price ($)')
        ax.set_title(f'{ticker} Actual vs Predicted (MAPE: {mape:.2f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{save_path}/actual_vs_predicted.png', dpi=150)
        logger.info(f"Saved plot: {save_path}/actual_vs_predicted.png")
        plt.close(fig)
        
        # Plot 2: Residuals
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(residuals, color='red', linewidth=1, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.fill_between(range(len(residuals)), residuals, 0, alpha=0.3, color='red')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Residual ($)')
        ax.set_title(f'{ticker} Prediction Residuals')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{save_path}/residuals.png', dpi=150)
        logger.info(f"Saved plot: {save_path}/residuals.png")
        plt.close(fig)
