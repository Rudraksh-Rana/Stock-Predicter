import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
from src.config import CONFIG
from src.model import LSTMPredictor, count_parameters

logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Trainer:
    """Training loop for LSTM model."""

    def __init__(self, input_size: int, device: str = "cpu"):
        """
        Initialize trainer with model, optimizer, and loss function.
        
        Args:
            input_size: Number of input features
            device: Device to use (cpu or cuda)
        """
        self.device = device
        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size_1=CONFIG.HIDDEN_SIZE_1,
            hidden_size_2=CONFIG.HIDDEN_SIZE_2,
            dropout=CONFIG.DROPOUT
        ).to(device)
        
        self.optimizer = Adam(self.model.parameters(), lr=CONFIG.LR)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=CONFIG.LR_FACTOR,
            patience=CONFIG.LR_PATIENCE
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Count parameters
        count_parameters(self.model)

        # Start MLflow run
        if HAS_MLFLOW:
            mlflow.start_run()
            mlflow.log_params({
                'hidden_size_1': CONFIG.HIDDEN_SIZE_1,
                'hidden_size_2': CONFIG.HIDDEN_SIZE_2,
                'dropout': CONFIG.DROPOUT,
                'batch_size': CONFIG.BATCH_SIZE,
                'lr': CONFIG.LR,
                'lookback': CONFIG.LOOKBACK,
            })

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Run one training epoch.
        
        Args:
            loader: Training DataLoader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        return avg_loss

    def val_epoch(self, loader: DataLoader) -> float:
        """
        Run one validation epoch.
        
        Args:
            loader: Validation DataLoader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Train model with early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        """
        logger.info(f"Starting training for {CONFIG.EPOCHS} epochs")
        
        for epoch in range(CONFIG.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            if HAS_MLFLOW:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", current_lr, step=epoch)

            # Print progress
            print(f"Epoch {epoch+1}/{CONFIG.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            # Save checkpoint if val loss improves
            if val_loss < self.best_val_loss:
                logger.info(f"Val loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save(CONFIG.MODEL_PATH)
            else:
                self.patience_counter += 1
                logger.warning(f"No improvement for {self.patience_counter}/{CONFIG.EARLY_STOP_PATIENCE} epochs")
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if self.patience_counter >= CONFIG.EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        logger.info("Training completed")

    def save(self, path: str) -> None:
        """
        Save model and config to disk.
        
        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': CONFIG.__dict__,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
