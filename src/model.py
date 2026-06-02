import logging
import torch
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """Stacked LSTM model for stock price prediction."""

    def __init__(self, input_size: int, hidden_size_1: int, hidden_size_2: int, dropout: float):
        """
        Initialize LSTM architecture.
        
        Args:
            input_size: Number of input features
            hidden_size_1: Hidden size of first LSTM layer
            hidden_size_2: Hidden size of second LSTM layer
            dropout: Dropout probability
        """
        super(LSTMPredictor, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size_2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Prediction tensor (batch_size, 1)
        """
        # LSTM 1: (batch, seq_len, input_size) -> (batch, seq_len, hidden_size_1)
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # LSTM 2: (batch, seq_len, hidden_size_1) -> (batch, seq_len, hidden_size_2)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Take last timestep: (batch, seq_len, hidden_size_2) -> (batch, hidden_size_2)
        last_output = lstm_out2[:, -1, :]
        
        # FC layer: (batch, hidden_size_2) -> (batch, 1)
        output = self.fc(last_output)
        
        return output


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total:,}")
    return total
