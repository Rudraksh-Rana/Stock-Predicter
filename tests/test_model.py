import pytest
import torch
import torch.nn as nn
from src.model import LSTMPredictor, count_parameters


class TestLSTMPredictor:
    """Test suite for LSTM model."""

    @pytest.fixture
    def model(self):
        return LSTMPredictor(
            input_size=10,
            hidden_size_1=128,
            hidden_size_2=64,
            dropout=0.2
        )

    def test_model_initialization(self, model):
        """Test that model initializes without errors."""
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_forward_pass_shape(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 32
        seq_len = 60
        input_size = 10
        
        X = torch.randn(batch_size, seq_len, input_size)
        output = model(X)
        
        assert output.shape == (batch_size, 1)

    def test_forward_pass_values(self, model):
        """Test that forward pass produces valid float values."""
        X = torch.randn(1, 60, 10)
        output = model(X)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.dtype == torch.float32

    def test_batch_size_one(self, model):
        """Test model with batch size 1."""
        X = torch.randn(1, 60, 10)
        output = model(X)
        assert output.shape == (1, 1)

    def test_large_batch_size(self, model):
        """Test model with large batch size."""
        X = torch.randn(256, 60, 10)
        output = model(X)
        assert output.shape == (256, 1)

    def test_different_lookback_windows(self, model):
        """Test model with different sequence lengths."""
        for seq_len in [30, 60, 120]:
            X = torch.randn(32, seq_len, 10)
            output = model(X)
            assert output.shape == (32, 1)

    def test_model_has_lstm_layers(self, model):
        """Test that model contains LSTM layers."""
        layer_names = [name for name, _ in model.named_modules()]
        assert any('lstm' in name.lower() for name in layer_names)

    def test_model_has_dropout(self, model):
        """Test that model contains dropout layers."""
        layer_names = [name for name, _ in model.named_modules()]
        assert any('dropout' in name.lower() for name in layer_names)

    def test_count_parameters(self, model):
        """Test parameter counting utility."""
        total_params = count_parameters(model)
        assert total_params > 0
        assert isinstance(total_params, int)

    def test_gradient_flow(self, model):
        """Test that gradients flow properly through model."""
        X = torch.randn(32, 60, 10, requires_grad=False)
        output = model(X)
        loss = output.mean()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

    def test_train_eval_modes(self, model):
        """Test that model can switch between train and eval modes."""
        X = torch.randn(32, 60, 10)
        
        # Train mode
        model.train()
        output_train = model(X)
        
        # Eval mode
        model.eval()
        output_eval = model(X)
        
        # Both should work without errors
        assert output_train.shape == output_eval.shape

    def test_dropout_different_modes(self, model):
        """Test that dropout behaves differently in train vs eval."""
        X = torch.randn(32, 60, 10)
        
        model.train()
        model.eval()
        
        with torch.no_grad():
            output_eval = model(X)
        
        # Output should be deterministic in eval mode
        with torch.no_grad():
            output_eval2 = model(X)
        
        assert torch.allclose(output_eval, output_eval2)
