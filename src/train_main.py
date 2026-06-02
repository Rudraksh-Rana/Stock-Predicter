"""
Main training script for LSTM stock predictor.

Run with: python src/train.py
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.config import CONFIG
from src.data_pipeline import DataPipeline
from src.train import Trainer
from src.evaluate import Evaluator
import json


def main():
    """Execute full training pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("LSTM STOCK PRICE PREDICTOR - TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Data Pipeline
        logger.info("\n[1/5] INITIALIZING DATA PIPELINE...")
        pipeline = DataPipeline()
        
        # Download data
        logger.info(f"Downloading {CONFIG.TICKER} data from {CONFIG.START_DATE} to {CONFIG.END_DATE}")
        df = pipeline.download(CONFIG.TICKER, CONFIG.START_DATE, CONFIG.END_DATE)
        logger.info(f"Downloaded {len(df)} rows")
        
        # Add features
        logger.info("\nEngineering technical indicators...")
        df = pipeline.add_features(df)
        logger.info(f"Features: {CONFIG.FEATURES}")
        
        # Scale data (fit on all data first for feature engineering demo)
        logger.info("\nScaling features...")
        feature_cols = [col for col in CONFIG.FEATURES if col in df.columns]
        scaled_data, scaler = pipeline.scale(df[feature_cols], fit=True)
        
        # Create sequences
        logger.info("\nCreating sequences...")
        X, y = pipeline.create_sequences(scaled_data, CONFIG.LOOKBACK)
        
        # Split chronologically
        logger.info("\nSplitting data (chronological, no shuffling)...")
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split(X, y)
        
        # Create dataloaders
        logger.info("\nCreating PyTorch DataLoaders...")
        train_loader, val_loader = pipeline.get_dataloaders(X_train, y_train, X_val, y_val)
        
        # Step 2: Train Model
        logger.info("\n[2/5] INITIALIZING AND TRAINING MODEL...")
        trainer = Trainer(input_size=len(feature_cols))
        trainer.fit(train_loader, val_loader)
        
        # Step 3: Evaluate
        logger.info("\n[3/5] EVALUATING ON TEST SET...")
        trainer.model.eval()
        
        import torch
        X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
        with torch.no_grad():
            y_pred_scaled = trainer.model(X_test_tensor).cpu().numpy()
        
        # Inverse transform
        import numpy as np
        y_pred = Evaluator.inverse_transform(y_pred_scaled, scaler)
        y_test_unscaled = Evaluator.inverse_transform(y_test, scaler)
        
        # Calculate metrics
        logger.info("\nCalculating evaluation metrics...")
        metrics = Evaluator.metrics(y_test_unscaled, y_pred)
        sharpe = Evaluator.sharpe_ratio(y_test_unscaled, y_pred)
        
        # Save metrics to JSON
        metrics['Sharpe_Ratio'] = sharpe
        metrics_file = "models/metrics.json"
        Path("models").mkdir(exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Plot results
        logger.info("\nGenerating evaluation plots...")
        Evaluator.plot_results(y_test_unscaled, y_pred, CONFIG.TICKER)

        # Step 4: Log to MLflow
        logger.info("\n[4/5] LOGGING TO MLFLOW...")
        try:
            import mlflow
            mlflow.log_metrics(metrics)
            mlflow.log_metric("sharpe_ratio", sharpe)
            mlflow.end_run()
            logger.info("Metrics logged to MLflow")
        except ImportError:
            logger.info("MLflow not available, skipping metrics logging")

        # Step 5: Final summary
        logger.info("\n[5/5] TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {CONFIG.MODEL_PATH}")
        logger.info(f"Scaler saved to: {CONFIG.SCALER_PATH}")
        logger.info(f"Metrics saved to: {metrics_file}")
        logger.info(f"Plots saved to: models/plots/")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. View MLflow: mlflow ui")
        logger.info("2. Start API: uvicorn api.main:app --reload")
        logger.info("3. Launch dashboard: streamlit run app/streamlit_app.py")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
