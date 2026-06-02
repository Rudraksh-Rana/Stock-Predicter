import logging
from datetime import datetime
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import CONFIG
from src.predict import Predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LSTM Stock Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        predictor = Predictor(CONFIG.MODEL_PATH, CONFIG.SCALER_PATH, device="cpu")
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests."""
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    return response


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "lstm",
        "ticker": CONFIG.TICKER
    }


@app.api_route("/predict", methods=["GET", "POST"])
async def predict(ticker: str = CONFIG.TICKER, days_ahead: int = 1):
    """
    Make a stock price prediction.
    
    Args:
        ticker: Stock ticker symbol
        days_ahead: Number of days ahead to predict (currently 1)
        
    Returns:
        Prediction with confidence interval and metadata
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = predictor.predict(ticker, days_ahead)
        result['generated_at'] = datetime.now().isoformat()
        
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get last recorded metrics from validation.
    
    Returns:
        Dictionary with MAE, RMSE, MAPE, Sharpe ratio
    """
    metrics_file = "models/metrics.json"
    
    try:
        import json
        if Path(metrics_file).exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return {
                "status": "no metrics available",
                "message": "Model hasn't been evaluated yet"
            }
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{ticker}")
async def get_history(ticker: str, lookback_days: int = 30):
    """
    Get historical actual and predicted prices.
    
    Args:
        ticker: Stock ticker symbol
        lookback_days: Number of past days to retrieve
        
    Returns:
        Historical data with actual and predicted prices
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = predictor.predict_historical(ticker, lookback_days)
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
