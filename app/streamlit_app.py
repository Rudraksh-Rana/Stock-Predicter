import logging
from datetime import datetime, timedelta

import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_URL = "http://localhost:8000"

st.title("📈 LSTM Stock Price Predictor")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    predict_button = st.button("🔮 Run Prediction", use_container_width=True)

# Main area
if predict_button or True:  # Always show if page loads
    try:
        # Fetch prediction
        with st.spinner("📡 Fetching prediction from API..."):
            try:
                response = httpx.post(
                    f"{API_URL}/predict",
                    params={"ticker": ticker, "days_ahead": 1},
                    timeout=30.0
                )
                response.raise_for_status()
                pred_data = response.json()
            except httpx.ConnectError:
                st.error("❌ Cannot connect to API. Make sure the API server is running on http://localhost:8000")
                st.info("Start the API with: `uvicorn api.main:app --reload`")
                st.stop()
            except httpx.HTTPStatusError as e:
                st.error(f"❌ API Error: {e.response.status_code} - {e.response.text}")
                st.stop()
        
        # Calculate buy/sell recommendation and profit/loss
        current_price = pred_data['last_actual_close']
        predicted_price = pred_data['predicted_close']
        price_change = predicted_price - current_price
        pnl_percentage = (price_change / current_price) * 100
        
        # Generate recommendation
        if pred_data['direction'].upper() == 'UP' and pnl_percentage > 0:
            recommendation = "BUY"
            rec_color = "green"
        else:
            recommendation = "SELL"
            rec_color = "red"
        
        # Display metric cards
        st.markdown("### 📊 Key Metrics")
        cols = st.columns(5)
        
        with cols[0]:
            st.metric(
                "Last Close",
                f"${current_price:.2f}",
                delta=None
            )
        
        with cols[1]:
            st.metric(
                "Predicted Close",
                f"${predicted_price:.2f}",
                delta=f"${price_change:.2f}",
                delta_color="inverse"
            )
        
        with cols[2]:
            st.metric(
                "Direction",
                pred_data['direction'].upper(),
                delta=None
            )
        
        with cols[3]:
            ci_lower, ci_upper = pred_data['confidence_interval']
            ci_range = ci_upper - ci_lower
            st.metric(
                "Confidence Interval",
                f"±${ci_range/2:.2f}",
                delta=None
            )
        
        with cols[4]:
            st.metric(
                "P&L %",
                f"{pnl_percentage:.2f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # Display recommendation box — always show BOTH profit & loss scenarios
        st.markdown("### 🎯 Trading Recommendation")

        rec_col1, rec_col2 = st.columns(2)

        # ── Profit (BUY) scenario ─────────────────────────────────────────────
        profit_per_share = abs(price_change) if price_change > 0 else abs(price_change)
        profit_pct       = abs(pnl_percentage)
        buy_target       = max(current_price, predicted_price)
        loss_per_share   = abs(price_change)
        loss_pct         = abs(pnl_percentage)
        sell_target      = min(current_price, predicted_price)

        with rec_col1:
            if recommendation == "BUY":
                st.success(f"""
### ✅ BUY  ← **MODEL RECOMMENDS THIS**
| Field | Value |
|---|---|
| **Signal** | Price predicted to go UP |
| **Entry Price** | ${current_price:.2f} |
| **Target Price** | ${buy_target:.2f} |
| **Profit / Share** | +${profit_per_share:.2f} |
| **Expected Return** | +{profit_pct:.2f}% |
| **CI Range** | ${pred_data['confidence_interval'][0]:.2f} – ${pred_data['confidence_interval'][1]:.2f} |

> Buy now and sell at predicted target to capture the upside.
""")
            else:
                st.info(f"""
### 📈 BUY  ← *Alternative / Contrarian*
| Field | Value |
|---|---|
| **Signal** | Price predicted to go DOWN |
| **Entry Price** | ${current_price:.2f} |
| **Upside Target** | ${pred_data['confidence_interval'][1]:.2f} (CI upper) |
| **Max Upside / Share** | +${pred_data['confidence_interval'][1] - current_price:.2f} |
| **Max Upside %** | +{((pred_data['confidence_interval'][1] - current_price) / current_price * 100):.2f}% |
| **Risk** | High — model predicts DOWN move |

> Buying against the model signal is HIGH risk. Only for contrarian bets.
""")

        # ── Loss / Risk (SELL) scenario ───────────────────────────────────────
        with rec_col2:
            if recommendation == "SELL":
                st.error(f"""
### ❌ SELL  ← **MODEL RECOMMENDS THIS**
| Field | Value |
|---|---|
| **Signal** | Price predicted to go DOWN |
| **Entry Price** | ${current_price:.2f} |
| **Predicted Price** | ${sell_target:.2f} |
| **Loss / Share (if held)** | −${loss_per_share:.2f} |
| **Expected Loss %** | −{loss_pct:.2f}% |
| **CI Range** | ${pred_data['confidence_interval'][0]:.2f} – ${pred_data['confidence_interval'][1]:.2f} |

> Sell or short now to avoid the predicted downside move.
""")
            else:
                st.warning(f"""
### ⚠️ SELL / RISK  ← *Downside Scenario*
| Field | Value |
|---|---|
| **Signal** | Model says UP, but downside possible |
| **Entry Price** | ${current_price:.2f} |
| **Worst-Case Price** | ${pred_data['confidence_interval'][0]:.2f} (CI lower) |
| **Max Loss / Share** | −${current_price - pred_data['confidence_interval'][0]:.2f} |
| **Max Loss %** | −{((current_price - pred_data['confidence_interval'][0]) / current_price * 100):.2f}% |
| **Risk** | Low — model predicts UP move |

> Even in a BUY signal, track this downside risk for stop-loss planning.
""")

        # ── P&L summary bar ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 P&L Summary")
        pnl_cols = st.columns(4)
        with pnl_cols[0]:
            st.metric("Recommendation", recommendation,
                      delta="Model Signal", delta_color="off")
        with pnl_cols[1]:
            st.metric("Expected P&L / Share",
                      f"{'+'if price_change>=0 else ''}{price_change:.2f}",
                      delta=f"{'+' if pnl_percentage>=0 else ''}{pnl_percentage:.2f}%",
                      delta_color="normal")
        with pnl_cols[2]:
            st.metric("Max Upside / Share",
                      f"+${pred_data['confidence_interval'][1] - current_price:.2f}",
                      delta=f"+{((pred_data['confidence_interval'][1]-current_price)/current_price*100):.2f}%",
                      delta_color="normal")
        with pnl_cols[3]:
            st.metric("Max Downside / Share",
                      f"-${current_price - pred_data['confidence_interval'][0]:.2f}",
                      delta=f"-{((current_price-pred_data['confidence_interval'][0])/current_price*100):.2f}%",
                      delta_color="inverse")

        st.markdown("---")
        
        # Display confidence interval details
        st.markdown("### 🎯 Prediction Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Ticker:** {pred_data['ticker']}")
            st.write(f"**Generated At:** {pred_data['generated_at']}")
        
        with col2:
            ci_lower, ci_upper = pred_data['confidence_interval']
            st.write(f"**CI Lower:** ${ci_lower:.2f}")
            st.write(f"**CI Upper:** ${ci_upper:.2f}")
        
        st.markdown("---")
        
        # Fetch historical data
        st.markdown("### 📈 Historical Data & Predictions")
        
        with st.spinner("📡 Fetching historical data..."):
            try:
                response = httpx.get(
                    f"{API_URL}/history/{ticker}",
                    params={"lookback_days": 30},
                    timeout=30.0
                )
                response.raise_for_status()
                hist_data = response.json()
                
                # Create DataFrame
                df_hist = pd.DataFrame({
                    'Date': hist_data['dates'],
                    'Actual': hist_data['actual'],
                    'Predicted': hist_data['predictions']
                })
                
                # Create Plotly chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df_hist['Date'],
                    y=df_hist['Actual'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df_hist['Date'],
                    y=df_hist['Predicted'],
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{ticker} - 30 Day Historical Performance",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display historical data in table format
                st.markdown("#### 📋 Historical Data Table")
                st.dataframe(
                    df_hist.assign(
                        Actual=df_hist['Actual'].apply(lambda x: f"${x:.2f}"),
                        Predicted=df_hist['Predicted'].apply(lambda x: f"${x:.2f}")
                    ),
                    use_container_width=True,
                    hide_index=True
                )
                
            except Exception as e:
                st.warning(f"⚠️ Could not fetch historical data: {e}")
        
        st.markdown("---")
        
        # Display prediction summary table
        st.markdown("### 💹 Prediction Summary Table")
        pred_summary = pd.DataFrame({
            'Metric': [
                'Current Close Price',
                'Predicted Close Price',
                'Price Change',
                'Direction',
                'Expected P&L (Per Share)',
                'Expected P&L (%)',
                'Recommendation',
                'Confidence Interval Lower',
                'Confidence Interval Upper',
                'Confidence Range',
                'Last Update'
            ],
            'Value': [
                f"${current_price:.2f}",
                f"${predicted_price:.2f}",
                f"${price_change:.2f}",
                pred_data['direction'].upper(),
                f"${price_change:.2f}",
                f"{pnl_percentage:.2f}%",
                recommendation,
                f"${pred_data['confidence_interval'][0]:.2f}",
                f"${pred_data['confidence_interval'][1]:.2f}",
                f"${pred_data['confidence_interval'][1] - pred_data['confidence_interval'][0]:.2f}",
                pred_data['last_update']
            ]
        })
        st.dataframe(pred_summary, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Model details expander
        with st.expander("ℹ️ Model Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Features:**")
                features_text = """
                - Close Price
                - Volume
                - RSI (14)
                - MACD (12, 26, 9)
                - MACD Histogram
                - Bollinger Bands (20, 2.0)
                - SMA (20, 50)
                - EMA (12)
                """
                st.markdown(features_text)
            
            with col2:
                st.markdown("**Architecture:**")
                arch_text = """
                - LSTM Layer 1: 128 units
                - LSTM Layer 2: 64 units
                - Dropout: 0.2
                - Output: Dense(1)
                - Loss: MSE
                - Optimizer: Adam (lr=0.001)
                """
                st.markdown(arch_text)
            
            st.markdown("**Training Config:**")
            config_text = f"""
            - Lookback Window: 60 days
            - Batch Size: 32
            - Max Epochs: 100
            - Early Stopping Patience: 10 epochs
            - Train/Val/Test Split: 70/15/15
            """
            st.markdown(config_text)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        logger.exception("Error in Streamlit app")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with PyTorch, FastAPI, and Streamlit | Free Stack ⚡</p>
</div>
""", unsafe_allow_html=True)
