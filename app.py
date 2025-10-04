"""
Stock Price Predictor - Streamlit Web Application

This interactive web application allows users to:
1. Input a stock ticker symbol
2. View historical price data with interactive charts
3. Get AI-powered price predictions
4. Receive buy/sell/hold recommendations

To run the app: streamlit run app.py
"""

import os
import sys
import logging
import pickle
import subprocess
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .recommendation-buy {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .recommendation-sell {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .recommendation-hold {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


class StockPredictorApp:
    """Main application class for the Stock Predictor."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sequence_length = 60
    
    def train_new_model(self, ticker: str) -> bool:
        """
        Train a new model for the given ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, ingest data for the ticker
            logger.info(f"Starting data ingestion for {ticker}")
            
            # Use the simplified training script
            result = subprocess.run([
                sys.executable, 
                'stock_predictor/train_model_simple.py'
            ], 
            input=f"{ticker}\n", 
            text=True, 
            capture_output=True,
            cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully trained model for {ticker}")
                return True
            else:
                logger.error(f"Training failed for {ticker}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            return False
    
    def load_model_and_scaler(self, ticker: str) -> bool:
        """
        Load the trained model and scaler for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = f'stock_predictor_{ticker.lower()}.keras'
            scaler_path = f'scaler_{ticker.lower()}.pkl'
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return False
            
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info(f"Successfully loaded model and scaler for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}")
            return False
    
    def fetch_stock_data(self, ticker: str, period: str = "1y", min_days: int = 60) -> Optional[pd.DataFrame]:
        """
        Fetch recent stock data using yfinance with automatic period adjustment.
        
        Args:
            ticker: Stock ticker symbol
            period: Initial data period
            min_days: Minimum days required for prediction
        
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Try the requested period first
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
            # Check if we have enough data
            if len(data) < min_days:
                # Try progressively longer periods
                fallback_periods = ["3mo", "6mo", "1y", "2y", "5y"]
                for fallback_period in fallback_periods:
                    if fallback_period != period:  # Don't retry the same period
                        logger.info(f"Insufficient data with {period}, trying {fallback_period}")
                        data = stock.history(period=fallback_period)
                        if len(data) >= min_days:
                            logger.info(f"Successfully fetched {len(data)} days with {fallback_period}")
                            break
                
                # If still insufficient, return what we have
                if len(data) < min_days:
                    logger.warning(f"Only {len(data)} days available, need {min_days}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def predict_price(self, data: pd.DataFrame) -> Optional[float]:
        """
        Predict the next day's closing price.
        
        Args:
            data: Recent stock data
        
        Returns:
            Predicted price or None if error
        """
        try:
            if self.model is None or self.scaler is None:
                return None
            
            # Get the last sequence_length days of closing prices
            recent_data = data['Close'].tail(self.sequence_length).values.reshape(-1, 1)
            
            # Check if we have enough data
            if len(recent_data) < self.sequence_length:
                logger.error(f"Insufficient data for prediction. Need {self.sequence_length} days, got {len(recent_data)}")
                return None
            
            # Scale the data
            scaled_data = self.scaler.transform(recent_data)
            
            # Reshape for prediction
            X = scaled_data.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            prediction = self.model.predict(X, verbose=0)
            
            # Inverse transform to get actual price
            predicted_price = self.scaler.inverse_transform(prediction)[0, 0]
            
            return predicted_price
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def get_recommendation(self, current_price: float, predicted_price: float) -> Tuple[str, str]:
        """
        Generate buy/sell/hold recommendation based on prediction.
        
        Args:
            current_price: Current stock price
            predicted_price: Predicted next day price
        
        Returns:
            Tuple of (recommendation, reasoning)
        """
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        if price_change_percent > 2:
            return "BUY", f"Strong upward trend predicted (+{price_change_percent:.2f}%)"
        elif price_change_percent < -2:
            return "SELL", f"Downward trend predicted ({price_change_percent:.2f}%)"
        else:
            return "HOLD", f"Minimal change predicted ({price_change_percent:+.2f}%)"
    
    def create_price_chart(self, data: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Create an interactive price chart.
        
        Args:
            data: Stock data
            ticker: Stock ticker symbol
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{ticker} Stock Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Sidebar
        st.sidebar.header("🔧 Configuration")
        
        # Get available ticker symbols from trained models
        model_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        
        # Show available models count with refresh button
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if model_files:
                st.sidebar.success(f"📊 {len(model_files)} trained model(s) available")
            else:
                st.sidebar.warning("⚠️ No trained models found")
        with col2:
            if st.sidebar.button("🔄", help="Refresh model list"):
                st.rerun()
        
        available_tickers = []
        if model_files:
            available_tickers = [f.replace('stock_predictor_', '').replace('.keras', '').upper() for f in model_files]
        
        # Ticker selection
        if available_tickers:
            ticker = st.sidebar.selectbox(
                "Stock Ticker Symbol",
                options=available_tickers,
                index=0 if "AAPL" not in available_tickers else available_tickers.index("AAPL"),
                help="Select from available trained models"
            )
        else:
            # Fallback to text input if no models available
            ticker = st.sidebar.text_input(
                "Stock Ticker Symbol",
                value="AAPL",
                help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT). No trained models found."
            ).upper()
        
        # Period selection with smart defaults
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
        period_labels = {
            "1mo": "1 Month (⚠️ May not work with all models)",
            "3mo": "3 Months (✅ Recommended)",
            "6mo": "6 Months (✅ Recommended)", 
            "1y": "1 Year (✅ Recommended)",
            "2y": "2 Years (✅ Recommended)",
            "5y": "5 Years (✅ Recommended)"
        }
        
        # Smart default based on available models
        default_index = 1  # Default to 3mo for better compatibility
        
        period = st.sidebar.selectbox(
            "Data Period",
            options=period_options,
            index=default_index,
            format_func=lambda x: period_labels[x],
            help="Select the time period for historical data. Models need at least 60 days of data."
        )
        
        # Show warning for 1mo period
        if period == "1mo":
            st.sidebar.warning("⚠️ 1 month may not provide enough data for predictions. Consider using 3+ months.")
        
        # Prediction button
        predict_button = st.sidebar.button("🚀 Get Prediction", type="primary")
        
        # Training section
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Train New Model")
        
        # Predefined popular tickers
        popular_tickers = ["GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "NFLX", "AMD", "INTC", "CRM"]
        
        # Get already trained tickers to exclude them
        trained_tickers = [f.replace('stock_predictor_', '').replace('.keras', '').upper() for f in model_files]
        available_to_train = [t for t in popular_tickers if t not in trained_tickers]
        
        if available_to_train:
            st.sidebar.write("**Popular tickers to train:**")
            selected_ticker = st.sidebar.selectbox(
                "Select Ticker",
                options=available_to_train,
                help="Choose from popular tickers that haven't been trained yet"
            )
        else:
            st.sidebar.info("All popular tickers are already trained!")
            selected_ticker = None
        
        # Custom ticker input
        custom_ticker = st.sidebar.text_input(
            "Or enter custom ticker",
            placeholder="e.g., BRK.A, JPM, V",
            help="Enter any valid ticker symbol"
        ).upper()
        
        # Determine which ticker to use
        ticker_to_train = custom_ticker if custom_ticker else selected_ticker
        
        train_button = st.sidebar.button("🏋️ Train Model", type="secondary", disabled=not ticker_to_train)
        
        if train_button and ticker_to_train:
            with st.spinner(f"Training model for {ticker_to_train}... This may take 2-3 minutes."):
                try:
                    # First ingest data
                    st.sidebar.info(f"📥 Fetching data for {ticker_to_train}...")
                    
                    # Run data ingestion
                    ingest_result = subprocess.run([
                        'stock_predictor/venv/Scripts/python.exe',
                        'ingest_ticker.py',
                        ticker_to_train
                    ], 
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                    )
                    
                    # Check if data ingestion actually succeeded despite potential Unicode errors
                    output_text = ingest_result.stdout if ingest_result.stdout else ""
                    success_indicators = ["SUCCESS:", "Successfully ingested", "Data saved to"]
                    actual_success = any(indicator in output_text for indicator in success_indicators)
                    
                    if ingest_result.returncode != 0 and not actual_success:
                        error_msg = ingest_result.stderr if ingest_result.stderr else "Unknown error"
                        st.sidebar.error(f"❌ Data ingestion failed: {error_msg}")
                        # Show debug info
                        st.sidebar.text(f"Debug - Return code: {ingest_result.returncode}")
                        st.sidebar.text(f"Debug - Output: {ingest_result.stdout}")
                        return
                    elif actual_success:
                        st.sidebar.success(f"✅ Data ingestion completed for {ticker_to_train}")
                    else:
                        st.sidebar.warning(f"⚠️ Data ingestion completed with warnings")
                    
                    st.sidebar.info(f"🤖 Training LSTM model for {ticker_to_train}...")
                    
                    # Run model training
                    train_result = subprocess.run([
                        'stock_predictor/venv/Scripts/python.exe',
                        'train_ticker.py',
                        ticker_to_train
                    ], 
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd()
                    )
                    
                    # Check if training actually succeeded despite potential Unicode errors
                    train_output = train_result.stdout if train_result.stdout else ""
                    train_success_indicators = ["SUCCESS:", "Model training completed successfully", "Model saved as"]
                    train_actual_success = any(indicator in train_output for indicator in train_success_indicators)
                    
                    if train_result.returncode == 0 or train_actual_success:
                        # Copy model files to main directory
                        model_file = f'stock_predictor/stock_predictor_{ticker_to_train.lower()}.keras'
                        scaler_file = f'stock_predictor/scaler_{ticker_to_train.lower()}.pkl'
                        
                        if os.path.exists(model_file) and os.path.exists(scaler_file):
                            import shutil
                            shutil.copy2(model_file, '.')
                            shutil.copy2(scaler_file, '.')
                        
                        st.sidebar.success(f"✅ {ticker_to_train} model trained successfully!")
                        st.sidebar.info("🔄 Please refresh the page to see the new model in the dropdown.")
                        st.rerun()  # Refresh the app
                    else:
                        error_msg = train_result.stderr if train_result.stderr else "Unknown error"
                        st.sidebar.error(f"❌ Training failed: {error_msg}")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
        
        # Main content
        if predict_button and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                # Check if model exists
                if not self.load_model_and_scaler(ticker):
                    st.error(f"❌ No trained model found for {ticker}. Please train a model first using train_model.py")
                    st.info("💡 Available tickers with models: Check your directory for .keras and .pkl files")
                    return
                
                # Fetch data with smart period adjustment
                st.info(f"📥 Fetching data for {ticker}...")
                data = self.fetch_stock_data(ticker, period, min_days=60)
                if data is None:
                    st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return
                
                # Show data info
                data_days = len(data)
                st.success(f"✅ Fetched {data_days} days of data for {ticker}")
                
                # Check if we have enough data for prediction
                if data_days < 60:
                    st.warning(f"⚠️ Only {data_days} days available. Model needs 60 days for accurate predictions.")
                    st.info("💡 Consider using a longer data period (3+ months) for better results.")
                    # Still try to make prediction with available data
                
                # Get current price
                current_price = data['Close'].iloc[-1]
                
                # Make prediction
                predicted_price = self.predict_price(data)
                if predicted_price is None:
                    st.error("❌ Could not generate prediction. Please try again.")
                    return
                
                # Get recommendation
                recommendation, reasoning = self.get_recommendation(current_price, predicted_price)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:.2f}",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Predicted Price",
                        value=f"${predicted_price:.2f}",
                        delta=f"${predicted_price - current_price:.2f}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Price Change %",
                        value=f"{((predicted_price - current_price) / current_price * 100):+.2f}%",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendation
                st.markdown("---")
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Investment Recommendation")
                
                if recommendation == "BUY":
                    st.markdown(f'<p class="recommendation-buy">🟢 {recommendation}</p>', unsafe_allow_html=True)
                elif recommendation == "SELL":
                    st.markdown(f'<p class="recommendation-sell">🔴 {recommendation}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="recommendation-hold">🟡 {recommendation}</p>', unsafe_allow_html=True)
                
                st.markdown(f"**Reasoning:** {reasoning}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Price chart
                st.markdown("---")
                st.markdown("### 📊 Historical Price Chart")
                fig = self.create_price_chart(data, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Data period info
                st.markdown("---")
                st.markdown("### 📊 Data Information")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Data Period Used", f"{data_days} days")
                with col_info2:
                    st.metric("Requested Period", period.upper())
                
                # Additional metrics
                st.markdown("---")
                st.markdown("### 📈 Additional Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("52-Week High", f"${data['High'].max():.2f}")
                
                with col2:
                    st.metric("52-Week Low", f"${data['Low'].min():.2f}")
                
                with col3:
                    avg_volume = data['Volume'].mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                
                with col4:
                    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility", f"{volatility:.1f}%")
        
        else:
            # Welcome message
            st.markdown("""
            ### 🎯 Welcome to the Stock Price Predictor!
            
            This application uses advanced LSTM neural networks to predict stock prices and provide investment recommendations.
            
            **How to use:**
            1. Select a stock ticker symbol from the dropdown
            2. Choose a data period (3+ months recommended for best results)
            3. Click "Get Prediction" to see the AI-powered forecast
            
            **💡 Pro Tip:** Models need at least 60 days of data. If you select 1 month, the app will automatically fetch more data to ensure accurate predictions.
            
            **Features:**
            - 📊 Interactive price charts with candlestick visualization
            - 🤖 AI-powered price predictions using LSTM models
            - 💡 Buy/Sell/Hold recommendations
            - 📈 Key financial metrics and volatility analysis
            
            **Note:** Make sure you have trained models available for the ticker you want to analyze.
            """)
            
            # Show available models
            st.markdown("### 📁 Available Models")
            model_files = [f for f in os.listdir('.') if f.endswith('.keras')]
            if model_files:
                st.success(f"Found {len(model_files)} trained models:")
                for model_file in model_files:
                    ticker_name = model_file.replace('stock_predictor_', '').replace('.keras', '').upper()
                    st.write(f"✅ {ticker_name}")
            else:
                st.warning("No trained models found. Please run train_model.py first.")


def main():
    """Main function to run the application."""
    app = StockPredictorApp()
    app.run()


if __name__ == "__main__":
    main()
