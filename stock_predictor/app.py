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
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch recent stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
        
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                return None
            
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
        
        # Ticker input
        ticker = st.sidebar.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Period selection
        period = st.sidebar.selectbox(
            "Data Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select the time period for historical data"
        )
        
        # Prediction button
        predict_button = st.sidebar.button("🚀 Get Prediction", type="primary")
        
        # Main content
        if predict_button and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                # Check if model exists
                if not self.load_model_and_scaler(ticker):
                    st.error(f"❌ No trained model found for {ticker}. Please train a model first using train_model.py")
                    st.info("💡 Available tickers with models: Check your directory for .keras and .pkl files")
                    return
                
                # Fetch data
                data = self.fetch_stock_data(ticker, period)
                if data is None:
                    st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return
                
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
            1. Enter a stock ticker symbol in the sidebar
            2. Select the data period for analysis
            3. Click "Get Prediction" to see the AI-powered forecast
            
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
