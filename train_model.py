"""
Stock Prediction Model Training Script

This script loads stock data from the database, preprocesses it, and trains an LSTM model
for stock price prediction using TensorFlow/Keras.
"""

import os
import sys
import logging
import pickle
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class StockDataLoader:
    """Handles loading stock data from PostgreSQL database."""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME', 'stock_predictor'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password')
            )
            logger.info("Successfully connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def load_data_from_db(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from database for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
        
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            query = """
                SELECT date, open, high, low, close, volume
                FROM stock_prices
                WHERE ticker = %s
                ORDER BY date ASC
            """
            
            data = pd.read_sql_query(query, self.connection, params=[ticker.upper()])
            
            if data.empty:
                logger.warning(f"No data found for ticker {ticker}")
                return None
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Set date as index
            data.set_index('date', inplace=True)
            
            logger.info(f"Loaded {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
            return None


class StockPredictor:
    """Handles stock price prediction model training and inference."""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: DataFrame with stock data
            target_column: Column to use as target variable
        
        Returns:
            Tuple of (X, y) arrays for training
        """
        # Extract the target column
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape X for LSTM (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (time_steps, features)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model built successfully")
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, ticker: str, 
                   epochs: int = 100, batch_size: int = 32) -> None:
        """
        Train the LSTM model.
        
        Args:
            X: Training features
            y: Training targets
            ticker: Stock ticker for model naming
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Split data into train and validation sets
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                f'model_checkpoint_{ticker.lower()}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        logger.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        
        # Evaluate the model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        logger.info(f"Training MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
        logger.info(f"Validation MSE: {val_mse:.6f}, MAE: {val_mae:.6f}")
    
    def save_model(self, ticker: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            ticker: Stock ticker for file naming
        """
        try:
            # Save model in .keras format
            model_filename = f'stock_predictor_{ticker.lower()}.keras'
            self.model.save(model_filename)
            logger.info(f"Model saved as {model_filename}")
            
            # Save scaler
            scaler_filename = f'scaler_{ticker.lower()}.pkl'
            with open(scaler_filename, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved as {scaler_filename}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def predict_next_price(self, data: pd.DataFrame, target_column: str = 'close') -> float:
        """
        Predict the next day's closing price.
        
        Args:
            data: Recent stock data
            target_column: Column to use for prediction
        
        Returns:
            Predicted price
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded")
        
        # Get the last sequence_length days of data
        recent_data = data[target_column].tail(self.sequence_length).values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape for prediction
        X = scaled_data.reshape(1, self.sequence_length, 1)
        
        # Make prediction
        prediction = self.model.predict(X, verbose=0)
        
        # Inverse transform to get actual price
        predicted_price = self.scaler.inverse_transform(prediction)[0, 0]
        
        return predicted_price


def train_stock_model(ticker: str, sequence_length: int = 60, epochs: int = 100) -> bool:
    """
    Complete pipeline to train a stock prediction model.
    
    Args:
        ticker: Stock ticker symbol
        sequence_length: Number of days to look back
        epochs: Number of training epochs
    
    Returns:
        True if successful, False otherwise
    """
    data_loader = None
    try:
        # Load data
        data_loader = StockDataLoader()
        data = data_loader.load_data_from_db(ticker)
        
        if data is None:
            logger.error(f"No data available for {ticker}")
            return False
        
        # Check if we have enough data
        if len(data) < sequence_length + 100:
            logger.error(f"Insufficient data for {ticker}. Need at least {sequence_length + 100} days")
            return False
        
        # Initialize predictor
        predictor = StockPredictor(sequence_length)
        
        # Prepare data
        X, y = predictor.prepare_data(data)
        
        # Train model
        predictor.train_model(X, y, ticker, epochs)
        
        # Save model
        predictor.save_model(ticker)
        
        logger.info(f"Model training completed successfully for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error training model for {ticker}: {e}")
        return False
    
    finally:
        if data_loader:
            data_loader.close()


def main():
    """Main function to demonstrate usage."""
    # Example usage with AAPL
    ticker = "AAPL"
    sequence_length = 60
    epochs = 100
    
    logger.info(f"Starting model training for {ticker}")
    
    success = train_stock_model(ticker, sequence_length, epochs)
    
    if success:
        logger.info(f"Model training completed successfully for {ticker}")
    else:
        logger.error(f"Model training failed for {ticker}")
        sys.exit(1)


if __name__ == "__main__":
    main()
