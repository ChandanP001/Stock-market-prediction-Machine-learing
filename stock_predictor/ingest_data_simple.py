"""
Simplified Stock Data Ingestion Script (No Database Required)

This script fetches historical stock data using yfinance and saves it to CSV files.
Perfect for getting started without setting up PostgreSQL.
"""

import os
import sys
import logging
from typing import Optional
from datetime import datetime
import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str, period: str = "5y") -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        DataFrame with stock data or None if error
    """
    try:
        logger.info(f"Fetching data for {ticker} for period {period}")
        
        # Create yfinance ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        data = stock.history(period=period)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Add ticker column
        data['ticker'] = ticker.upper()
        
        # Select and reorder columns
        columns = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[columns]
        
        logger.info(f"Successfully fetched {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def save_data_to_csv(data: pd.DataFrame, ticker: str) -> bool:
    """
    Save stock data to CSV file.
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker symbol
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        filename = f'data/{ticker.lower()}_stock_data.csv'
        data.to_csv(filename, index=False)
        
        logger.info(f"Data saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")
        return False


def ingest_stock_data(ticker: str, period: str = "5y") -> bool:
    """
    Ingest stock data and save to CSV.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for yfinance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, period)
        if stock_data is None:
            return False
        
        # Save to CSV
        success = save_data_to_csv(stock_data, ticker)
        
        if success:
            logger.info(f"Successfully ingested {len(stock_data)} records for {ticker}")
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f"Error ingesting data for {ticker}: {e}")
        return False


def main():
    """Main function to demonstrate usage."""
    # Example usage with AAPL
    ticker = "AAPL"
    period = "5y"
    
    logger.info(f"Starting data ingestion for {ticker}")
    
    # Ingest data
    success = ingest_stock_data(ticker, period)
    
    if success:
        logger.info(f"Data ingestion completed successfully for {ticker}")
        logger.info("You can now proceed to train the model!")
    else:
        logger.error(f"Data ingestion failed for {ticker}")
        sys.exit(1)


if __name__ == "__main__":
    main()
