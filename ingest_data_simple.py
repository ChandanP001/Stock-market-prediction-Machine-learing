"""
Simplified Stock Data Ingestion Script (No Database Required)

This script fetches historical stock data using yfinance and saves it to CSV files.
Perfect for getting started without setting up PostgreSQL.
"""

import os
import sys
import logging
from typing import Optional
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
    """
    try:
        logger.info(f"Fetching data for {ticker} for period {period}")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        
        if data.empty:
            logger.warning(f"No data found for ticker {ticker}")
            return None
        
        data = data.reset_index()
        data['ticker'] = ticker.upper()
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
    """
    try:
        # Create data directory, but don't raise an error if it already exists
        os.makedirs('data', exist_ok=True)
        
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
    """
    try:
        stock_data = fetch_stock_data(ticker, period)
        if stock_data is None:
            return False
        
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
    """Main function to handle command-line execution."""
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        success = ingest_stock_data(ticker)
        if not success:
            sys.exit(1)
    else:
        # Example usage for direct run
        ticker = "AAPL"
        logger.info(f"Starting data ingestion for {ticker}")
        ingest_stock_data(ticker)


if __name__ == "__main__":
    main()
