"""
Stock Data Ingestion Script

This script fetches historical stock data using yfinance and stores it in a PostgreSQL database.
It includes robust error handling and supports incremental data updates.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime, date
import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Handles database connection and operations."""
    
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
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a SQL query."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()
        except psycopg2.Error as e:
            logger.error(f"Error executing query: {e}")
            self.connection.rollback()
            raise


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
        
        # Rename columns to match database schema
        data.columns = data.columns.str.lower()
        data = data.rename(columns={'date': 'date'})
        
        # Add ticker column
        data['ticker'] = ticker.upper()
        
        # Select and reorder columns
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        data = data[columns]
        
        # Convert date to date type
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        logger.info(f"Successfully fetched {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return None


def ingest_stock_data(ticker: str, period: str = "5y") -> bool:
    """
    Ingest stock data into PostgreSQL database.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period for yfinance
    
    Returns:
        True if successful, False otherwise
    """
    db = None
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(ticker, period)
        if stock_data is None:
            return False
        
        # Connect to database
        db = DatabaseConnection()
        
        # Prepare data for insertion
        data_tuples = [
            (
                row['ticker'],
                row['date'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            )
            for _, row in stock_data.iterrows()
        ]
        
        # Insert data with conflict handling (upsert)
        insert_query = """
            INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (ticker, date)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                created_at = CURRENT_TIMESTAMP
        """
        
        with db.connection.cursor() as cursor:
            execute_values(
                cursor,
                insert_query,
                data_tuples,
                template=None,
                page_size=1000
            )
            db.connection.commit()
        
        logger.info(f"Successfully ingested {len(data_tuples)} records for {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Error ingesting data for {ticker}: {e}")
        return False
    
    finally:
        if db:
            db.close()


def get_latest_date(ticker: str) -> Optional[date]:
    """
    Get the latest date for a ticker in the database.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Latest date or None if no data exists
    """
    db = None
    try:
        db = DatabaseConnection()
        
        query = "SELECT MAX(date) FROM stock_prices WHERE ticker = %s"
        
        with db.connection.cursor() as cursor:
            cursor.execute(query, (ticker.upper(),))
            result = cursor.fetchone()
            
            return result[0] if result and result[0] else None
            
    except Exception as e:
        logger.error(f"Error getting latest date for {ticker}: {e}")
        return None
    
    finally:
        if db:
            db.close()


def main():
    """Main function to demonstrate usage."""
    # Example usage with AAPL
    ticker = "AAPL"
    period = "5y"
    
    logger.info(f"Starting data ingestion for {ticker}")
    
    # Check if we have existing data
    latest_date = get_latest_date(ticker)
    if latest_date:
        logger.info(f"Latest data for {ticker} is from {latest_date}")
    
    # Ingest data
    success = ingest_stock_data(ticker, period)
    
    if success:
        logger.info(f"Data ingestion completed successfully for {ticker}")
    else:
        logger.error(f"Data ingestion failed for {ticker}")
        sys.exit(1)


if __name__ == "__main__":
    main()
