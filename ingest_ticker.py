"""
Simple script to ingest data for a specific ticker.
Usage: python ingest_ticker.py TICKER
"""

import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python ingest_ticker.py TICKER")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    try:
        # Change to stock_predictor directory
        os.chdir('stock_predictor')
        
        # Import and run the ingestion
        from ingest_data_simple import ingest_stock_data
        
        print(f"Ingesting data for {ticker}...")
        success = ingest_stock_data(ticker, period="5y")
        
        if success:
            print(f"SUCCESS: Data ingestion completed successfully for {ticker}")
            sys.exit(0)  # Explicit success exit
        else:
            print(f"ERROR: Data ingestion failed for {ticker}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Exception during ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
