"""
Simple script to train a model for a specific ticker.
Usage: python train_ticker.py TICKER
"""

import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python train_ticker.py TICKER")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    try:
        # Change to stock_predictor directory
        os.chdir('stock_predictor')
        
        # Import and run the training
        from train_model_simple import train_stock_model
        
        print(f"Training model for {ticker}...")
        success = train_stock_model(ticker, sequence_length=60, epochs=50)
        
        if success:
            print(f"SUCCESS: Model training completed successfully for {ticker}")
            sys.exit(0)  # Explicit success exit
        else:
            print(f"ERROR: Model training failed for {ticker}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Exception during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
