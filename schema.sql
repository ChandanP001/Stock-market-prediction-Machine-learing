-- Stock Prediction Application Database Schema
-- This script creates the necessary tables for storing stock price data

-- Create the stock_prices table
CREATE TABLE IF NOT EXISTS stock_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2) NOT NULL,
    high DECIMAL(10, 2) NOT NULL,
    low DECIMAL(10, 2) NOT NULL,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Composite primary key on ticker and date
    PRIMARY KEY (ticker, date)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices(ticker, date);

-- Add comments for documentation
COMMENT ON TABLE stock_prices IS 'Historical stock price data for various tickers';
COMMENT ON COLUMN stock_prices.ticker IS 'Stock ticker symbol (e.g., AAPL, GOOGL)';
COMMENT ON COLUMN stock_prices.date IS 'Trading date';
COMMENT ON COLUMN stock_prices.open IS 'Opening price';
COMMENT ON COLUMN stock_prices.high IS 'Highest price of the day';
COMMENT ON COLUMN stock_prices.low IS 'Lowest price of the day';
COMMENT ON COLUMN stock_prices.close IS 'Closing price';
COMMENT ON COLUMN stock_prices.volume IS 'Trading volume';
COMMENT ON COLUMN stock_prices.created_at IS 'Timestamp when record was inserted';
