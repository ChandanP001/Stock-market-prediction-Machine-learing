# 📈 End-to-End Stock Prediction Application

A complete, production-ready stock prediction application using modern Python 3.12 practices. This project demonstrates the full machine learning pipeline from data ingestion to interactive web application deployment.

## 🎯 Project Overview

This application provides AI-powered stock price predictions using Long Short-Term Memory (LSTM) neural networks. It includes:

- **Data Ingestion**: Automated fetching and storage of historical stock data in PostgreSQL
- **Model Training**: LSTM-based neural network training with TensorFlow/Keras
- **Interactive Web App**: Beautiful Streamlit interface for predictions and analysis
- **Production Ready**: Robust error handling, logging, and scalable architecture

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  PostgreSQL DB  │───▶│  LSTM Model     │───▶│  Streamlit App  │
│   (yfinance)    │    │  (stock_prices) │    │  (TensorFlow)   │    │  (Web Interface)│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 12+
- Git

### 1. Setup Environment

```bash
# Clone and setup the project
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source stock_predictor/venv/bin/activate  # Linux/Mac
# or
stock_predictor\\venv\\Scripts\\activate  # Windows
```

### 2. Database Setup

```bash
# Create PostgreSQL database
createdb stock_predictor

# Run schema creation
psql -d stock_predictor -f schema.sql

# Configure environment variables
cp env_example.txt .env
# Edit .env with your database credentials
```

### 3. Data Ingestion

```bash
# Ingest stock data (example with AAPL)
python ingest_data.py
```

### 4. Model Training

```bash
# Train LSTM model
python train_model.py
```

### 5. Launch Web Application

```bash
# Start the Streamlit app
streamlit run app.py
```

## 📁 Project Structure

```
stock_predictor/
├── setup.sh                 # Environment setup script
├── requirements.txt         # Python dependencies
├── schema.sql              # Database schema
├── ingest_data.py          # Data ingestion script
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── env_example.txt         # Environment variables template
├── README.md              # This file
└── venv/                  # Virtual environment (created by setup.sh)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your database credentials:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_predictor
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### Model Parameters

Key parameters in `train_model.py`:

- `sequence_length`: Number of days to look back (default: 60)
- `epochs`: Training epochs (default: 100)
- `batch_size`: Training batch size (default: 32)

## 📊 Features

### Data Ingestion (`ingest_data.py`)
- ✅ Fetches historical data using yfinance
- ✅ Robust error handling and logging
- ✅ Upsert functionality for incremental updates
- ✅ Type hints and modern Python practices
- ✅ Environment-based configuration

### Model Training (`train_model.py`)
- ✅ LSTM architecture with dropout regularization
- ✅ MinMaxScaler for data normalization
- ✅ Early stopping and model checkpointing
- ✅ Comprehensive evaluation metrics
- ✅ Modern .keras format for model saving

### Web Application (`app.py`)
- ✅ Interactive Streamlit interface
- ✅ Real-time data fetching
- ✅ Beautiful candlestick charts with Plotly
- ✅ AI-powered predictions
- ✅ Buy/Sell/Hold recommendations
- ✅ Key financial metrics display

## 🎨 Web Application Features

### Main Dashboard
- **Stock Ticker Input**: Enter any valid ticker symbol
- **Period Selection**: Choose data range (1mo to 5y)
- **Interactive Charts**: Candlestick price charts with volume
- **Real-time Predictions**: AI-powered next-day price forecasts

### Analysis Metrics
- Current vs Predicted Price
- Price Change Percentage
- Investment Recommendations
- 52-Week High/Low
- Average Volume
- Volatility Analysis

### Visualizations
- Interactive candlestick charts
- Volume analysis
- Price trend indicators
- Responsive design for all devices

## 🧠 Model Architecture

The LSTM model uses the following architecture:

```
Input Layer (60 days × 1 feature)
    ↓
LSTM Layer 1 (50 units) + Dropout (0.2)
    ↓
LSTM Layer 2 (50 units) + Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Output Layer (1 unit - predicted price)
```

**Training Features:**
- Adam optimizer with learning rate 0.001
- Mean Squared Error loss function
- Early stopping with patience of 10 epochs
- 80/20 train/validation split

## 📈 Usage Examples

### Training Multiple Models

```python
# Train models for different tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

for ticker in tickers:
    # Ingest data
    ingest_stock_data(ticker, period="5y")
    
    # Train model
    train_stock_model(ticker, sequence_length=60, epochs=100)
```

### Batch Data Ingestion

```python
# Ingest data for multiple tickers
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
periods = ['1y', '2y', '5y']

for ticker in tickers:
    for period in periods:
        ingest_stock_data(ticker, period)
```

## 🔍 Monitoring and Logging

The application includes comprehensive logging:

- **Data Ingestion**: Track successful/failed data fetches
- **Model Training**: Monitor training progress and metrics
- **Web App**: Log user interactions and predictions
- **Error Handling**: Detailed error messages and stack traces

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
# Using Streamlit Cloud, Heroku, or Docker
# Ensure all dependencies are installed
pip install -r requirements.txt

# Set environment variables
export DB_HOST=your_production_db_host
export DB_PORT=5432
# ... other variables

# Run the application
streamlit run app.py --server.port 8501
```

## 🧪 Testing

### Data Quality Checks
- Verify data completeness
- Check for missing values
- Validate price ranges
- Monitor data freshness

### Model Performance
- Track prediction accuracy
- Monitor validation metrics
- Compare with baseline models
- A/B test different architectures

## 🔒 Security Considerations

- Database credentials in environment variables
- Input validation for ticker symbols
- Error handling to prevent information leakage
- Secure model file storage

## 📚 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **yfinance**: Stock data fetching
- **scikit-learn**: Data preprocessing
- **tensorflow**: Deep learning framework
- **streamlit**: Web application framework
- **psycopg2-binary**: PostgreSQL connectivity

### Visualization
- **plotly**: Interactive charts
- **streamlit**: Web interface

### Utilities
- **python-dotenv**: Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Database Connection Errors:**
- Verify PostgreSQL is running
- Check database credentials in .env
- Ensure database exists

**Model Loading Errors:**
- Verify .keras and .pkl files exist
- Check file permissions
- Ensure TensorFlow version compatibility

**Data Fetching Issues:**
- Check internet connection
- Verify ticker symbol validity
- Monitor yfinance API limits

### Performance Optimization

- Use GPU for model training
- Implement data caching
- Optimize database queries
- Use connection pooling

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the logs for error details

---

**Happy Trading! 📈🚀**

