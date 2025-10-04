# 📈 Stock Market Prediction (CSV & Streamlit)

A simplified, production-ready stock prediction application using modern Python 3.12+ practices. This project demonstrates the full machine learning pipeline from data ingestion to interactive web application deployment—no database required!

---

## 🎯 Project Overview

- **Data Ingestion**: Fetch and store historical stock data as CSV files
- **Model Training**: LSTM-based neural network with TensorFlow/Keras
- **Interactive Web App**: Streamlit interface for predictions and analysis
- **No Database**: All data is stored in CSV files for simplicity
- **Automated Setup**: Dependencies are installed automatically on first run

---

## 🚀 Quick Start

### 1. Project Structure

```
/stock-market-prediction/
├── app.py                  # Streamlit web application (auto-installs dependencies)
├── ingest_data_simple.py   # Data ingestion script (CSV-based)
├── train_model_simple.py   # Model training script (CSV-based)
├── requirements.txt        # Python dependencies
├── data/                   # CSV data files (created automatically)
│   ├── aapl_stock_data.csv
│   └── amd_stock_data.csv
├── stock_predictor_aapl.keras   # Trained model for AAPL
├── scaler_aapl.pkl              # Scaler for AAPL
├── stock_predictor_amd.keras    # Trained model for AMD
├── scaler_amd.pkl               # Scaler for AMD
└── README.md
```

### 2. First-Time Setup & Run

Just run the app! All dependencies will be installed automatically if missing:

```bash
streamlit run app.py
```

- On first run, the app will detect missing packages, install them, and reload itself.
- On subsequent runs, the app will start immediately.

### 3. Ingest Data for a New Ticker

```bash
python ingest_data_simple.py GOOGL
```
- This will create a CSV file with stock data inside the `data/` directory.

### 4. Train a Model for a New Ticker

```bash
python train_model_simple.py GOOGL
```
- This will generate a `.keras` model file and a `.pkl` scaler file in your project directory.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```
- The app will now include the new ticker in its dropdown list of available models.

---

## 📁 Features

- **No database required**: All data is stored as CSV files
- **Automated dependency installation**: No manual `pip install` needed
- **Interactive Streamlit UI**: For predictions, charts, and recommendations
- **Easy extensibility**: Add new tickers with a single command

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt` and installed automatically by `app.py`.

- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- streamlit
- plotly
- python-dotenv (optional)

---

## 🆘 Troubleshooting

- If auto-install fails, run: `pip install -r requirements.txt`
- Ensure you are using Python 3.12+
- For any issues, check the logs in the Streamlit app or raise an issue.

---

**Happy Trading! 📈🚀**

