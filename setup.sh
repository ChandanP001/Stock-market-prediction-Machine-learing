#!/bin/bash

# Stock Predictor Project Setup Script
# This script sets up the project environment for the stock prediction application

echo "Setting up Stock Predictor project..."

# Create project directory
mkdir -p stock_predictor
cd stock_predictor

# Initialize Git repository
echo "Initializing Git repository..."
git init

# Create Python 3.12 virtual environment
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

echo "Project setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"
