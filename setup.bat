@echo off
REM Stock Predictor Project Setup Script for Windows
REM This script sets up the project environment for the stock prediction application

echo Setting up Stock Predictor project...

REM Create project directory
if not exist "stock_predictor" mkdir stock_predictor
cd stock_predictor

REM Initialize Git repository
echo Initializing Git repository...
git init

REM Create Python virtual environment
echo Creating Python virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing required packages...
pip install -r ..\requirements.txt

echo Project setup complete!
echo To activate the virtual environment, run: venv\Scripts\activate.bat
echo To deactivate, run: deactivate
pause

