@echo off
REM This script will check if a conda environment is available and create it if not
set ENV_NAME="RVC-Studio"

conda info --envs | findstr /i %ENV_NAME%
if %errorlevel% == 0 (
    echo %ENV_NAME% environment is already available
) else (
    echo %ENV_NAME% environment does not exist
    echo Creating a new environment
    CALL conda create -n %ENV_NAME% python=3.8.17 -y
)

CALL conda activate %ENV_NAME%

if %errorlevel% == 0 (
    CALL pip install -r requirements.txt
    CALL streamlit run Home.py
) else (
    echo Failed to activate environment...
)
PAUSE