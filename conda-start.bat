@echo off

REM This script will check if a conda environment is available and create it if not
for /f %%i in ('cd') do set ENV_NAME=%%~nxi
SET INSTALL_DIR=%userprofile%\Miniconda3
SET PATH=%INSTALL_DIR%\condabin;%PATH%

CALL conda info --envs | findstr /i %ENV_NAME%
if %errorlevel% == 0 (
    echo %ENV_NAME% environment is already available
) else (
    echo %ENV_NAME% environment does not exist
    echo Creating a new environment
    CALL conda create -n %ENV_NAME% python=3.10 -y
)

rem Activate environment
CALL conda activate %ENV_NAME%

if %errorlevel% == 0 (
    rem install packages
    CALL pip install -r requirements.txt
    CALL streamlit run Home.py
) else (
    echo Failed to activate environment...
)
PAUSE