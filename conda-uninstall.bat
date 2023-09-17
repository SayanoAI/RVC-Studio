@echo off
REM This script will check if a conda environment is available and create it if not
set ENV_NAME="RVC-Studio"

echo Are you sure you wish to uninstall? Close this window if you clicked this by mistake.
pause

conda remove --name %ENV_NAME% --all

echo Successfully uninstalled the app. Press any key to close.
pause