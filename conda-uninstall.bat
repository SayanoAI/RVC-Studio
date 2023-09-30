@echo off
REM This script will check if a conda environment is available and create it if not
for /f %%i in ('cd') do set ENV_NAME=%%~nxi

echo Are you sure you wish to uninstall %ENV_NAME%? Close this window if you clicked this by mistake.
pause

conda remove --name %ENV_NAME% --all

echo Successfully uninstalled the app. Press any key to close.
pause