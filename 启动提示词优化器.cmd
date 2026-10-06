@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project Python environment was not found. See README.md.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0launcher.py"
set "optimizer_exit=%errorlevel%"
if not "%optimizer_exit%"=="0" pause
exit /b %optimizer_exit%
