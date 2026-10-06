@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Project Python environment was not found. See README.md. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0cli.py" %*
exit /b %errorlevel%
