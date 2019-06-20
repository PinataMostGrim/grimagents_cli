@echo off
:: Run grimagents using a virtual environment.

:: In order to support executing this batch file from from any location,
:: the module's parent folder must be added to the Python path.
setlocal
set PYTHONPATH=%~dp0grim-agents
"%~dp0.venv\Scripts\python.exe" -m grimagents %*
endlocal
