@echo off
:: Run grimagents using a virtual environment.

:: In order to support executing this batch file from from any location,
:: the module's parent folder must be added to the Python path.

:: NOTE: By default, this file must be copied or moved up one folder for the grimagents folder to be added to PYTHONPATH.
setlocal
set PYTHONPATH=%~dp0grim-agents
"%~dp0.venv\Scripts\python.exe" -m grimagents %*
endlocal
