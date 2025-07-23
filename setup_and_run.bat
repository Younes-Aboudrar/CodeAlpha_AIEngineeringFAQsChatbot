@echo off
setlocal enabledelayedexpansion

echo Setting up AI Engineering FAQ Chatbot...

rem Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b 1
)

rem Create a virtual environment if it doesn't exist
if not exist venv\ (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment. Please check your Python installation.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

rem Activate the virtual environment
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Virtual environment activated.

echo Preparing for installation...
echo Installing pip, setuptools and wheel...
pip install -U pip setuptools wheel
if %ERRORLEVEL% NEQ 0 (
    echo Failed to update pip and setuptools.
    pause
    exit /b 1
)

echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install required dependencies.
    pause
    exit /b 1
)

echo.
echo NOTE: SpaCy and its language models are not included in the basic setup.
echo To install SpaCy and choose a language model, please run:
echo   python spacy_setup.py
echo.

echo Setup complete!
pause
exit /b 0
