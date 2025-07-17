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

echo Checking for required build tools...
where cl.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Microsoft Visual C++ Build Tools are required but not found.
    echo Please install Visual C++ Build Tools from:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo After installation, run this script again.
    echo.
    echo Alternatively, we'll try to install pre-built wheels to avoid compilation.
    echo.
    choice /C YN /M "Do you want to try installing pre-built wheels instead? (Y/N)"
    if !ERRORLEVEL! EQU 2 (
        pause
        exit /b 1
    )
    echo Will try to use pre-built wheels...
)

echo Installing Flask...
pip install flask
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Flask.
    pause
    exit /b 1
)

echo Installing pre-built wheels for SpaCy dependencies...
pip install wheel
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install wheel package.
    pause
    exit /b 1
)

rem Try to install pre-built wheels if available
echo Installing SpaCy (this may take some time)...
pip install spacy --no-build-isolation
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Failed to install SpaCy.
    echo.
    echo You need to install Microsoft Visual C++ Build Tools to compile SpaCy.
    echo Please visit: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo The application will still run using a fallback text processing method.
    echo.
    choice /C YN /M "Continue with setup? (Y/N)"
    if !ERRORLEVEL! EQU 2 (
        pause
        exit /b 1
    )
)

echo Installing NumPy...
pip install numpy
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install NumPy.
    pause
    exit /b 1
)

echo Installing scikit-learn...
pip install scikit-learn
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install scikit-learn.
    pause
    exit /b 1
)

rem Download SpaCy model (using the small model which is faster to download and doesn't need compilation)
echo Downloading SpaCy model (this may take a few minutes)...
python -m spacy download en_core_web_sm
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to download SpaCy model.
    echo The application will still run using a fallback text processing method.
    echo.
    choice /C YN /M "Continue with setup? (Y/N)"
    if !ERRORLEVEL! EQU 2 (
        pause
        exit /b 1
    )
) else (
    echo SpaCy model downloaded successfully.
)

echo Setup complete! Starting the application...
python app.py

pause
