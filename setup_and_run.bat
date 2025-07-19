@echo off
setlocal enabledelayedexpansion

rem Check if the user only wants to run the application
if /i "%1"=="run" goto :runonly

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

echo Installing Flask...
pip install flask
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Flask.
    pause
    exit /b 1
)

echo Installing SpaCy (this may take some time)...
pip install spacy
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Failed to install SpaCy using standard installation.
    echo.
    echo This might be because Microsoft Visual C++ Build Tools are required but not found.
    echo Checking if build tools are available...
    
    where cl.exe >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Microsoft Visual C++ Build Tools are not found.
        echo Please install Visual C++ Build Tools from:
        echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
        echo The application will still run using a fallback text processing method.
        echo.
    ) else (
        echo Build tools found, but installation still failed.
        echo Trying alternative installation methods...
        pip install spacy --no-build-isolation
    )
    
    echo.
    choice /C YN /M "Continue with setup? (Y/N)"
    if !ERRORLEVEL! EQU 2 (
        pause
        exit /b 1
    )
    echo The application will use a fallback text processing method.
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

rem Let user choose which SpaCy model to download
echo.
echo Please choose which SpaCy model to download:
echo.
echo 1. Small model (en_core_web_sm) - Faster download, less accurate, ~13MB
echo 2. Medium model (en_core_web_md) - Better performance with vectors, ~45MB
echo 3. Large model (en_core_web_lg) - Best performance, includes vectors, ~550MB
echo.
choice /C 123 /N /M "Enter your choice (1-3): "

set MODEL=en_core_web_sm
if !ERRORLEVEL! EQU 2 set MODEL=en_core_web_md
if !ERRORLEVEL! EQU 3 set MODEL=en_core_web_lg

echo.
echo Downloading SpaCy model !MODEL! (this may take a few minutes)...
python -m spacy download !MODEL!
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Failed to download SpaCy model !MODEL!.
    echo The application will still run using a fallback text processing method.
    echo.
    choice /C YN /M "Continue with setup? (Y/N)"
    if !ERRORLEVEL! EQU 2 (
        pause
        exit /b 1
    )
) else (
    echo SpaCy model !MODEL! downloaded successfully.
)

echo Setup complete! Starting the application...
goto :startapp

:runonly
echo Starting AI Engineering FAQ Chatbot...

rem When only running, activate the virtual environment if it exists
if exist venv\ (
    call venv\Scripts\activate
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to activate virtual environment.
        echo Running with system Python...
    ) else (
        echo Virtual environment activated.
    )
) else (
    echo No virtual environment found. Using system Python...
    echo If this fails, run the script without parameters first to set up the environment.
)

:startapp
rem Run the application
python app.py

pause
exit /b 0
