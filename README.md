# AI Engineering FAQ Chatbot

A simple chatbot application that answers questions about AI Engineering using Natural Language Processing techniques.

## Features

- Processes FAQ data using SpaCy (or a fallback method if SpaCy is unavailable)
- Matches user questions with the most similar FAQ using semantic similarity
- Provides a clean, responsive chat interface
- Offers real-time responses to queries about AI Engineering

## Prerequisites

- Python 3.7+ installed
- pip package manager
- For optimal performance: Microsoft Visual C++ Build Tools (for SpaCy)

## Installation - Easy Method

1. Simply run the `setup_and_run.bat` script
2. Follow the prompts in the console

The script will:
- Set up a Python virtual environment
- Install required dependencies
- Download language models (if possible)
- Start the Flask application

## Manual Installation

If you prefer to install manually:

1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment: `venv\Scripts\activate`
3. Install Flask: `pip install flask`
4. Install NumPy: `pip install numpy`
5. Install scikit-learn: `pip install scikit-learn`
6. Install SpaCy (optional): `pip install spacy`
7. Download SpaCy model (optional): `python -m spacy download en_core_web_sm`
8. Run the application: `python app.py`

## Troubleshooting

### SpaCy Installation Issues

If you encounter errors installing SpaCy:

1. **Microsoft Visual C++ Build Tools:** SpaCy requires C++ compilation. Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. **Don't worry:** The application has a fallback mode that works without SpaCy

### Python Version Compatibility

- This application works best with Python 3.8-3.10
- If using newer versions (3.11+), you may need to install pre-compiled wheels

### Other Issues

- Check your Python installation is working: `python --version`
- Verify pip is installed: `pip --version`
- Make sure you're running commands from the project directory

## Running the Application

1. If you didn't use the setup script, start the Flask server: `python app.py`
2. Open a web browser and go to: `http://localhost:5000`
3. Start asking questions about AI Engineering!

## How It Works

1. **Data Processing**: The application loads FAQ data from `faqs.json` and processes it using either:
   - SpaCy's natural language processing (preferred)
   - A simpler tokenization and TF-IDF approach (fallback)

2. **Question Matching**: When a user asks a question, the system finds the best match using:
   - Cosine similarity with SpaCy's word vectors (if available)
   - TF-IDF based similarity (fallback)

3. **Response Generation**: The system returns the answer associated with the most similar question if the similarity score exceeds a threshold.

## Project Structure

- `app.py`: Main application file containing the Flask server and NLP logic
- `faqs.json`: JSON file containing FAQ data (questions and answers)
- `templates/index.html`: HTML template for the chat interface
- `static/style.css`: CSS styling for the chat interface
- `static/script.js`: JavaScript code for handling user interactions
- `setup_and_run.bat`: Windows batch script for easy setup and execution
