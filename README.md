# AI Engineering FAQ Chatbot

A simple chatbot application that answers questions about AI Engineering using Natural Language Processing techniques.

## Features

- Advanced NLP processing using SpaCy (with intelligent fallback methods if SpaCy is unavailable)
- Smart question matching using semantic similarity, entity recognition, and intent detection
- Clean, responsive chat interface with typing indicators and animations
- Conversational context awareness and follow-up question handling
- Intelligent suggestion chips for related questions and follow-ups
- Multi-source answer generation for complex questions
- Conversation history tracking with context preservation
- Enhanced UI with timestamps, suggestion chips, and visual feedback

## Prerequisites

- Python 3.7+ installed
- pip package manager
- For optimal performance: Microsoft Visual C++ Build Tools (for SpaCy)

## Installation & Running

### First-time Setup

1. Run the `setup_and_run.bat` script
2. Follow the prompts in the console

The script will:

- Set up a Python virtual environment
- Install required dependencies (Flask, SpaCy, etc.)
- Let you choose which SpaCy language model to download:
  - Small model (~13MB): Faster but less accurate
  - Medium model (~45MB): Good balance of size and capability
  - Large model (~550MB): Best accuracy but largest download
- Start the Flask application

### Running After Setup

For subsequent runs after installation:

- Run `setup_and_run.bat run` to skip the setup process and start the application directly

This command will:

- Activate the existing virtual environment
- Start the Flask application immediately

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

1. **Enhanced Text Processing**: The application loads and processes FAQ data using:
   - SpaCy's advanced natural language processing (preferred method)
   - Enhanced TF-IDF approach with domain-specific optimizations (fallback)
   - Intelligent entity recognition and extraction

2. **Advanced Question Matching**: When a user asks a question, the system:
   - Detects question intent (definition, comparison, how-to, etc.)
   - Identifies key entities and concepts
   - Considers conversation history and context
   - Uses vector similarity with contextual boosting
   - Applies domain-specific weights for AI engineering terms

3. **Intelligent Response Generation**: The system generates responses by:
   - Selecting the best matching FAQ or combining multiple relevant FAQs
   - Maintaining conversation context for follow-up questions
   - Generating suggested follow-up questions based on the current topic
   - Providing related questions from the knowledge base
   - Adding contextual information when continuing a topic

## Project Structure

- `app.py`: Main application with Flask server, advanced NLP processing, and intelligent response generation
- `faqs.json`: Knowledge base containing FAQ data (questions and answers)
- `templates/index.html`: HTML template for the responsive chat interface
- `static/style.css`: CSS styling with animations and responsive design
- `static/script.js`: JavaScript code for enhanced UI interactions and conversation management
- `setup_and_run.bat`: Intelligent setup script with error handling and diagnostics

## Advanced Features

### Conversation Context Management

The chatbot remembers previous interactions and maintains context, allowing for natural follow-up questions without repeating information.

### Intent Recognition

The system identifies the intent behind questions (definitions, comparisons, how-to, etc.) to provide more relevant answers.

### Multi-source Answer Generation

For complex questions, the system can combine information from multiple FAQs to create comprehensive answers.

### Suggestion Chips

After answering a question, the chatbot provides suggestion chips for related topics and follow-up questions to guide the conversation.

### Graceful Degradation

The system automatically detects available capabilities and provides the best possible experience regardless of dependencies.
