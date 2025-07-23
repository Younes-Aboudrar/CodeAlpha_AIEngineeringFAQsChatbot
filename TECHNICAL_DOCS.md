# AI Engineering FAQ Chatbot - Technical Documentation

## Overview

This document provides technical details about the implementation of the AI Engineering FAQ Chatbot. The system uses natural language processing techniques to match user queries with the most relevant pre-defined FAQs about AI Engineering topics.

The application is designed with a modular, fault-tolerant architecture that separates core functionality from enhanced NLP capabilities:

- **Core Application**: The main Flask application (`app.py`) handling HTTP requests, question processing, and response generation
- **Knowledge Base**: A structured collection of FAQs in `faqs.json` covering key AI Engineering topics
- **Natural Language Processing**: Dual-mode text processing with SpaCy (when available) or a custom TF-IDF fallback system
- **User Interface**: Responsive web interface implemented with HTML, CSS, and JavaScript
- **Setup Utilities**:
  - `setup_and_run.bat`: Sets up the Python environment and installs basic dependencies
  - `spacy_setup.py`: Dedicated utility for SpaCy model selection and installation

## Technical Stack

- **Backend**: Python 3.7+ with Flask for server-side processing and API endpoints
- **Natural Language Processing**:
  - **Primary**: SpaCy with configurable language models (small, medium, or large)
  - **Fallback**: TF-IDF implementation from scikit-learn when SpaCy is unavailable
- **Text Matching**: Vector similarity (SpaCy) or TF-IDF cosine similarity (fallback)
- **Frontend**: HTML/CSS with JavaScript for interactive chat experience
- **Configuration**: JSON-based configuration for SpaCy model persistence

## NLP Implementation Details

### Text Processing Approaches

The system employs a dual-mode text processing approach with automatic fallback:

#### SpaCy Processing (Primary Method)

When SpaCy is installed and properly configured:

1. **Model Loading**: The system loads the SpaCy model specified in `config.json`
2. **Document Processing**: User questions are processed into SpaCy Doc objects
3. **Vector Representation**: Text is converted into numerical vectors that capture semantic meaning
4. **Similarity Calculation**: SpaCy's vector similarity is used to compare questions

The user can select from three SpaCy model options via the `spacy_setup.py` utility:

- **Small Model** (~12MB): Fastest but least accurate
- **Medium Model** (~40MB): Good balance of size and capability
- **Large Model** (~560MB): Best accuracy but largest download

#### TF-IDF Fallback Processing

If SpaCy is unavailable, the system activates a fallback mechanism:

1. **Standard Tokenization**: Basic text processing to separate words
2. **TF-IDF Vectorization**: Converting text to sparse vectors using scikit-learn
3. **Cosine Similarity**: Calculating similarity between TF-IDF vectors

### Question Matching Process

The system matches user questions to the FAQ database using:

1. **Text Processing**: Process user question with SpaCy (if available) or TF-IDF
2. **Similarity Calculation**: Using either:
   - SpaCy's vector similarity (when SpaCy is available)
   - TF-IDF cosine similarity (fallback method)
3. **Match Selection**: Identifying the most similar questions in the knowledge base
4. **Threshold Filtering**: Only returning matches above a minimum confidence threshold
5. **Response Generation**: Returning the answer from the best-matching FAQ entry

## Web Interface

The chatbot provides a responsive web interface with:

- **Clean Design**: Simple, intuitive chat interface with clear distinction between user and bot messages
- **AJAX Communication**: Asynchronous JavaScript for seamless interaction without page reloads
- **Responsive Layout**: Adapts to different screen sizes for desktop and mobile use
- **Message Formatting**: Properly displays answers with preserved formatting
- **Loading Indicator**: Visual feedback while waiting for responses

## Configuration System

### SpaCy Model Management

The application uses a configuration system for SpaCy integration:

- **Setup Utility**: The `spacy_setup.py` script handles model selection and installation
- **User Choice**: Interactive prompts allow selection of small, medium, or large models
- **Configuration File**: Selected model is stored in `config.json` for future use
- **Automatic Detection**: The application checks for SpaCy and configured models at startup
- **Fallback Mechanism**: Automatically switches to TF-IDF when SpaCy is unavailable

### Environment Setup

The project includes automated setup with:

- **Virtual Environment**: Creation of an isolated Python environment
- **Dependency Management**: Installation of required packages from `requirements.txt`
- **Error Handling**: Detection and reporting of common installation issues
- **Separate SpaCy Setup**: Two-phase installation to manage large language model downloads

## Application Structure

### Core Components

The chatbot consists of several key components:

- **Flask Application (`app.py`)**:
  - Handles HTTP requests and serves the web application
  - Processes user questions using the appropriate NLP method
  - Matches questions against the knowledge base
  - Returns formatted responses to the client

- **Knowledge Base (`faqs.json`)**:
  - Contains structured question-answer pairs
  - Serves as the source of information for the chatbot

- **SpaCy Setup Utility (`spacy_setup.py`)**:
  - Prompts for model selection (small/medium/large)
  - Downloads and installs the selected SpaCy model
  - Creates/updates the `config.json` file

- **Setup Script (`setup_and_run.bat`)**:
  - Creates a virtual environment
  - Installs required Python packages
  - Provides error handling for common setup issues

## Setup Process

The system uses a three-step setup process:

1. **Environment Setup**: Run `setup_and_run.bat` to:
   - Create a Python virtual environment
   - Install core dependencies from `requirements.txt`

2. **SpaCy Configuration**: Run `python spacy_setup.py` to:
   - Select a SpaCy model (small, medium, or large)
   - Download and install the selected model
   - Create a `config.json` file with the model name

3. **Application Launch**: Run `python app.py` to:
   - Start the Flask server
   - Load the appropriate NLP processing method
   - Serve the web interface
