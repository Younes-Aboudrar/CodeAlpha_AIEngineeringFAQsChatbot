import json
import re
import numpy as np
from collections import Counter
import math
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Flag to determine if we're using SpaCy or a simple fallback
USE_SPACY = True

# Try to load SpaCy and its model
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")  # Small English model - lighter but still effective
        print("SpaCy loaded successfully with en_core_web_sm model")
    except OSError:
        try:
            print("SpaCy model not found. Trying to download model...")
            import subprocess
            subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            print("SpaCy model downloaded and loaded successfully")
        except Exception as e:
            print(f"Error downloading SpaCy model: {e}")
            USE_SPACY = False
except ImportError:
    print("SpaCy not installed. Using fallback text processing.")
    USE_SPACY = False

# Simple fallback functions for text processing
def simple_preprocess_text(text):
    """
    Simple text preprocessing without spaCy
    - Convert to lowercase
    - Remove special characters
    - Tokenize into words
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    
    # Very simple stopwords list - can be expanded
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of'}
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stopwords]
    
    return tokens

def calculate_tf_idf_similarity(query_tokens, doc_tokens):
    """
    Calculate a simple TF-IDF based similarity between query and document tokens
    """
    # Convert to Counter objects to count term frequencies
    query_tf = Counter(query_tokens)
    doc_tf = Counter(doc_tokens)
    
    # Get all unique terms
    all_terms = set(query_tokens) | set(doc_tokens)
    
    # Simple dot product
    dot_product = sum(query_tf[term] * doc_tf[term] for term in all_terms)
    
    # Calculate magnitudes
    query_magnitude = math.sqrt(sum(count * count for count in query_tf.values()))
    doc_magnitude = math.sqrt(sum(count * count for count in doc_tf.values()))
    
    # Avoid division by zero
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
        
    # Return cosine similarity
    return dot_product / (query_magnitude * doc_magnitude)

# Load FAQ data
with open('faqs.json', 'r') as f:
    faqs = json.load(f)

# Initialize processed questions list
processed_questions = []

# Preprocess FAQ questions based on available method
if USE_SPACY:
    print("Using SpaCy for text processing")
    for faq in faqs:
        # Process each question with spaCy
        doc = nlp(faq["question"])
        # Store the processed document and the corresponding FAQ
        processed_questions.append({"doc": doc, "faq": faq})
else:
    print("Using fallback text processing")
    for faq in faqs:
        # Process each question with our simple tokenizer
        tokens = simple_preprocess_text(faq["question"])
        # Store the tokens and the corresponding FAQ
        processed_questions.append({"tokens": tokens, "faq": faq})

def preprocess_text(text):
    """
    Preprocess user input using available method
    """
    if USE_SPACY:
        return nlp(text)
    else:
        return simple_preprocess_text(text)

def get_best_match(user_question, threshold=0.65):
    """
    Find the best matching FAQ for the user's question using available similarity method
    
    Args:
        user_question (str): The question asked by the user
        threshold (float): Minimum similarity score to consider a match valid
        
    Returns:
        dict: The best matching FAQ or a default response if no good match is found
    """
    # Process the user's question
    user_processed = preprocess_text(user_question)
    
    # If the user's question is too short or empty
    if USE_SPACY and len(user_processed) < 2:
        return {"question": "", "answer": "Please provide a longer question for me to understand better."}
    elif not USE_SPACY and len(user_processed) < 2:
        return {"question": "", "answer": "Please provide a longer question for me to understand better."}
    
    best_similarity = -1
    best_match = None
    
    # Calculate similarity with each FAQ question
    for processed in processed_questions:
        if USE_SPACY:
            similarity = user_processed.similarity(processed["doc"])
        else:
            similarity = calculate_tf_idf_similarity(user_processed, processed["tokens"])
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = processed["faq"]
    
    # Adjust threshold for fallback method
    actual_threshold = threshold
    if not USE_SPACY:
        actual_threshold = 0.1  # Lower threshold for fallback method
    
    # If the best similarity is below our threshold, return a default response
    if best_similarity < actual_threshold:
        return {"question": "", "answer": "I don't have a specific answer for that. Please try rephrasing your question about AI Engineering."}
    
    return best_match

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question', '')
    
    # Get the best matching FAQ
    best_match = get_best_match(user_question)
    
    return jsonify({
        'question': best_match['question'],
        'answer': best_match['answer']
    })

if __name__ == '__main__':
    print("AI Engineering FAQ Chatbot is running!")
    app.run(debug=True)
