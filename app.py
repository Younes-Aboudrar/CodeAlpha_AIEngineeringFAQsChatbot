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

# Enhanced fallback functions for text processing
def simple_preprocess_text(text):
    """
    Enhanced text preprocessing without spaCy
    - Convert to lowercase
    - Remove special characters
    - Tokenize into words
    - Remove stopwords
    - Extract key phrases and concepts
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Split into tokens
    tokens = text.split()
    
    # More comprehensive stopwords list
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
        'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'of', 'from',
        'as', 'this', 'that', 'these', 'those', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'should', 'can', 'could',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    }
    
    # Remove stopwords but preserve question keywords which are important for matching
    question_words = {'what', 'why', 'who', 'when', 'where', 'which', 'how'}
    tokens = [token for token in tokens if token not in stopwords or token in question_words]
    
    # Find and add bigrams (pairs of consecutive words) that might be important concepts
    bigrams = []
    original_tokens = text.split()
    for i in range(len(original_tokens) - 1):
        if original_tokens[i] not in stopwords or original_tokens[i+1] not in stopwords:
            bigram = f"{original_tokens[i]}_{original_tokens[i+1]}"
            bigrams.append(bigram)
    
    # Add important technical terms related to AI (domain-specific)
    ai_terms = {
        'ai', 'ml', 'machine', 'learning', 'deep', 'neural', 'network', 'data',
        'algorithm', 'model', 'training', 'supervised', 'unsupervised', 'reinforcement',
        'nlp', 'computer', 'vision', 'cnn', 'rnn', 'lstm', 'gan', 'transformer',
        'bert', 'gpt', 'overfitting', 'bias', 'feature', 'classification', 'regression',
        'clustering', 'engineering'
    }
    
    # Boost importance of domain-specific terms
    boosted_tokens = []
    for token in tokens:
        if token in ai_terms:
            # Add important terms multiple times to increase their weight
            boosted_tokens.extend([token] * 3)
        else:
            boosted_tokens.append(token)
    
    # Combine everything
    return boosted_tokens + bigrams

def calculate_tf_idf_similarity(query_tokens, doc_tokens):
    """
    Calculate an enhanced TF-IDF based similarity between query and document tokens
    with improved weighting and partial matching
    """
    # Convert to Counter objects to count term frequencies
    query_tf = Counter(query_tokens)
    doc_tf = Counter(doc_tokens)
    
    # Get all unique terms
    all_terms = set(query_tokens) | set(doc_tokens)
    
    # Calculate document frequencies across all FAQs for better IDF
    global doc_frequencies
    if 'doc_frequencies' not in globals():
        # Calculate this only once for efficiency
        doc_frequencies = Counter()
        for processed in processed_questions:
            if not USE_SPACY:  # Only needed for fallback mode
                unique_terms = set(processed["tokens"])
                for term in unique_terms:
                    doc_frequencies[term] += 1
    
    # Number of documents for IDF calculation
    num_docs = len(processed_questions)
    
    # Enhanced dot product with IDF weighting
    dot_product = 0
    for term in all_terms:
        # Calculate TF values
        query_term_tf = query_tf[term]
        doc_term_tf = doc_tf[term]
        
        # Skip if the term doesn't appear in both
        if query_term_tf == 0 or doc_term_tf == 0:
            # For partial matching: if term is in query but not in doc,
            # check if any similar term exists in doc (handles typos and variations)
            if query_term_tf > 0:
                for doc_term in doc_tf:
                    # Check for terms that share a prefix (3+ chars)
                    if len(term) >= 4 and len(doc_term) >= 4:
                        if term[:3] == doc_term[:3]:
                            # Found similar term, add partial contribution
                            doc_term_tf = doc_tf[doc_term] * 0.5  # Partial weight
                            break
            
            # If still no match, continue to next term
            if doc_term_tf == 0:
                continue
        
        # Calculate IDF (Inverse Document Frequency)
        idf = math.log(num_docs / (1 + doc_frequencies.get(term, 1)))
        
        # Apply higher weight to technical terms
        term_importance = 1.0
        if any(keyword in term.lower() for keyword in ['ai', 'ml', 'data', 'model', 'learning', 
                                                       'neural', 'algorithm', 'train']):
            term_importance = 2.5
        
        # Add to dot product with TF-IDF weighting
        dot_product += query_term_tf * doc_term_tf * idf * term_importance
    
    # Calculate magnitudes with IDF weighting
    query_magnitude = math.sqrt(sum((query_tf[term] * math.log(num_docs / (1 + doc_frequencies.get(term, 1))))**2 
                             for term in query_tf))
    doc_magnitude = math.sqrt(sum((doc_tf[term] * math.log(num_docs / (1 + doc_frequencies.get(term, 1))))**2 
                           for term in doc_tf))
    
    # Avoid division by zero
    if query_magnitude == 0 or doc_magnitude == 0:
        return 0
        
    # Calculate cosine similarity
    cosine_sim = dot_product / (query_magnitude * doc_magnitude)
    
    # Apply length normalization to prefer longer matches
    # This helps when the user query is very long and detailed
    query_length_factor = min(1.0, len(query_tokens) / 15)  # Normalize to max 1.0
    length_bonus = 0.1 * query_length_factor
    
    return cosine_sim + length_bonus

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
    Enhanced preprocessing of user input to better understand user intent
    - Normalizes text
    - Handles special cases
    - Expands abbreviations
    - Applies appropriate processing method
    """
    if not text:
        return [] if not USE_SPACY else nlp("")
    
    # Normalize text
    text = text.strip()
    
    # Handle greeting patterns that aren't questions
    greeting_patterns = [
        r'^hi\b', r'^hello\b', r'^hey\b', r'^greetings\b', 
        r'^good (morning|afternoon|evening)\b', r'^howdy\b'
    ]
    
    # Check if this is just a greeting
    is_just_greeting = any(re.match(pattern, text.lower()) for pattern in greeting_patterns) and len(text.split()) < 3
    
    # If it's just a greeting, expand it to a question about AI
    if is_just_greeting:
        text = text + " What is AI Engineering?"
    
    # Expand common abbreviations
    abbreviations = {
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'nn': 'neural network',
        'nlp': 'natural language processing',
        'cv': 'computer vision',
        'rl': 'reinforcement learning'
    }
    
    # Apply abbreviation expansion
    words = text.lower().split()
    for i, word in enumerate(words):
        if word in abbreviations:
            # Only expand stand-alone abbreviations, not within other words
            if (i == 0 or words[i-1] in [',', '.', '?', '!', ';', ':', '(']) and \
               (i == len(words)-1 or words[i+1] in [',', '.', '?', '!', ';', ':', ')']):
                words[i] = abbreviations[word]
    
    # Reconstruct text with expanded abbreviations
    expanded_text = ' '.join(words)
    
    # Apply appropriate processing method
    if USE_SPACY:
        return nlp(expanded_text)
    else:
        return simple_preprocess_text(expanded_text)

def extract_question_intent(user_question):
    """
    Extract the intent and key concepts from a user question
    """
    # Convert to lowercase for matching
    question = user_question.lower()
    
    # Identify question type based on patterns
    intent = {
        'definition': False,
        'comparison': False,
        'example': False,
        'how_to': False,
        'listing': False,
        'yes_no': False
    }
    
    # Definition questions
    if re.search(r'what\s+is|define|meaning|explain|describe', question):
        intent['definition'] = True
    
    # Comparison questions
    if re.search(r'(compare|versus|vs\.?|difference|how\s+does.*differ|better)', question):
        intent['comparison'] = True
    
    # Example requests
    if re.search(r'(example|instance|show|illustrate)', question):
        intent['example'] = True
    
    # How-to questions
    if re.search(r'how\s+to|steps|procedure|guide|implement|create', question):
        intent['how_to'] = True
    
    # Listing questions
    if re.search(r'(list|what are|types of|kinds of|categories|all)', question):
        intent['listing'] = True
    
    # Yes/No questions
    if re.search(r'^(is|are|can|does|do|will|has|have|should)', question):
        intent['yes_no'] = True
    
    # Extract key entities - AI-related terms
    entities = []
    ai_terms = [
        'machine learning', 'deep learning', 'neural network', 'supervised learning',
        'unsupervised learning', 'reinforcement learning', 'nlp', 'natural language processing',
        'computer vision', 'cnn', 'rnn', 'lstm', 'gan', 'transformer', 'bert', 'gpt', 
        'overfitting', 'bias', 'feature engineering', 'classification', 'regression',
        'clustering', 'tensorflow', 'pytorch', 'ai ethics'
    ]
    
    for term in ai_terms:
        if term in question:
            entities.append(term)
    
    return intent, entities

def get_best_match(user_question, threshold=0.65):
    """
    Find the best matching FAQ for the user's question using advanced matching techniques
    
    Args:
        user_question (str): The question asked by the user
        threshold (float): Minimum similarity score to consider a match valid
        
    Returns:
        dict: The best matching FAQ or a default response if no good match is found
    """
    # Process the user's question
    user_processed = preprocess_text(user_question)
    
    # If the user's question is too short or empty
    if (USE_SPACY and len(user_processed) < 2) or (not USE_SPACY and len(user_processed) < 2):
        return {"question": "", "answer": "Please provide a longer question for me to understand better."}
    
    # Extract intent and key entities from the user question
    intent, entities = extract_question_intent(user_question)
    
    # Initialize variables to track results
    best_similarity = -1
    best_match = None
    top_matches = []
    
    # Calculate similarity with each FAQ question
    for processed in processed_questions:
        if USE_SPACY:
            # Use SpaCy's built-in vector similarity
            similarity = user_processed.similarity(processed["doc"])
        else:
            # Use our enhanced TF-IDF similarity
            similarity = calculate_tf_idf_similarity(user_processed, processed["tokens"])
        
        # Store current FAQ and its similarity score
        faq_entry = {
            "faq": processed["faq"],
            "similarity": similarity
        }
        
        # Add to our top matches list
        top_matches.append(faq_entry)
        
        # Update best match if this is better
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = processed["faq"]
    
    # Sort matches by similarity score
    top_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Get top 3 matches for consideration
    top_3_matches = top_matches[:3]
    
    # Adjust threshold based on the method and query complexity
    actual_threshold = threshold
    if not USE_SPACY:
        actual_threshold = 0.1  # Lower threshold for fallback method
    
    # If we have entities detected, boost matches that contain these entities
    if entities and top_3_matches:
        for match in top_3_matches:
            faq = match["faq"]
            # Check if any of the extracted entities appear in the FAQ
            for entity in entities:
                if entity.lower() in faq["question"].lower() or entity.lower() in faq["answer"].lower():
                    # Boost the similarity score
                    match["similarity"] += 0.1
        
        # Re-sort after boosting
        top_3_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Update best match
        if top_3_matches:
            best_match = top_3_matches[0]["faq"]
            best_similarity = top_3_matches[0]["similarity"]
    
    # Intent matching boost - if we detected a specific intent, favor FAQs that match it
    if any(intent.values()) and top_3_matches:
        for match in top_3_matches:
            faq = match["faq"]
            # Definition questions should match FAQs that define concepts
            if intent['definition'] and re.search(r'what is|is a|refers to', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.05
            
            # How-to questions should match FAQs with instructions
            if intent['how_to'] and re.search(r'how (to|do|does|can)', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.05
            
            # Comparison questions should match FAQs that compare things
            if intent['comparison'] and re.search(r'difference|versus|compared|differ', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.05
        
        # Re-sort after intent boosting
        top_3_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Update best match
        if top_3_matches:
            best_match = top_3_matches[0]["faq"]
            best_similarity = top_3_matches[0]["similarity"]
    
    # If the best similarity is below our threshold, try to provide a helpful response
    if best_similarity < actual_threshold:
        # Provide a more helpful response based on detected intent
        if intent['definition']:
            terms = ', '.join(entities) if entities else "this concept"
            return {
                "question": "", 
                "answer": f"I don't have a specific definition for {terms}. Please try asking about another AI Engineering concept."
            }
        elif intent['how_to']:
            return {
                "question": "", 
                "answer": "I don't have specific instructions for that. Could you try asking about a different AI Engineering technique or concept?"
            }
        else:
            return {
                "question": "", 
                "answer": "I don't have a specific answer for that. Please try rephrasing your question about AI Engineering or ask about a different topic."
            }
    
    return best_match

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question', '')
    
    # Handle special commands or queries
    if user_question.lower() in ['help', 'help me', 'what can you do?', 'what can i ask?', 'what do you know?']:
        return jsonify({
            'question': "What can I ask you?",
            'answer': "You can ask me questions about AI Engineering! For example: What is machine learning? How does deep learning work? What's the difference between supervised and unsupervised learning? What skills do I need for AI Engineering? Feel free to ask about concepts, techniques, career advice, or anything related to AI!"
        })
    
    # Detect thanks/goodbye patterns
    thanks_patterns = [r'^thank', r'^thanks', r'appreciate it', r'helpful']
    bye_patterns = [r'^bye', r'^goodbye', r'^see you', r'^exit', r'^quit']
    
    if any(re.search(pattern, user_question.lower()) for pattern in thanks_patterns):
        return jsonify({
            'question': "",
            'answer': "You're welcome! Feel free to ask if you have any other questions about AI Engineering."
        })
        
    if any(re.search(pattern, user_question.lower()) for pattern in bye_patterns):
        return jsonify({
            'question': "",
            'answer': "Goodbye! Feel free to come back anytime you have questions about AI Engineering."
        })
    
    # Handle empty or very short queries
    if not user_question or len(user_question.strip()) < 2:
        return jsonify({
            'question': "",
            'answer': "Please type a question about AI Engineering, and I'll do my best to help you!"
        })
    
    # Get the best matching FAQ
    best_match = get_best_match(user_question)
    
    # Add a contextual prefix to answers for better conversation flow
    answer_prefix = ""
    
    # Check if it's a direct question
    if re.search(r'^what|^how|^why|^when|^where|^which|^who|^can|^do|^is|^are', user_question.lower()):
        # This is a direct question, maybe use a prefix like:
        if len(user_question) < 20:  # If it's a short, direct question
            answer_prefix = "Here's what I found: "
    
    # Format the answer with any needed prefix
    final_answer = answer_prefix + best_match['answer']
    
    return jsonify({
        'question': best_match['question'],
        'answer': final_answer
    })

if __name__ == '__main__':
    print("AI Engineering FAQ Chatbot is running!")
    app.run(debug=True)
