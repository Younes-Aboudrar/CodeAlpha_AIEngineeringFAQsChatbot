import json
import re
import numpy as np
from collections import Counter
import math
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize session storage for conversation history
# We'll use this to maintain context across multiple questions
from flask import session

# Enable session with a secure key
app.secret_key = "ai_engineering_chatbot_secure_key_2025"

# Maximum conversation history to remember
MAX_CONVERSATION_HISTORY = 5

# Flag to determine if we're using SpaCy or a simple fallback
USE_SPACY = False
nlp = None

# Try to load SpaCy and its model from config
try:
    import spacy
    import os
    
    # Check if config.json exists
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as config_file:
                config = json.load(config_file)
                if "spacy_model" in config:
                    model_name = config["spacy_model"]
                    try:
                        nlp = spacy.load(model_name)
                        print(f"SpaCy loaded successfully with {model_name} model from config")
                        USE_SPACY = True
                    except OSError as e:
                        print(f"Error loading SpaCy model '{model_name}' from config: {e}")
                        print("Using fallback text processing. Run spacy_setup.py to setup SpaCy properly.")
                else:
                    print("No SpaCy model specified in config.json.")
                    print("Using fallback text processing. Run spacy_setup.py to setup SpaCy properly.")
        except Exception as e:
            print(f"Error reading config.json: {e}")
            print("Using fallback text processing. Run spacy_setup.py to setup SpaCy properly.")
    else:
        print("Config file not found. Using fallback text processing.")
        print("Run spacy_setup.py to setup SpaCy and download a language model.")
except ImportError:
    print("SpaCy not installed. Using fallback text processing.")
    print("Run spacy_setup.py to setup SpaCy properly.")

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

def extract_question_intent(user_question, conversation_history=None):
    """
    Extract the intent, key concepts, and context from a user question
    
    Args:
        user_question (str): The user's current question
        conversation_history (list): Optional list of previous Q&A pairs
        
    Returns:
        dict: Intent information
        list: Detected entities
        dict: Context information
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
        'yes_no': False,
        'opinion': False,
        'clarification': False,
        'follow_up': False,
        'command': False,
        'greeting': False,
        'farewell': False
    }
    
    # Definition questions
    if re.search(r'what\s+is|what\'s|define|meaning|explain|describe|tell me about', question):
        intent['definition'] = True
    
    # Comparison questions
    if re.search(r'(compare|versus|vs\.?|difference|how\s+does.*differ|better|which is|preferred|advantages|disadvantages)', question):
        intent['comparison'] = True
    
    # Example requests
    if re.search(r'(example|instance|show|illustrate|demonstrate|sample|such as)', question):
        intent['example'] = True
    
    # How-to questions
    if re.search(r'how\s+to|how\s+do|steps|procedure|guide|implement|create|build|develop|make|setup|configure', question):
        intent['how_to'] = True
    
    # Listing questions
    if re.search(r'(list|what are|types of|kinds of|categories|all|various|different|enumerate)', question):
        intent['listing'] = True
    
    # Yes/No questions
    if re.search(r'^(is|are|can|does|do|will|has|have|should|would|could|might)', question):
        intent['yes_no'] = True
        
    # Opinion questions
    if re.search(r'(what do you think|opinion|thoughts|view|perspective|recommend|suggest|advise)', question):
        intent['opinion'] = True
        
    # Clarification questions
    if re.search(r'(what do you mean|clarify|elaborate|explain more|in other words|rephrase)', question):
        intent['clarification'] = True
        
    # Command or request
    if re.search(r'^(tell|show|give|list|explain|define|compare|help|find)', question):
        intent['command'] = True
        
    # Greeting detection
    if re.search(r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)', question):
        intent['greeting'] = True
        
    # Farewell detection
    if re.search(r'^(goodbye|bye|see you|exit|quit|end)', question):
        intent['farewell'] = True
        
    # Follow-up question detection
    if re.search(r'^(and|but|so|what about|how about|what if|also|additionally)', question) or len(question.split()) <= 4:
        intent['follow_up'] = True
    
    # Extract key entities - AI-related terms (expanded list)
    entities = []
    ai_terms = [
        # Core AI concepts
        'machine learning', 'deep learning', 'neural network', 'supervised learning',
        'unsupervised learning', 'reinforcement learning', 'nlp', 'natural language processing',
        'computer vision', 'cnn', 'rnn', 'lstm', 'gan', 'transformer', 'bert', 'gpt', 
        'overfitting', 'bias', 'feature engineering', 'classification', 'regression',
        'clustering', 'tensorflow', 'pytorch', 'ai ethics', 'artificial intelligence',
        
        # Advanced concepts
        'attention mechanism', 'transfer learning', 'generative ai', 'federated learning',
        'quantum ai', 'explainable ai', 'xai', 'ai alignment', 'agent', 'multi-agent',
        'knowledge graph', 'semantic web', 'computer vision', 'image recognition',
        'object detection', 'semantic segmentation', 'instance segmentation',
        
        # Techniques and methods
        'backpropagation', 'gradient descent', 'activation function', 'loss function',
        'hyperparameter', 'parameter', 'regularization', 'dropout', 'batch normalization',
        'embedding', 'fine-tuning', 'pre-training', 'few-shot learning', 'zero-shot learning',
        
        # Application domains
        'recommendation system', 'fraud detection', 'sentiment analysis', 'chatbot',
        'autonomous vehicle', 'robotics', 'healthcare ai', 'financial ai', 'predictive maintenance',
        'anomaly detection', 'speech recognition', 'text generation', 'translation', 'summarization',
        
        # Tools and libraries
        'numpy', 'pandas', 'scikit-learn', 'keras', 'tensorflow', 'pytorch', 'hugging face',
        'transformers', 'opencv', 'nltk', 'spacy', 'jupyter', 'mlflow', 'kubeflow', 'mlops',
        
        # Career and skills
        'data scientist', 'machine learning engineer', 'ai engineer', 'data engineer',
        'devops', 'mlops', 'data analysis', 'statistics', 'mathematics', 'calculus',
        'linear algebra', 'probability', 'algorithm', 'data structure'
    ]
    
    # Check for AI terms in the question
    for term in ai_terms:
        if term in question:
            entities.append(term)
    
    # Also check for partial matches for single-word terms
    single_word_ai_terms = [term for term in ai_terms if ' ' not in term]
    for term in single_word_ai_terms:
        # Check for terms with word boundaries to avoid partial word matches
        if re.search(r'\b' + re.escape(term) + r'\b', question):
            if term not in entities:  # Avoid duplicates
                entities.append(term)
    
    # Extract context from conversation history
    context = {
        'referenced_entities': [],
        'previous_topic': None,
        'has_previous_question': False,
        'pronoun_reference': False,
    }
    
    # If there's conversation history, analyze it for context
    if conversation_history and len(conversation_history) > 0:
        # Check for pronouns that might reference previous topics
        if re.search(r'\b(it|this|that|these|those|they|them|he|she|his|her|its|their)\b', question):
            context['pronoun_reference'] = True
            
        # Get the most recent Q&A pair
        last_qa = conversation_history[-1]
        if 'entities' in last_qa and last_qa['entities']:
            context['referenced_entities'] = last_qa['entities']
            context['previous_topic'] = last_qa['entities'][0]  # Use the first entity as main topic
            
        context['has_previous_question'] = True
        
        # If this is likely a follow-up question and very short, add the entities from the previous question
        if intent['follow_up'] and len(question.split()) < 6 and context['referenced_entities']:
            entities.extend(context['referenced_entities'])
    
    return intent, entities, context

def compose_answer_from_multiple_faqs(top_matches, user_question, intent):
    """
    Creates a composite answer from multiple related FAQs when appropriate
    
    Args:
        top_matches: List of top matching FAQs with similarity scores
        user_question: The original user question
        intent: The detected intent dictionary
        
    Returns:
        dict: A composed answer or None if not applicable
    """
    # Only use this for certain types of questions where composing makes sense
    if not (intent['listing'] or intent['comparison'] or 
            (len(top_matches) >= 3 and all(m["similarity"] > 0.5 for m in top_matches[:3]))):
        return None
        
    # For listing questions, try to combine information
    if intent['listing'] and len(top_matches) >= 2:
        # Extract top 3 matches that have good scores
        good_matches = [m for m in top_matches[:5] if m["similarity"] > 0.4]
        
        if len(good_matches) >= 2:
            # Create a composite answer
            answer = "Based on several sources:\n\n"
            
            # Add each relevant answer
            for i, match in enumerate(good_matches[:3]):  # Limit to top 3
                faq = match["faq"]
                answer += f"{i+1}. {faq['answer']}\n\n"
            
            return {
                "question": user_question,
                "answer": answer.strip()
            }
    
    # For comparison questions with multiple good matches
    if intent['comparison'] and len(top_matches) >= 2:
        # Check if we have good matches for the comparison
        if top_matches[0]["similarity"] > 0.5 and top_matches[1]["similarity"] > 0.4:
            faq1 = top_matches[0]["faq"]
            faq2 = top_matches[1]["faq"]
            
            # Create a composite comparative answer
            answer = f"Let me compare these concepts for you:\n\n"
            answer += f"Regarding {faq1['question'].strip('?')}:\n{faq1['answer']}\n\n"
            answer += f"Regarding {faq2['question'].strip('?')}:\n{faq2['answer']}"
            
            return {
                "question": user_question,
                "answer": answer
            }
    
    return None

def combine_with_context(user_question, best_match, conversation_history, intent, entities, context):
    """
    Enhances the answer by considering conversation context
    
    Args:
        user_question: The original user question
        best_match: The best matching FAQ
        conversation_history: Previous conversation turns
        intent: The detected intent
        entities: The detected entities
        context: The conversation context
        
    Returns:
        dict: Enhanced answer with context
    """
    # If there's no conversation history or no context indicators, return the original match
    if not conversation_history or len(conversation_history) == 0:
        return best_match
        
    # Handle follow-up questions
    if intent['follow_up'] and context['has_previous_question']:
        # Get the most recent Q&A
        last_qa = conversation_history[-1]
        
        # Add context from previous question if this is clearly a follow-up
        if len(user_question.split()) <= 5 or context['pronoun_reference']:
            # Create an enhanced answer that acknowledges the follow-up nature
            return {
                "question": best_match["question"],
                "answer": f"Following up on {last_qa.get('topic', 'your previous question')}: {best_match['answer']}"
            }
    
    # Handle clarification requests
    if intent['clarification'] and conversation_history:
        last_qa = conversation_history[-1]
        return {
            "question": best_match["question"],
            "answer": f"To clarify: {best_match['answer']}"
        }
        
    return best_match

def get_best_match(user_question, threshold=0.65, conversation_history=None):
    """
    Find the best matching FAQ for the user's question using advanced matching techniques
    with conversation context awareness
    
    Args:
        user_question (str): The question asked by the user
        threshold (float): Minimum similarity score to consider a match valid
        conversation_history (list): Optional list of previous Q&A exchanges
        
    Returns:
        dict: The best matching FAQ or a default response if no good match is found
        list: Detected entities for future context
        dict: Additional metadata about the match
    """
    # Process the user's question
    user_processed = preprocess_text(user_question)
    
    # If the user's question is too short or empty
    if (USE_SPACY and len(user_processed) < 2) or (not USE_SPACY and len(user_processed) < 2):
        return {
            "question": "", 
            "answer": "Please provide a longer question for me to understand better."
        }, [], {"confidence": 0}
    
    # Extract intent, entities and context from the user question
    intent, entities, context = extract_question_intent(user_question, conversation_history)
    
    # Initialize variables to track results
    best_similarity = -1
    best_match = None
    top_matches = []
    
    # Calculate similarity with each FAQ question
    for processed in processed_questions:
        similarity = 0
        
        if USE_SPACY:
            # Use SpaCy's built-in vector similarity
            similarity = user_processed.similarity(processed["doc"])
        else:
            # Use our enhanced TF-IDF similarity
            similarity = calculate_tf_idf_similarity(user_processed, processed["tokens"])
        
        # Apply context boosting if this is a follow-up question
        if context['has_previous_question'] and context['pronoun_reference']:
            # Get the FAQ question
            faq = processed["faq"]["question"].lower()
            
            # If previous entities appear in this FAQ, boost its score
            for entity in context['referenced_entities']:
                if entity.lower() in faq:
                    similarity += 0.15  # Significant boost for contextual continuity
        
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
    
    # Get top matches for consideration
    top_5_matches = top_matches[:5]
    
    # Adjust threshold based on the method and query complexity
    actual_threshold = threshold
    if not USE_SPACY:
        actual_threshold = 0.1  # Lower threshold for fallback method
    
    # Entity and intent boosting
    # If we have entities detected, boost matches that contain these entities
    if entities and top_5_matches:
        for match in top_5_matches:
            faq = match["faq"]
            # Check if any of the extracted entities appear in the FAQ
            for entity in entities:
                if entity.lower() in faq["question"].lower() or entity.lower() in faq["answer"].lower():
                    # Boost the similarity score
                    match["similarity"] += 0.1
        
        # Re-sort after boosting
        top_5_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Update best match
        if top_5_matches:
            best_match = top_5_matches[0]["faq"]
            best_similarity = top_5_matches[0]["similarity"]
    
    # Intent matching boost - if we detected a specific intent, favor FAQs that match it
    if any(intent.values()) and top_5_matches:
        for match in top_5_matches:
            faq = match["faq"]
            # Definition questions should match FAQs that define concepts
            if intent['definition'] and re.search(r'what is|is a|refers to|definition|mean', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.08
            
            # How-to questions should match FAQs with instructions
            if intent['how_to'] and re.search(r'how (to|do|does|can)|steps|guide|implement', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.08
            
            # Comparison questions should match FAQs that compare things
            if intent['comparison'] and re.search(r'difference|versus|compared|differ|better|advantage', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.08
                
            # Example questions should match FAQs with examples
            if intent['example'] and re.search(r'example|case|instance|scenario', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.08
                
            # Listing questions should match FAQs with lists
            if intent['listing'] and re.search(r'list|what are|types|kinds|categories', faq["question"], re.IGNORECASE):
                match["similarity"] += 0.08
        
        # Re-sort after intent boosting
        top_5_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Update best match
        if top_5_matches:
            best_match = top_5_matches[0]["faq"]
            best_similarity = top_5_matches[0]["similarity"]
    
    # Try to compose an answer from multiple FAQs if appropriate
    composite_answer = compose_answer_from_multiple_faqs(top_5_matches, user_question, intent)
    if composite_answer:
        # Add metadata about the match
        metadata = {
            "confidence": best_similarity,
            "is_composite": True,
            "matches_used": len([m for m in top_5_matches[:3] if m["similarity"] > 0.4])
        }
        return composite_answer, entities, metadata
    
    # If the best similarity is below our threshold, try to provide a helpful response
    if best_similarity < actual_threshold:
        # Provide a more helpful response based on detected intent
        if intent['definition']:
            terms = ', '.join(entities) if entities else "this concept"
            response = {
                "question": "", 
                "answer": f"I don't have a specific definition for {terms}. Please try asking about another AI Engineering concept."
            }
            return response, entities, {"confidence": 0}
            
        elif intent['how_to']:
            response = {
                "question": "", 
                "answer": "I don't have specific instructions for that. Could you try asking about a different AI Engineering technique or concept?"
            }
            return response, entities, {"confidence": 0}
            
        elif intent['comparison']:
            response = {
                "question": "", 
                "answer": "I don't have enough information to make that comparison. Could you try asking about specific AI Engineering concepts to compare?"
            }
            return response, entities, {"confidence": 0}
            
        else:
            # Try to suggest related topics based on detected entities
            suggestions = ""
            if entities:
                related_faqs = []
                for entity in entities:
                    for faq in faqs[:5]:  # Look at first few FAQs
                        if entity.lower() in faq["question"].lower():
                            related_faqs.append(faq["question"])
                
                if related_faqs:
                    suggestions = "\n\nYou might want to try asking about:\n- " + "\n- ".join(related_faqs[:3])
            
            response = {
                "question": "", 
                "answer": f"I don't have a specific answer for that question.{suggestions}"
            }
            return response, entities, {"confidence": 0}
    
    # Apply contextual enhancements for conversational flow
    enhanced_match = combine_with_context(user_question, best_match, conversation_history, 
                                        intent, entities, context)
    
    # Add metadata about the match
    metadata = {
        "confidence": best_similarity,
        "is_composite": False,
        "intent": {k: v for k, v in intent.items() if v}  # Only include true intents
    }
    
    return enhanced_match, entities, metadata

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question', '')
    
    # Get conversation history from request (sent from frontend) or session
    client_history = request.json.get('history', None)
    
    # Initialize conversation history if it doesn't exist in session
    if 'conversation' not in session:
        session['conversation'] = []
    
    # Get conversation history from session
    conversation_history = session['conversation']
    
    # If client sent history and it's different from server session, reconcile them
    if client_history:
        # Extract user messages from client history
        client_user_messages = [item.get('message') for item in client_history if item.get('role') == 'user']
        
        # If client has messages our session doesn't, rebuild conversation history
        if len(client_user_messages) > len(conversation_history):
            # This can happen if the server session expired but client still has history
            # In this case, we'll use the client history to rebuild our context
            temp_history = []
            for i, msg in enumerate(client_user_messages[-MAX_CONVERSATION_HISTORY:]):
                if i < len(client_history):
                    temp_history.append({
                        'user_question': msg,
                        'topic': 'previous conversation',
                        'entities': []  # We don't have this info from client history
                    })
            
            # Update session with reconstructed history
            session['conversation'] = temp_history
            conversation_history = temp_history
    
    # Process meta-commands
    if user_question.lower() == 'clear history' or user_question.lower() == 'restart':
        # Clear conversation history
        session['conversation'] = []
        return jsonify({
            'question': "",
            'answer': "I've cleared our conversation history. What would you like to know about AI Engineering?"
        })
    
    # Handle special commands or queries
    if user_question.lower() in ['help', 'help me', 'what can you do?', 'what can i ask?', 'what do you know?']:
        response = {
            'question': "What can I ask you?",
            'answer': "You can ask me questions about AI Engineering! For example:\n\n" +
                     "• What is machine learning?\n" +
                     "• How does deep learning work?\n" +
                     "• What's the difference between supervised and unsupervised learning?\n" +
                     "• What skills do I need for AI Engineering?\n" +
                     "• Can you explain neural networks?\n" +
                     "• What are the applications of AI in healthcare?\n\n" +
                     "You can also ask follow-up questions, and I'll remember our conversation context."
        }
        # Add this interaction to history
        conversation_history.append({
            'user_question': user_question,
            'response': response,
            'topic': 'help',
            'entities': ['help']
        })
        session['conversation'] = conversation_history[-MAX_CONVERSATION_HISTORY:]
        return jsonify(response)
    
    # Detect thanks/goodbye patterns
    thanks_patterns = [r'^thank', r'^thanks', r'appreciate it', r'helpful']
    bye_patterns = [r'^bye', r'^goodbye', r'^see you', r'^exit', r'^quit']
    
    if any(re.search(pattern, user_question.lower()) for pattern in thanks_patterns):
        response = {
            'question': "",
            'answer': "You're welcome! Feel free to ask if you have any other questions about AI Engineering."
        }
        # Don't add thanks to conversation history as it's not informative for context
        return jsonify(response)
        
    if any(re.search(pattern, user_question.lower()) for pattern in bye_patterns):
        response = {
            'question': "",
            'answer': "Goodbye! Feel free to come back anytime you have questions about AI Engineering."
        }
        # Don't add goodbyes to conversation history
        return jsonify(response)
    
    # Handle empty or very short queries
    if not user_question or len(user_question.strip()) < 2:
        return jsonify({
            'question': "",
            'answer': "Please type a question about AI Engineering, and I'll do my best to help you!"
        })
    
    # Get the best matching FAQ with conversation history for context
    best_match, entities, metadata = get_best_match(user_question, conversation_history=conversation_history)
    
    # Generate a conversational response based on the matching confidence
    final_answer = best_match['answer']
    
    # Add contextual prefixes based on the type of question and confidence
    if metadata.get("confidence", 0) > 0.8:
        # High confidence match
        if re.search(r'^what|^how|^why|^when|^where|^which|^who', user_question.lower()):
            if len(user_question) < 25:  # Short, direct question
                final_answer = f"Here's what I know: {final_answer}"
    elif 0.6 < metadata.get("confidence", 0) <= 0.8:
        # Medium confidence match
        if not metadata.get("is_composite", False):
            final_answer = f"Based on your question, I think this is relevant: {final_answer}"
    
    # Identify the main topic of this conversation turn
    topic = best_match.get('question', '').split('?')[0] if best_match.get('question') else ''
    if not topic and entities:
        topic = entities[0]  # Use first entity as topic if no question available
    
    # Store this interaction in the conversation history
    conversation_history.append({
        'user_question': user_question,
        'response': {
            'question': best_match.get('question', ''),
            'answer': final_answer
        },
        'entities': entities,
        'topic': topic,
        'confidence': metadata.get("confidence", 0)
    })
    
    # Keep only the most recent MAX_CONVERSATION_HISTORY interactions
    session['conversation'] = conversation_history[-MAX_CONVERSATION_HISTORY:]
    
    # Enhanced response with additional features for complex questions
    if metadata.get("is_composite", False):
        # For composite answers from multiple sources, we already have the formatting
        return jsonify({
            'question': best_match.get('question', ''),
            'answer': final_answer
        })
    
    # Generate suggested follow-up questions based on current topic
    suggested_questions = []
    
    # If we have a good match and entities, generate relevant follow-ups
    if metadata.get("confidence", 0) > 0.5 and entities:
        main_entity = entities[0] if entities else ""
        
        if main_entity:
            if metadata.get("is_composite", False):
                # For composite answers (usually listings or comparisons)
                suggested_questions = [
                    f"Can you explain more about {main_entity}?",
                    f"What are the applications of {main_entity}?",
                    f"What skills do I need to work with {main_entity}?"
                ]
            elif any(intent_name for intent_name, is_active in metadata.get("intent", {}).items() 
                    if is_active and intent_name in ['definition', 'how_to']):
                # For definition or how-to questions
                suggested_questions = [
                    f"What are examples of {main_entity} in real-world applications?",
                    f"How does {main_entity} compare to other approaches?",
                    f"What are the limitations of {main_entity}?"
                ]
    
    # Determine if we should provide context info in the response
    context_info = None
    if conversation_history and len(conversation_history) >= 2:
        if metadata.get("confidence", 0) > 0.7:
            recent_topics = [h.get('topic', '') for h in conversation_history[-2:]]
            if all(recent_topics) and recent_topics[0] == recent_topics[1]:
                context_info = f"Still discussing {recent_topics[1]}"
            elif all(entities) and any(e for e in entities if e in conversation_history[-2].get('entities', [])):
                context_info = "Following up on our previous conversation"
    
    # Get related questions from our FAQs
    related_questions = []
    if entities:
        for entity in entities[:2]:  # Only use top 2 entities
            for faq in faqs[:10]:  # Look at top FAQs
                if (entity.lower() in faq["question"].lower() and 
                    faq["question"] != best_match.get('question', '') and
                    faq["question"] not in related_questions):
                    related_questions.append(faq["question"])
                    if len(related_questions) >= 3:  # Limit to 3 related questions
                        break
    
    return jsonify({
        'question': best_match.get('question', ''),
        'answer': final_answer,
        'suggestions': suggested_questions[:3],  # Limit to 3 suggestions
        'related': related_questions[:3],  # Limit to 3 related questions
        'context_info': context_info
    })

if __name__ == '__main__':
    print("AI Engineering FAQ Chatbot is running!")
    app.run(debug=True)
