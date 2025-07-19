// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

// Conversation history management
let conversationHistory = [];
const MAX_HISTORY_LENGTH = 10;

// Function to format text with markdown-like syntax
function formatText(text) {
    if (!text) return '';
    
    // Convert bullet points
    text = text.replace(/•\s+/g, '• ');
    
    // Handle line breaks
    const paragraphs = text.split('\n\n');
    const formattedParagraphs = paragraphs.map(p => {
        // If paragraph is a bulleted list
        if (p.includes('• ')) {
            const items = p.split('• ').filter(item => item.trim().length > 0);
            if (items.length > 0) {
                return `<ul>${items.map(item => `<li>${item.trim()}</li>`).join('')}</ul>`;
            }
            return `<p>${p}</p>`;
        }
        
        // Handle numbered lists (1. 2. 3. etc)
        if (p.match(/^\d+\.\s/)) {
            const items = p.split(/\d+\.\s/).filter(item => item.trim().length > 0);
            if (items.length > 0) {
                return `<ol>${items.map(item => `<li>${item.trim()}</li>`).join('')}</ol>`;
            }
            return `<p>${p}</p>`;
        }
        
        return p ? `<p>${p}</p>` : '';
    });
    
    return formattedParagraphs.join('');
}

// Function to add a message to the chat
function addMessage(message, isUser = false, faqQuestion = '', context = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    // Add timestamp
    const timestamp = new Date();
    const timeStr = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // If it's a bot message, use formatting
    if (!isUser) {
        messageContent.innerHTML = formatText(message);
        
        // If there's a related FAQ question, add it
        if (faqQuestion && faqQuestion.length > 0) {
            const faqRef = document.createElement('div');
            faqRef.classList.add('faq-reference');
            faqRef.textContent = `Related to: "${faqQuestion}"`;
            messageContent.appendChild(faqRef);
        }
        
        // If context is provided, add it as additional info
        if (context) {
            const contextInfo = document.createElement('div');
            contextInfo.classList.add('context-info');
            contextInfo.textContent = context;
            messageContent.appendChild(contextInfo);
        }
        
        // Add to conversation history
        conversationHistory.push({
            role: 'bot',
            message: message,
            timestamp: timestamp,
            faqQuestion: faqQuestion
        });
    } else {
        // For user messages, just use the text directly
        messageContent.textContent = message;
        
        // Add to conversation history
        conversationHistory.push({
            role: 'user',
            message: message,
            timestamp: timestamp
        });
    }
    
    // Trim conversation history if needed
    if (conversationHistory.length > MAX_HISTORY_LENGTH * 2) {
        conversationHistory = conversationHistory.slice(-MAX_HISTORY_LENGTH * 2);
    }
    
    // Add timestamp display
    const timeElement = document.createElement('span');
    timeElement.classList.add('message-time');
    timeElement.textContent = timeStr;
    messageContent.appendChild(timeElement);
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to the bottom of the chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to add a system message (like hints, tips, etc.)
function addSystemMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'system-message');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    messageContent.innerHTML = message;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to the bottom of the chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to add typing indicator
function showTypingIndicator() {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.classList.add('message', 'bot-message', 'typing-indicator');
    indicatorDiv.id = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('typing-dot');
        indicatorDiv.appendChild(dot);
    }
    
    chatMessages.appendChild(indicatorDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Function to remove typing indicator
function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Function to handle suggestion chips
function addSuggestionChips(suggestions = null) {
    // Default suggestions if none provided
    const defaultSuggestions = [
        { text: 'Help', action: () => { userInput.value = 'help'; sendMessage(); }},
        { text: 'What is AI?', action: () => { userInput.value = 'What is Artificial Intelligence?'; sendMessage(); }},
        { text: 'ML vs DL', action: () => { userInput.value = 'What is the difference between Machine Learning and Deep Learning?'; sendMessage(); }},
        { text: 'AI careers', action: () => { userInput.value = 'What careers are available in AI Engineering?'; sendMessage(); }}
    ];
    
    const suggestionsList = suggestions || defaultSuggestions;
    
    const suggestionDiv = document.createElement('div');
    suggestionDiv.classList.add('suggestion-chips');
    
    suggestionsList.forEach(suggestion => {
        const chip = document.createElement('button');
        chip.classList.add('suggestion-chip');
        chip.textContent = suggestion.text;
        
        if (typeof suggestion.action === 'function') {
            chip.addEventListener('click', suggestion.action);
        } else {
            chip.addEventListener('click', () => {
                userInput.value = suggestion.text;
                sendMessage();
            });
        }
        
        suggestionDiv.appendChild(chip);
    });
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'bot-message');
    messageDiv.appendChild(suggestionDiv);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Manage conversation state
let isFirstInteraction = true;

// Function to handle sending a message
async function sendMessage() {
    const message = userInput.value.trim();
    
    if (message === '') return;
    
    // Add user message to chat
    addMessage(message, true);
    
    // Clear input field
    userInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    // If this is the first interaction, set flag
    if (isFirstInteraction) {
        isFirstInteraction = false;
    }
    
    try {
        // Calculate a realistic typing time based on answer length
        const typingTimeBase = 500; // minimum delay in ms
        
        // Send request to server with conversation history
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                question: message,
                history: conversationHistory.slice(-MAX_HISTORY_LENGTH) // Send recent history
            }),
        });
        
        const data = await response.json();
        
        // Calculate a realistic typing delay (longer answers should take longer)
        // But cap it to feel responsive
        const answerLength = data.answer ? data.answer.length : 0;
        const typingTime = Math.min(typingTimeBase + answerLength * 2, 2000);
        
        // Delay to simulate typing
        setTimeout(() => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response to chat
            addMessage(data.answer, false, data.question, data.context_info);
            
            // If there are related questions or suggested next questions, add them as chips
            if ((data.related && data.related.length > 0) || (data.suggestions && data.suggestions.length > 0)) {
                setTimeout(() => {
                    // First, check for suggestions (these are proactive)
                    if (data.suggestions && data.suggestions.length > 0) {
                        const suggestionChips = data.suggestions.map(suggestion => ({
                            text: suggestion,
                            action: () => {
                                userInput.value = suggestion;
                                sendMessage();
                            }
                        }));
                        
                        addSuggestionChips(suggestionChips);
                    }
                    // Then check for related questions (from FAQs)
                    else if (data.related && data.related.length > 0) {
                        const relatedChips = data.related.map(q => ({
                            text: q,
                            action: () => {
                                userInput.value = q;
                                sendMessage();
                            }
                        }));
                        
                        addSuggestionChips(relatedChips);
                    }
                }, 500);
            }
        }, typingTime);
        
    } catch (error) {
        console.error('Error:', error);
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add error message to chat
        addMessage('Sorry, I encountered an error while processing your request. Please try again later.');
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

userInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Add command to clear chat history
function clearChat() {
    // Clear UI
    while (chatMessages.firstChild) {
        chatMessages.removeChild(chatMessages.firstChild);
    }
    
    // Clear conversation history
    conversationHistory = [];
    
    // Add initial bot message
    addMessage('Hello! I\'m your AI Engineering FAQ assistant. How can I help you today?', false);
    
    // Add suggestion chips
    addSuggestionChips();
    
    // Reset first interaction flag
    isFirstInteraction = true;
    
    // Clear server-side conversation
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: 'clear history' }),
    });
}

// Function to generate smart follow-up questions
function generateFollowUps(conversation) {
    // Simple heuristic for now - could be enhanced with actual intelligence from backend
    if (conversation.length < 2) return null;
    
    const lastBotMessage = conversation.filter(msg => msg.role === 'bot').pop();
    const lastUserMessage = conversation.filter(msg => msg.role === 'user').pop();
    
    if (!lastBotMessage || !lastUserMessage) return null;
    
    // This could be enhanced with actual intelligence or suggestions from the backend
    const keywords = ['AI', 'machine learning', 'neural networks', 'deep learning', 'careers', 'course'];
    
    for (const keyword of keywords) {
        if (lastUserMessage.message.toLowerCase().includes(keyword.toLowerCase())) {
            switch (keyword.toLowerCase()) {
                case 'ai':
                    return [
                        'Tell me about AI applications',
                        'What is the future of AI?',
                        'How is AI used in industry?'
                    ];
                case 'machine learning':
                    return [
                        'What ML algorithms are most common?',
                        'How is ML different from statistics?',
                        'What skills do I need for ML?'
                    ];
                case 'neural networks':
                case 'deep learning':
                    return [
                        'How do neural networks work?',
                        'What is the difference between CNN and RNN?',
                        'When should I use deep learning?'
                    ];
                case 'careers':
                case 'course':
                    return [
                        'What skills are needed for AI Engineering?',
                        'How to prepare for AI interviews?',
                        'Best AI projects for portfolio?'
                    ];
            }
        }
    }
    
    return null;
}

// Initialize the chat
window.addEventListener('load', () => {
    // Focus input field
    userInput.focus();
    
    // Add initial bot message
    addMessage('Hello! I\'m your AI Engineering FAQ assistant. How can I help you today?', false);
    
    // Add suggestion chips
    addSuggestionChips();
    
    // Add clear button to the UI
    const clearButton = document.createElement('button');
    clearButton.id = 'clear-button';
    clearButton.textContent = 'Clear Chat';
    clearButton.addEventListener('click', clearChat);
    
    document.querySelector('.chat-header').appendChild(clearButton);
    
    // Add event listener for scrolling - could be used for loading more messages
    chatMessages.addEventListener('scroll', function() {
        // Future implementation: if we scroll near top, load older messages
        // if (this.scrollTop < 50) { loadOlderMessages(); }
    });
    
    // Make user input more interactive
    userInput.addEventListener('input', function() {
        const inputText = this.value.trim();
        if (inputText.length > 0) {
            sendButton.classList.add('active');
        } else {
            sendButton.classList.remove('active');
        }
    });
});
