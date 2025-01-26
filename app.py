from flask import Flask, request, jsonify, render_template
import your_chatbot_module  # Import your chatbot module here

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    bot_response = your_chatbot_module.get_response(user_message)  # Replace with your response logic
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)

from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize emotion detection model
emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

# Define emotions and states
emotions = ["happy", "sad", "angry", "surprised", "neutral"]
current_emotion = "neutral"

# Function to detect emotion from text
def get_emotion(text):
    result = emotion_classifier(text)
    return result[0]['label']

# Function to update current emotion based on user input
def update_emotion(user_input):
    global current_emotion
    detected_emotion = get_emotion(user_input)
    if detected_emotion in emotions:
        current_emotion = detected_emotion
    else:
        current_emotion = "neutral"

# Function to generate response based on current emotion
def generate_response(user_input):
    update_emotion(user_input)
    if current_emotion == "happy":
        return "I'm feeling great! 😊 How can I assist you today?"
    elif current_emotion == "sad":
        return "I'm here for you. What's on your mind? 😔"
    elif current_emotion == "angry":
        return "Let's find a solution together. What’s bothering you? 😡"
    elif current_emotion == "surprised":
        return "Wow! That's interesting! 😮 Tell me more!"
    else:
        return "How can I help you today?"

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = generate_response(user_message)
    return jsonify({'response': response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81)