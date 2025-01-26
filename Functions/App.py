import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

emotion_classifier = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

emotions = ["happy", "sad", "angry", "surprised", "neutral"]
current_emotion = "neutral"

def get_emotion(text):
    result = emotion_classifier(text)
    return result[0]['label']

def update_emotion(user_input):
    global current_emotion
    detected_emotion = get_emotion(user_input)
    if detected_emotion in emotions:
        current_emotion = detected_emotion
    else:
        current_emotion = "neutral"

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = generate_response(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 3000)))
