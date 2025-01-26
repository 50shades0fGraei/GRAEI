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