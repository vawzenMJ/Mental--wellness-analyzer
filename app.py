from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS).
# This is crucial to allow your Android app (or any web app)
# to communicate with this server.
CORS(app)

# --- Model Loading ---
# We use a try-except block to handle errors gracefully if the model files are missing.
try:
    # Load the pre-trained TF-IDF vectorizer from the file 'vectorizer.joblib'
    # This is used to convert text into a numerical format the model understands.
    vectorizer = joblib.load('vectorizer.joblib')

    # Load the pre-trained Logistic Regression model from the file 'model.joblib'
    model = joblib.load('model.joblib')

except FileNotFoundError:
    print("Error: Make sure 'model.joblib' and 'vectorizer.joblib' are in the same directory as app.py")
    vectorizer = None
    model = None

# --- API Endpoint Definition ---
# This defines the URL endpoint for making predictions.
# Your app will send requests to 'https://your-api-url.onrender.com/predict'


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives text input from the frontend, transforms it using the TF-IDF vectorizer,
    and returns the model's prediction as a JSON response.
    """
    # First, check if the model and vectorizer were loaded successfully.
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500

    # Get the JSON data sent from the Android app
    data = request.get_json()

    # Validate the input to make sure it contains a 'text' field.
    if not data or 'text' not in data:
        return jsonify({'error': 'Invalid input: The request must include a "text" field.'}), 400

    text_input = data['text']

    # The vectorizer expects a list of documents (even if it's just one).
    # We transform the input text into its TF-IDF vector representation.
    text_vectorized = vectorizer.transform([text_input])

    # Use the loaded model to make a prediction on the vectorized text.
    prediction = model.predict(text_vectorized)

    # The predict method returns an array (e.g., ['Stable']), so we select the first element.
    # We then return this prediction in a standard JSON format.
    return jsonify({'prediction': prediction[0]})
<<<<<<< HEAD
=======


# This block allows you to run the server directly from your command line
# for testing purposes using 'python app.py'
#if __name__ == '__main__':
    # It's recommended to run on port 5000 for local testing.
    # Render will handle the port automatically in production.
    #app.run(debug=True, port=5000)
    #print(1)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello from Flask on Render!"

@app.route('/', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    # Dummy prediction for testing
    if "sad" in text.lower():
        prediction = "High Risk"
    elif "happy" in text.lower():
        prediction = "Low Risk"
    else:
        prediction = "Stable"

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
>>>>>>> bd894ff8307834fce207579ff4166d8a8fc9be7f
