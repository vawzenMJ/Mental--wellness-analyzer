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
