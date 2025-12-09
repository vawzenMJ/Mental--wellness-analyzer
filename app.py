from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json

# Initialize the Flask application
app = Flask(__name__)

# --- CORS Configuration ---
CORS(app)

# --- Define the Translation Map (Hybrid Bridge Logic) ---
FREQUENCY_MAP = {
    0: "not at all",
    1: "several days",
    2: "more than half the days",
    3: "nearly every day",
}


def synthesize_text_from_scores(data: dict) -> str:
    """Converts the four numerical scores (from the JSON dict) into a single text statement."""

    # Define keywords to improve NLP model alignment
    def get_phrasing(score, base_phrase, risk_keyword):
        if score == 3:
            return f"{base_phrase} nearly every day, strongly indicating {risk_keyword}."
        if score == 2:
            return f"{base_phrase} more than half the days, causing high stress."
        return f"{base_phrase} {FREQUENCY_MAP.get(score, 'unknown time')}."

    # Get inputs from the data dictionary (using .get() to avoid KeyErrors)
    q1_text = get_phrasing(
        data.get('q1', 0), "I felt little interest or pleasure", "lack of motivation")
    q2_text = get_phrasing(
        data.get('q2', 0), "I felt down or hopeless", "clinical depression")
    q3_text = get_phrasing(
        data.get('q3', 0), "I had trouble with sleep", "fatigue and anxiety")
    q4_text = get_phrasing(
        data.get('q4', 0), "I had thoughts of self-harm", "suicidal thoughts")

    # Combine into a single statement
    synthesized_statement = f"{q1_text} {q2_text} {q3_text} {q4_text}"

    return synthesized_statement


# --- Model Loading ---
vectorizer = None
model = None

try:
    # Load the pre-trained TF-IDF vectorizer and the Logistic Regression model.
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    print("Models loaded successfully.")
except Exception as e:
    # Catch any error during loading (FileNotFound, corrupted file, etc.)
    print(f"FATAL ERROR: Model loading failed. Details: {e}")


# --- API Endpoint Definition ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives numerical scores (q1, q2, q3, q4), synthesizes a text statement,
    and returns the model's text-based prediction.
    """
    # Check 1: Model status
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model is not loaded. Check server logs for deployment errors.'}), 500

    # --- FIX: Ensure JSON data is retrieved, even if headers are slightly off ---
    # request.get_json(force=True) attempts to parse the body as JSON regardless of Content-Type.
    data = request.get_json(force=True)

    # Check 2: Data presence and format (400 Bad Request)
    required_keys = ['q1', 'q2', 'q3', 'q4']

    # Check if data is None (meaning no JSON payload received)
    if data is None:
        return jsonify({
            'error': 'Invalid input: No JSON data received. Check Content-Type header.'
        }), 400

    # Check if all keys exist and are integers
    if not all(key in data and isinstance(data.get(key), int) for key in required_keys):
        return jsonify({
            'error': 'Invalid input: The request must include q1, q2, q3, and q4 as integers.'
        }), 400

    # --- HYBRID BRIDGE EXECUTION ---
    try:
        text_input = synthesize_text_from_scores(data)
        text_vectorized = vectorizer.transform([text_input])
        prediction = model.predict(text_vectorized)

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Catch unexpected errors during prediction or synthesis (500 Internal Server Error)
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction. Check server logs.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
