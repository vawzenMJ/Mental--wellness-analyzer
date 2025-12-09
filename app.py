from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd # <-- NEW: Needed for model input data structure
import json # <-- NEW: Used for error messages

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS).
CORS(app)

# --- Define the Translation Map (Hybrid Bridge Logic) ---
# Maps the numerical score to the human-readable, keyword-aligned phrase
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
    
    # Get inputs from the data dictionary
    q1_text = get_phrasing(data.get('q1', 0), "I felt little interest or pleasure", "lack of motivation")
    q2_text = get_phrasing(data.get('q2', 0), "I felt down or hopeless", "clinical depression")
    q3_text = get_phrasing(data.get('q3', 0), "I had trouble with sleep", "fatigue and anxiety")
    q4_text = get_phrasing(data.get('q4', 0), "I had thoughts of self-harm", "suicidal thoughts") 
    
    # Combine into a single statement
    synthesized_statement = f"{q1_text} {q2_text} {q3_text} {q4_text}"
    
    return synthesized_statement


# --- Model Loading ---
try:
    # Load the pre-trained TF-IDF vectorizer and the Logistic Regression model.
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')

except FileNotFoundError:
    print("FATAL ERROR: Make sure 'model.joblib' and 'vectorizer.joblib' are in the same directory.")
    vectorizer = None
    model = None


# --- API Endpoint Definition ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives numerical scores (q1, q2, q3, q4), synthesizes a text statement,
    and returns the model's text-based prediction.
    """
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500

    data = request.get_json()

    # --- INPUT VALIDATION (Checking for q1, q2, q3, q4) ---
    required_keys = ['q1', 'q2', 'q3', 'q4']
    if not data or not all(key in data and isinstance(data[key], int) for key in required_keys):
        return jsonify({
            'error': 'Invalid input: The request must include q1, q2, q3, and q4 as integers.'
        }), 400
    
    # --- HYBRID BRIDGE EXECUTION ---
    try:
        # 1. Synthesize the text statement from the numerical scores
        text_input = synthesize_text_from_scores(data)

        # 2. Transform the synthesized text using the original TF-IDF vectorizer
        text_vectorized = vectorizer.transform([text_input])

        # 3. Use the original NLP model to make a prediction
        prediction = model.predict(text_vectorized)

        # 4. Return the prediction
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500


# You must use a Procfile in Render to run this:
# web: gunicorn app:app
# If you run it locally:
if __name__ == '__main__':
    app.run(debug=True)