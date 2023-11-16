from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the regression model
# Replace 'model_name' with the actual model name from HuggingFace
model = pipeline("text-classification", model="model_name")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get list of strings from the POST request
        data = request.json
        input_texts = data.get("texts", [])

        # Check if input_texts is a list
        if not isinstance(input_texts, list):
            return jsonify({"error": "Input must be a list of strings."}), 400

        # Predict scores using the model
        results = model(input_texts)

        # Extract scores from results and return
        scores = [result['score'] for result in results]
        return jsonify(scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
