import io
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np

# Import the main class from your inference.py script
from inference import EmbeddingClassifier

app = Flask(__name__)

# --- CONFIGURATION & MODEL LOADING ---

# This dictionary points to your local model files
config = {
    "model": {
        "path": "model.ckpt",
        "device": "cpu"
    },
    "dataset": {
        "path": "database.pt"
    },
    "log_level": "INFO"
}

print("Loading local AI model... This may take a moment.")
# This creates a single instance of the classifier, loading the model into memory
classifier = EmbeddingClassifier(config)
print("Local AI Model loaded successfully.")


# --- WEB ROUTES ---
# Add this new route to your app.py
@app.route('/library')
def library():
    return render_template('library.html')

@app.route('/ai')
def ai_page():
    return render_template('AI.html')

@app.route('/')
def index():

    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyse():
    req_data = request.get_json()
    base64_data = req_data.get('data')

    if not base64_data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode the base64 string into image bytes
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert the image to a NumPy array for the classifier
        image_np = np.array(image)

        # Call the classifier with the image to get prediction results
        prediction_results = classifier(image_np)

        # Get the top prediction from the results list
        top_prediction = prediction_results[0]
        predicted_name = top_prediction.name

        # Create the HTML response string your frontend expects
        response_html = (
            f"<h3>Analysis Complete</h3>"
            f"<p><strong>Predicted Species:</strong> {predicted_name}</p>"
            f"<p><strong>Confidence Score:</strong> {top_prediction.accuracy:.2f}</p>"
        )

        # Build the final JSON response
        response_data = {
            "candidates": [{"content": {"parts": [{"text": response_html}]}}]
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {e}")  # This will show the error in your terminal
        return jsonify({'error': 'Failed to process image.', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')