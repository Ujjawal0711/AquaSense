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
try:
    classifier = EmbeddingClassifier(config)
    print("Local AI Model loaded successfully.")
except FileNotFoundError as e:
    raise SystemExit(
        f"\n[AquaSense] Could not load model files: {e}\n"
        "This app needs the trained weights `model.ckpt` and the embedding "
        "database `database.pt` in the project root. They are NOT distributed "
        "with the repository (see README for how to obtain/train them).\n"
    )


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
    req_data = request.get_json(silent=True) or {}
    base64_data = req_data.get('data')

    if not base64_data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode the base64 string into image bytes
        image_bytes = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert the image to a NumPy array for the classifier
        image_np = np.array(image)

        # The classifier returns the candidate matches for the single fish
        candidates = classifier(image_np)
        if not candidates:
            return jsonify({"html":
                "<h3>No confident match</h3>"
                "<p>This image did not match any known species above the similarity "
                "threshold. Try a clearer otolith image.</p>"})

        # Rank by confidence (highest first) rather than dict insertion order,
        # and surface the top-5 matches instead of only the single best.
        ranked = sorted(candidates, key=lambda r: r.accuracy, reverse=True)
        top = ranked[0]
        rows = "".join(f"<li>{r.name} &mdash; {r.accuracy:.2f}</li>" for r in ranked[:5])

        response_html = (
            f"<h3>Predicted Species: {top.name}</h3>"
            f"<p><strong>Confidence:</strong> {top.accuracy:.2f}</p>"
            f"<p><strong>Top matches:</strong></p><ol>{rows}</ol>"
        )
        return jsonify({"html": response_html})

    except Exception as e:
        print(f"An error occurred: {e}")  # This will show the error in your terminal
        return jsonify({'error': 'Failed to process image.', 'details': str(e)}), 500


if __name__ == '__main__':
    # debug=False: never ship the interactive debugger / auto-reloader on a
    # publicly reachable host (0.0.0.0). Flip to True only for local dev.
    app.run(debug=False, host='0.0.0.0', port=5000)