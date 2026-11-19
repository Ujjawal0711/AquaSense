import base64
import binascii
import io
import logging
import os
from html import escape

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] - %(message)s")
log = logging.getLogger("AquaSense")

app = Flask(__name__)
# The UI allows 10 MB images; base64 inflates that by ~4/3, plus JSON overhead.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config = {
    "model": {
        "path": os.getenv("AQUASENSE_MODEL", os.path.join(BASE_DIR, "model.ckpt")),
        "device": os.getenv("AQUASENSE_DEVICE", "cpu"),
    },
    "dataset": {
        "path": os.getenv("AQUASENSE_DATABASE", os.path.join(BASE_DIR, "database.pt")),
    },
    "log_level": "INFO",
}

TOP_K = 5


def _load_classifier():
    """Load the model once at startup. If the weights or ML dependencies are
    missing the UI still serves and /analyse answers 503 instead of crashing."""
    missing = [p for p in (config["model"]["path"], config["dataset"]["path"]) if not os.path.isfile(p)]
    if missing:
        return None, (
            "Model files not found: " + ", ".join(os.path.basename(p) for p in missing)
            + ". Place model.ckpt and database.pt in the project root (see README)."
        )
    try:
        from inference import EmbeddingClassifier  # heavy import (torch); only when needed
        log.info("Loading local AI model... This may take a moment.")
        clf = EmbeddingClassifier(config)
        log.info("Local AI model loaded successfully.")
        return clf, None
    except Exception as e:  # bad checkpoint, missing torch, etc.
        log.exception("Failed to load the AI model")
        return None, f"Failed to load the AI model: {e}"


classifier, classifier_error = _load_classifier()
if classifier_error:
    log.warning("[AquaSense] %s The web UI will run, but /analyse is disabled.", classifier_error)


# --- WEB ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ai')
def ai_page():
    return render_template('AI.html')


@app.route('/library')
def library():
    # File is Library.html; lowercase name 404s on case-sensitive filesystems.
    return render_template('Library.html')


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": classifier is not None, "error": classifier_error})


def _decode_image(b64: str) -> Image.Image:
    # Accept both raw base64 and full data URLs ("data:image/png;base64,....").
    if b64.startswith("data:") and "," in b64:
        b64 = b64.split(",", 1)[1]
    image_bytes = base64.b64decode(b64, validate=True)
    image = Image.open(io.BytesIO(image_bytes))
    # Phone photos store rotation in EXIF; without this the model sees them sideways.
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


@app.route('/analyse', methods=['POST'])
def analyse():
    if classifier is None:
        return jsonify({'error': 'AI model is not available.', 'details': classifier_error}), 503

    req_data = request.get_json(silent=True) or {}
    base64_data = req_data.get('data')
    if not base64_data or not isinstance(base64_data, str):
        return jsonify({'error': 'No image data provided'}), 400

    try:
        image = _decode_image(base64_data)
    except (binascii.Error, ValueError, OSError, Image.DecompressionBombError) as e:  # OSError covers truncated/unreadable files
        return jsonify({'error': 'Invalid image.', 'details': str(e)}), 400

    try:
        candidates = classifier(image)
    except Exception:
        log.exception("Inference failed")
        return jsonify({'error': 'Failed to process image.'}), 500

    if not candidates:
        return jsonify({"html":
            "<h3>No confident match</h3>"
            "<p>This image did not match any known species above the similarity "
            "threshold. Try a clearer, well-lit photo of a single fish.</p>"})

    ranked = sorted(candidates, key=lambda r: r.accuracy, reverse=True)[:TOP_K]
    top = ranked[0]
    # Species names come from the model database; escape before embedding in HTML.
    rows = "".join(f"<li>{escape(r.name)} &mdash; {r.accuracy:.2f}</li>" for r in ranked)
    response_html = (
        f"<h3>Predicted Species: {escape(top.name)}</h3>"
        f"<p><strong>Confidence:</strong> {top.accuracy:.2f}</p>"
        f"<p><strong>Top matches:</strong></p><ol>{rows}</ol>"
    )
    return jsonify({
        "html": response_html,
        "predictions": [{"name": r.name, "species_id": int(r.species_id), "confidence": round(r.accuracy, 4)}
                        for r in ranked],
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({'error': 'Image too large. Maximum upload is 10 MB.'}), 413


if __name__ == '__main__':
    # Default to localhost; set AQUASENSE_HOST=0.0.0.0 to expose on the network.
    # Never enable debug on a reachable host (the Werkzeug debugger allows code execution).
    app.run(debug=False, host=os.getenv("AQUASENSE_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
