# 🐟 AquaSense — AI Fish Species Identification

AquaSense identifies fish species from an uploaded image using a deep metric-learning
model and fast nearest-neighbour search. It runs entirely on a local machine: a Flask
server handles the web UI and image uploads, and inference is performed in-process by a
PyTorch model — no external AI API is called.

## How It Works

Recognition happens in two stages:

1. **Signature creation** — the uploaded image is passed through a ConvNeXt-Tiny
   backbone with an attention-pooling head and an ArcFace embedding layer
   (`StableEmbeddingModel`), producing a 256-dimensional L2-normalised embedding
   (a "signature") for the fish.
2. **Signature matching** — that embedding is compared against a precomputed library
   of known fish signatures (`database.pt`). Candidate classes are first narrowed by
   cosine similarity to per-class **centroids** (top-5), then refined with a
   **Faiss** inner-product search over the members of those classes (top-10). The
   best-scoring species is returned, along with the next-best matches.

## Tech Stack

- **Backend:** Flask
- **AI framework:** PyTorch / torchvision (ConvNeXt-Tiny)
- **AI libraries:** Faiss (nearest-neighbour search), scikit-learn, torch-ema
- **Frontend:** HTML, Tailwind CSS, vanilla JavaScript

## Model Files (important)

The trained weights **`model.ckpt`** and the embedding database **`database.pt`** are
**not distributed with this repository** — they are large and are gitignored
(`*.ckpt`, `*.pt`). The app expects both files in the project root and will exit with a
clear message if they are missing.

To run inference you need to supply them yourself, either by:
- training the model on a fish dataset (this repo contains the **inference** pipeline;
  the embedding architecture is defined in `inference.py` / `model_v2.py`), and then
  precomputing `database.pt` from your labelled image set, or
- obtaining a compatible checkpoint whose `state_dict` matches `StableEmbeddingModel`
  (`embedding_dim=256`, `num_classes=639`).

Without these files the web server still starts and serves the UI, but `/analyse` cannot
classify images.

## Local Setup

```bash
git clone https://github.com/Ujjawal0711/AquaSense.git
cd AquaSense

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

# place model.ckpt and database.pt in this folder (see "Model Files" above)

python app.py
```

The server runs at `http://127.0.0.1:5000`.

## Project Layout

| File | Purpose |
|---|---|
| `app.py` | Flask server, routes, and the `/analyse` inference endpoint |
| `inference.py` | `EmbeddingClassifier` + the `StableEmbeddingModel` architecture |
| `model_v2.py` | Model / training-side definitions |
| `database.py`, `models.py` | SQLAlchemy data layer for CMLRE otolith records |
| `templates/` | Web UI (`index.html`, `AI.html`, `Library.html`) |
| `labels.json` | Species id → label mapping |

## Acknowledgments

Model architecture and approach are adapted from the open-source
[Fishial.ai](https://github.com/fishial/fish-identification) fish-identification project.
