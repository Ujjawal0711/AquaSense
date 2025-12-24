## Architecture!
This application runs entirely on a local machine. It uses a Flask web server to handle requests from the browser, process images, and serve the frontend. The AI inference is performed locally by loading the model files directly into the backend.

(Insert the architecture diagram you created here, but ensure it only shows a local setup)

## How It Works
The AI recognition process works in two stages:

Signature Creation: An uploaded image is first processed and fed into a core neural network model (model.ckpt). This model acts as a feature extractor, creating a unique mathematical "signature" (an embedding) for the fish in the image.

Signature Matching: This new signature is then compared against a pre-computed library of thousands of known fish signatures (database.pt) using a high-speed search library (Faiss). The closest match in the library is identified as the predicted species.

## Tech Stack
Backend: Flask

AI Framework: PyTorch

AI Libraries: scikit-learn, Faiss, Timm, Torch-EMA

Frontend: HTML, CSS (Tailwind), JavaScript

## Local Setup & Installation
Follow these steps to run the application on your local machine.

Clone the Repository

Bash

git clone https://github.com/Ujjawal0711/AquaSense.git
cd AquaSense
Download the Model Files
You must download the model files from the project's GitHub Releases page.

[suspicious link removed]

Unzip the file and place model.ckpt and database.pt in the root project folder.

Create and Activate a Virtual Environment

Bash

python -m venv .venv
# On Windows:
# .venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate
Install Dependencies

Bash

pip install -r requirements.txt
Run the Application

Bash

python app.py
The server will be running at http://127.0.0.1:5000.

## Acknowledgments
This project uses pre-trained models and code adapted from the excellent Fishial.ai repository.
