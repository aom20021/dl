import os

import torch
import torchaudio
from flask import Flask, request, render_template
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lista para almacenar predicciones anteriores
previous_predictions = []

# Load model and feature extractor
genres = ["rock", "reggae", "pop", "metal", "jazz", "hiphop", "disco", "country", "classical", "blues"]
model_path = "./model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)


def preprocess_audio_for_prediction(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    if waveform.size(0) > 1:  # Convert stereo to mono
        waveform = waveform.mean(dim=0)
    return waveform.squeeze().numpy()


def predict_genre(file_path, model, feature_extractor):
    audio_data = preprocess_audio_for_prediction(file_path)
    inputs = feature_extractor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000 * 3,
    )
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return genres[predicted_label]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    global previous_predictions
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part in the request.")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            try:
                predicted_genre = predict_genre(file_path, model, feature_extractor)
                # Guardar la predicción en la lista
                previous_predictions.append({"filename": filename, "genre": predicted_genre})
                return render_template("index.html", genre=predicted_genre)
            except Exception as e:
                return render_template("index.html", error=f"Error processing file: {str(e)}")
    return render_template("index.html")


@app.route("/predictions", methods=["GET"])
def predictions():
    return render_template("predictions.html", predictions=previous_predictions)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
