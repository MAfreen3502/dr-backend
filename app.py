from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from datetime import datetime
from PIL import Image

# -------------------------------
# App setup
# -------------------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "dr_results.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------------------
# Database model
# -------------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(200))
    predicted_class = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "dr_efficientnet_model.h5")

model = keras.models.load_model(MODEL_PATH)

# ⚠️ Must be same order as training folder
CLASS_NAMES = [
    "Healthy",
    "Mild DR",
    "Moderate DR",
    "Proliferate DR",
    "Severe DR"
]

IMG_SIZE = 224


# -------------------------------
# Image preprocessing
# -------------------------------
def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


# -------------------------------
# API route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    upload_folder = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    save_path = os.path.join(upload_folder, file.filename)
    file.save(save_path)

    img = prepare_image(save_path)

    preds = model.predict(img)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])

    predicted_class = CLASS_NAMES[idx]

    # -------------------------------
    # Save to DB
    # -------------------------------
    row = Prediction(
        image_name=file.filename,
        predicted_class=predicted_class,
        confidence=confidence
    )

    db.session.add(row)
    db.session.commit()

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence
    })


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)