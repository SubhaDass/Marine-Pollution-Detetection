from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image 
import os

app = Flask(__name__)

import os

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = tf.keras.models.load_model("../model/model.h5")

classes = [
    "No Trash Detected",
    "Trash Detected"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image=file.filename
    )

if __name__ == "__main__":
    app.run(debug=True)