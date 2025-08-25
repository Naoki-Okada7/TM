# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# モデルとラベルの読み込み
model = tf.keras.models.load_model("keras_model.h5")
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # base64デコード → 画像変換
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # 推論
    prediction = model.predict(image)[0]
    result = [{"class": labels[i], "probability": float(prediction[i])} for i in range(len(labels))]

    return jsonify(result)
