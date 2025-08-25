import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# ===== モデル読み込み =====
MODEL_PATH = "savedmodel"
loaded = tf.saved_model.load(MODEL_PATH)
infer = loaded.signatures["serving_default"]

IMG_SIZE = (224, 224)

# ===== ラベル読み込み =====
LABELS_PATH = "labels.txt"   # Teachable Machine が出力する labels.txt
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "no image"}), 400

    # base64 デコード → 画像処理
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data)).convert("RGB").resize(IMG_SIZE)

    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    tensor = tf.convert_to_tensor(arr)

    # 推論
    outputs = infer(tensor)
    key = list(outputs.keys())[0]        # "sequential_3" など
    probs = outputs[key].numpy()[0]      # shape (num_classes,)

    # ラベルと結合
    results = [
        {"class": labels[i], "probability": float(probs[i])}
        for i in range(len(labels))
    ]

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
