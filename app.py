# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import base64, io, re

app = Flask(__name__)

# ====== 設定 ======
MODEL_PATH = "savedmodel"       # Teachable Machineから出力した .h5（ファイル名は合わせる）
LABELS_PATH = "labels.txt"    # 同梱のラベル一覧
IMG_SIZE = (224, 224)         # 画像分類の標準サイズ
NORMALIZE_0_1 = True          # 0-1正規化（結果が変なら False にして試す）
# ===================

# モデル＆ラベル読込（起動時1回）
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

def decode_base64_image(b64_str: str) -> Image.Image:
    """dataURL でも 素のbase64でもOKにする"""
    # data:image/png;base64,xxxx の前置きを除去
    b64_clean = re.sub("^data:image/.+;base64,", "", b64_str)
    img_bytes = base64.b64decode(b64_clean)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "No 'image' field in JSON."}), 400

        img = decode_base64_image(data["image"]).resize(IMG_SIZE)
        arr = np.asarray(img).astype("float32")
        if NORMALIZE_0_1:
            arr = arr / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

        preds = model.predict(arr)[0].tolist()  # list[float]
        result = [
            {"class": labels[i] if i < len(labels) else f"class_{i}",
             "probability": float(preds[i])}
            for i in range(len(preds))
        ]
        # 確率の高い順にソート
        result.sort(key=lambda x: x["probability"], reverse=True)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ローカル実行用
    app.run(host="0.0.0.0", port=8080, debug=True)
