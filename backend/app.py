from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_emotion

app = Flask(__name__)
# Enable CORS for all routes so React/Vite frontend can communicate seamlessly
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: { "text": "Input sentence here" }
    Response: { "emotion": "joy", "confidence": 0.87 }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field in request JSON body."}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text parameter cannot be empty."}), 400

        result = predict_emotion(text)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    print("Starting Flask Emotion Classifier Backend...")
    app.run(host="0.0.0.0", port=5000, debug=True)
