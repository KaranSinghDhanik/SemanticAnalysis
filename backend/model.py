import os
import joblib
import numpy as np
from preprocessing import preprocess_text

class EmotionClassifier:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")

        model_path = os.path.join(models_dir, "model.pkl")
        vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
        mapping_path = os.path.join(models_dir, "label_mapping.pkl")

        if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(mapping_path)):
            raise FileNotFoundError(
                f"Model files not found in {models_dir}. Please run 'python train_model.py' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_mapping = joblib.load(mapping_path)

    def predict(self, raw_text: str):
        if not raw_text or not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Input text cannot be empty.")

        # 1. Apply exact same preprocessing
        cleaned_text = preprocess_text(raw_text)

        # 2. Transform text using loaded TF-IDF vectorizer
        vectorized_text = self.vectorizer.transform([cleaned_text])

        # 3. Predict class label
        pred_num = self.model.predict(vectorized_text)[0]

        # 4. Get confidence / probability if available
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized_text)[0]
            confidence = float(np.max(probs))

        # 5. Map numerical prediction back to string emotion label
        emotion_str = self.label_mapping.get(pred_num, str(pred_num))

        return {
            "emotion": emotion_str,
            "confidence": round(confidence, 4)
        }

# Singleton instance or helper function
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = EmotionClassifier()
    return _classifier

def predict_emotion(text: str):
    classifier = get_classifier()
    return classifier.predict(text)
