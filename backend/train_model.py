import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocessing import preprocess_text

def train():
    # 1. Locate dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train.txt")
    if not os.path.exists(train_path):
        train_path = os.path.join(base_dir, "..", "dataset", "train.txt")

    print(f"Loading dataset from: {train_path}")
    df = pd.read_csv(train_path, sep=";", header=None, names=["text", "emotion"])

    # 2. Label Encoding (exactly as in main.ipynb)
    unique_emotions = df["emotion"].unique()
    emotion_numbers = {}
    number_to_emotion = {}
    i = 0
    for emotion in unique_emotions:
        emotion_numbers[emotion] = i
        number_to_emotion[i] = emotion
        i += 1

    df["emotion_num"] = df["emotion"].map(emotion_numbers)
    print(f"Emotion label mapping: {emotion_numbers}")

    # 3. Apply exact same preprocessing to all text
    print("Preprocessing text dataset...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    # 4. Train-test split (test_size=0.33, random_state=42 as in notebook)
    X = df["clean_text"]
    y = df["emotion_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 5. TF-IDF Feature Extraction
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 6. Train LogisticRegression model
    print("Training LogisticRegression model (max_iter=1000)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 7. Evaluate model
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Accuracy: {acc * 100:.2f}%")

    # 8. Save artifacts using joblib into backend/models/
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    mapping_path = os.path.join(models_dir, "label_mapping.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(number_to_emotion, mapping_path)

    print(f"Artifacts successfully saved to {models_dir}:")
    print(f" - {model_path}")
    print(f" - {vectorizer_path}")
    print(f" - {mapping_path}")

if __name__ == "__main__":
    train()
