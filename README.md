# Emotion Classifier - Text Emotion Prediction Web App

A minimal, full-stack Machine Learning web application for **Text Emotion Classification** built with **Python, Flask, scikit-learn, NLTK, and React (Vite)**. 

The application utilizes a **Logistic Regression + TF-IDF Vectorizer** model trained on labelled emotion text data (`train.txt`). It performs real-time text preprocessing and sentiment inference, returning predicted emotions (e.g., *joy, sadness, anger, fear, love, surprise*) alongside model prediction confidence.

---

## 🚀 Technologies Used

- **Machine Learning & NLP**: Python 3.11+, scikit-learn, NLTK, pandas, joblib, numpy
- **Backend API**: Flask, Flask-CORS
- **Frontend UI**: React, Vite, Vanilla CSS (Glassmorphism minimalist UI)

---

## 📁 Project Architecture & Structure

```
SemanticAnalysis/
├── main.ipynb             # Original Jupyter notebook with ML experiments
├── dataset/
│   ├── train.txt          # Original training dataset (text;emotion)
│   ├── test.txt
│   └── val.txt
├── backend/
│   ├── app.py             # Flask API server (exposes POST /predict)
│   ├── model.py           # Model loading & inference module
│   ├── preprocessing.py   # Reusable text preprocessing pipeline
│   ├── train_model.py     # Script to train & persist TF-IDF + LogisticRegression model
│   ├── train.txt          # Training data copy
│   └── models/            # Saved scikit-learn artifacts (.pkl)
│       ├── model.pkl
│       ├── vectorizer.pkl
│       └── label_mapping.pkl
├── frontend/              # React + Vite minimal UI
│   ├── src/
│   │   ├── App.jsx        # Emotion Classifier React component
│   │   ├── index.css      # Custom styling
│   │   └── main.jsx       # Entry point
│   ├── package.json
│   └── vite.config.js
├── requirements.txt       # Backend Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Step-by-Step Setup & Running

### 1. Install Python Dependencies

In the root directory, install all required backend Python packages:

```bash
pip install -r requirements.txt
```

### 2. Dataset Placement

Ensure `train.txt` is located inside `backend/train.txt` or `dataset/train.txt`. The format should be:

```text
i didnt feel humiliated;sadness
im grabbing a minute to post i feel greedy wrong;anger
i am ever feeling nostalgic about the fireplace;love
```

### 3. Train the Model

Run `train_model.py` to process the dataset, train the TF-IDF + Logistic Regression model, evaluate accuracy, and save the binary model artifacts to `backend/models/`:

```bash
python backend/train_model.py
```

*Output:*
- `backend/models/model.pkl`
- `backend/models/vectorizer.pkl`
- `backend/models/label_mapping.pkl`

### 4. Start the Flask Backend API

Launch the Flask server:

```bash
python backend/app.py
```

The server will start running locally at: `http://127.0.0.1:5000`

### 5. Start the React Frontend

In a separate terminal, navigate to the `frontend` directory, install packages, and launch the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite local URL (typically `http://localhost:5173`) in your browser.

---

## 📡 API Reference

### **POST** `/predict`

Performs emotion classification on input text.

#### Request Headers
`Content-Type: application/json`

#### Example Request Body
```json
{
  "text": "I am really happy and excited today!"
}
```

#### Example Response (200 OK)
```json
{
  "emotion": "joy",
  "confidence": 0.87
}
```

---

## 🧠 ML Pipeline Summary

1. **Preprocessing (`preprocessing.py`)**:
   - Convert text to lowercase
   - Remove punctuation (`string.punctuation`)
   - Remove digits/numbers
   - Remove Unicode emojis using regex
   - Remove English stopwords using NLTK `word_tokenize`
2. **Vectorization**: Transform cleaned text into numerical vectors using `TfidfVectorizer`.
3. **Classification**: Predict emotion label using `LogisticRegression(max_iter=1000)` and extract class probability using `predict_proba`.
4. **Decoding**: Map integer prediction back to string emotion label using saved `label_mapping.pkl`.
