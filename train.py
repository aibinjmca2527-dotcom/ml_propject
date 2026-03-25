"""
train.py - Spam Detection Model Trainer
========================================
This script loads the dataset, preprocesses text using NLP techniques,
trains a Naive Bayes classifier with TF-IDF features, and saves the model.

HOW TO USE:
    1. Place your Kaggle spam dataset CSV in the project folder
       (Expected columns: 'v1' = label [spam/ham], 'v2' = message text)
       OR columns named 'label' and 'text' / 'message'
    2. Run: python train.py
    3. Model files will be saved in the 'model/' folder
"""

import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH    = "spam.csv"          # Change if your file is named differently
MODEL_DIR       = "model"
MODEL_PATH      = os.path.join(MODEL_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING  (NLP Step)
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Clean and normalize raw email / SMS text.
    Steps:
      - Lowercase all characters
      - Remove URLs
      - Remove email addresses
      - Remove punctuation and special characters
      - Remove extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)            # remove emails
    text = re.sub(r"[^a-z0-9\s]", " ", text)        # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()         # remove extra spaces
    return text


# ─────────────────────────────────────────────
# 3. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the Kaggle spam dataset.
    Supports: 'v1'/'v2' OR 'label'/'text' OR 'label'/'message'
    """
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path, encoding="latin-1")

    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    elif "label" in df.columns and "text" in df.columns:
        df = df[["label", "text"]]
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]].rename(columns={"message": "text"})
    else:
        raise ValueError(
            f"Cannot detect columns. Expected 'v1'+'v2' or 'label'+'text/message'. "
            f"Found: {list(df.columns)}"
        )

    df.dropna(subset=["label", "text"], inplace=True)
    df["label"] = df["label"].str.strip().str.lower()
    print(f"[INFO] Loaded {len(df)} rows")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")
    return df


# ─────────────────────────────────────────────
# 4. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def train():
    df = load_dataset(DATASET_PATH)

    print("[INFO] Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)
    df["label_encoded"] = df["label"].map({"spam": 1, "ham": 0})

    X = df["clean_text"].values
    y = df["label_encoded"].values

    # 80/20 split, stratified so both classes appear in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # TF-IDF: converts text into a matrix of term importance scores
    # max_features=5000  → top 5000 words/phrases
    # ngram_range=(1,2)  → unigrams + bigrams for context
    # sublinear_tf=True  → log-scale term frequencies
    print("[INFO] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Multinomial Naive Bayes: fast, works great with word-frequency data
    # alpha=0.1 → Laplace smoothing to handle unseen words
    print("[INFO] Training Naive Bayes classifier...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred   = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "="*52)
    print("          MODEL EVALUATION RESULTS")
    print("="*52)
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Ham  (TN): {cm[0][0]}   |  False Spam (FP): {cm[0][1]}")
    print(f"  False Ham (FN): {cm[1][0]}   |  True Spam  (TP): {cm[1][1]}")
    print("="*52 + "\n")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"[OK] Model saved     -> {MODEL_PATH}")
    print(f"[OK] Vectorizer saved-> {VECTORIZER_PATH}")
    print("\n>>> Training complete. Run: python app.py")


if __name__ == "__main__":
    train()
