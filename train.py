import pandas as pd
import numpy as np
import os
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── 1. Load WELFake dataset ────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("archive/WELFake_Dataset.csv")

print(f"Raw dataset size: {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"Label counts:\n{df['label'].value_counts()}")

# ── 2. Clean and prepare ───────────────────────────────────────────────
df = df.dropna(subset=["title", "text", "label"])
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["label"] = df["label"].astype(int)
df = df[["content", "label"]]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"\nClean dataset size: {len(df)} rows")

# ── 3. Text cleaning function ──────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["content"] = df["content"].apply(clean_text)

# ── 4. Split data ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── 5. TF-IDF Vectorizer ───────────────────────────────────────────────
print("\nFitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ── 6. Train all models ────────────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "xgboost":             XGBClassifier(n_estimators=100, eval_metric="logloss"),
    "lightgbm":            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"  ✓ {name} accuracy: {acc*100:.2f}%")
    joblib.dump(model, f"models/{name}.pkl")
    print(f"  ✓ Saved to models/{name}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\n✓ Vectorizer saved to models/vectorizer.pkl")

# ── 7. Final report ────────────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING COMPLETE — Full Classification Report")
print("="*50)
lr = joblib.load("models/logistic_regression.pkl")
lr_preds = lr.predict(X_test_tfidf)
print(classification_report(y_test, lr_preds, target_names=["Fake", "Real"]))