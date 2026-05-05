from flask import Flask, render_template, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

# ── Load models once at startup ────────────────────────────────────────
print("Loading models...")
vectorizer = joblib.load("models/vectorizer.pkl")
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Random Forest":       joblib.load("models/random_forest.pkl"),
    "XGBoost":             joblib.load("models/xgboost.pkl"),
    "LightGBM":            joblib.load("models/lightgbm.pkl"),
}
print("All models loaded successfully.")

# ── Text cleaning (must match train.py) ───────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text.split()) < 5:
        return jsonify({"error": "Please enter at least 5 words."}), 400

    cleaned = clean_text(text)
    tfidf_vec = vectorizer.transform([cleaned])

    results = []
    votes = []

    for model_name, model in models.items():
        pred = int(model.predict(tfidf_vec)[0])
        proba = model.predict_proba(tfidf_vec)[0]
        confidence = round(float(max(proba)) * 100, 1)
        verdict = "REAL" if pred == 0 else "FAKE"
        votes.append(pred)
        results.append({
            "model": model_name,
            "verdict": verdict,
            "confidence": confidence
        })

    # Ensemble: majority vote
    fake_votes = sum(votes)
    real_votes = len(votes) - fake_votes
    ensemble_confidence = round((max(real_votes, fake_votes) / len(votes)) * 100, 1)

    if fake_votes > real_votes:
        final_verdict = "FAKE"
    elif real_votes > fake_votes:
        final_verdict = "REAL"
    else:
        final_verdict = "UNCERTAIN"

    return jsonify({
        "verdict": final_verdict,
        "ensemble_confidence": ensemble_confidence,
        "models": results,
        "real_votes": real_votes,
        "fake_votes": fake_votes
    })

if __name__ == "__main__":
    app.run(debug=True)