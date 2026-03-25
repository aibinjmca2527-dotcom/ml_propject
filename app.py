"""
app.py - Flask Web Application for Email Spam Detection
"""

import os
import re
import pickle
import sqlite3
import hashlib
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash
)

app = Flask(__name__)
app.secret_key = "spamdetector_secret_2024"

MODEL_PATH      = os.path.join("model", "spam_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")
DB_PATH         = "users.db"

_model      = None
_vectorizer = None

def get_model():
    global _model, _vectorizer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None, None
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)
    return _model, _vectorizer


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    UNIQUE NOT NULL,
            email    TEXT    UNIQUE NOT NULL,
            password TEXT    NOT NULL,
            created  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            message    TEXT    NOT NULL,
            result     TEXT    NOT NULL,
            confidence REAL    NOT NULL,
            checked    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── ROUTES ───────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        if not username or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("register.html")
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, hash_password(password))
            )
            conn.commit()
            conn.close()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT id, username FROM users WHERE email=? AND password=?",
            (email, hash_password(password))
        )
        user = c.fetchone()
        conn.close()
        if user:
            session["user_id"]  = user[0]
            session["username"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT message, result, confidence, checked FROM history "
        "WHERE user_id=? ORDER BY checked DESC LIMIT 10",
        (session["user_id"],)
    )
    history = c.fetchall()
    c.execute("SELECT COUNT(*) FROM history WHERE user_id=?", (session["user_id"],))
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE user_id=? AND result='SPAM'", (session["user_id"],))
    spam_count = c.fetchone()[0]
    conn.close()
    return render_template(
        "dashboard.html",
        history=history,
        total=total,
        spam_count=spam_count,
        ham_count=total - spam_count
    )


# ── PREDICT ──────────────────────────────────
# NOTE: No @login_required decorator here — we handle it manually
# so we can return JSON instead of an HTML redirect (which breaks AJAX)
@app.route("/predict", methods=["POST"])
def predict():
    # Manual session check — returns JSON error instead of redirect
    if "user_id" not in session:
        return jsonify({"error": "Session expired. Please log in again."}), 401

    try:
        model, vectorizer = get_model()

        if model is None:
            return jsonify({"error": "Model not found. Please run: python train.py"}), 503

        # get_json(silent=True) returns None instead of raising exception
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid request data."}), 400

        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify({"error": "Please enter a message."}), 400

        clean      = preprocess_text(message)
        vec        = vectorizer.transform([clean])
        pred       = model.predict(vec)[0]
        proba      = model.predict_proba(vec)[0]
        result     = "SPAM" if pred == 1 else "HAM"
        confidence = float(max(proba)) * 100

        # Save to history
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO history (user_id, message, result, confidence) VALUES (?,?,?,?)",
            (session["user_id"], message[:500], result, round(confidence, 2))
        )
        conn.commit()
        conn.close()

        return jsonify({
            "result":     result,
            "confidence": round(confidence, 2),
            "is_spam":    bool(pred == 1)
        })

    except Exception as e:
        # Log the real error to terminal, return clean message to browser
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Prediction failed: " + str(e)}), 500


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    print("Starting SpamShield...")
    print("Open: http://127.0.0.1:5000")
    app.run(debug=True)