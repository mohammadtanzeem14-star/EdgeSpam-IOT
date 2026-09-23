import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "Templates"),
    static_folder=os.path.join(BASE_DIR, "public")
)

# In Vercel serverless environments, root is read-only; /tmp is writable
if os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    DATABASE = "/tmp/predictions.db"
else:
    DATABASE = os.path.join(BASE_DIR, "predictions.db")


# --------------------------------------------------
# CORS SUPPORT
# --------------------------------------------------

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
        return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
    return response


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def init_database():
    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                classification TEXT NOT NULL,
                probability REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        connection.commit()
        connection.close()
    except Exception as e:
        print("Database initialization note:", e)


init_database()


# --------------------------------------------------
# MACHINE LEARNING MODEL
# --------------------------------------------------

NUM_FEATURES = 512

vectorizer = HashingVectorizer(
    n_features=NUM_FEATURES,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b"
)

# Load pre-trained model for fast cold starts
model = None
model_candidate_paths = [
    os.path.join(BASE_DIR, "model.joblib"),
    os.path.join(BASE_DIR, "api", "model.joblib"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "model.joblib"),
]

for p in model_candidate_paths:
    if os.path.exists(p):
        try:
            model = joblib.load(p)
            print(f"Loaded model from {p}")
            break
        except Exception as e:
            print(f"Error loading {p}: {e}")

if model is None:
    print("Training model from SMSSpamCollection fallback...")
    import pandas as pd
    DATASET_PATH = os.path.join(BASE_DIR, "SMSSpamCollection")
    df = pd.read_csv(DATASET_PATH, sep="\t", names=["label", "text"])
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    X = vectorizer.transform(df["text"])
    y = df["target"]
    model = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=1e-4,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X, y)

print("ML model ready!")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
@app.route("/api")
@app.route("/api/")
def home():
    return render_template("index.html")


@app.route("/admin")
@app.route("/admin/")
def admin_login():
    return render_template("admin.html")


@app.route("/admin/dashboard")
@app.route("/admin/dashboard/")
def admin_dashboard():
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "EdgeSpam backend is running"})


@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Please provide an SMS message."}), 400

    message = data["message"]
    if not message.strip():
        return jsonify({"error": "SMS message cannot be empty."}), 400

    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    classification = "SPAM" if prediction == 1 else "HAM"
    probability = round(probability, 2)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    last_id = int(datetime.now().timestamp() * 1000)
    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (message, classification, probability, timestamp)
            VALUES (?, ?, ?, ?)
        """, (message, classification, probability, timestamp))
        if cursor.lastrowid:
            last_id = cursor.lastrowid
        connection.commit()
        connection.close()
    except Exception as e:
        print("Database save note:", e)

    return jsonify({
        "id": last_id,
        "message": message,
        "classification": classification,
        "spam_probability": probability,
        "timestamp": timestamp
    })


@app.route("/history", methods=["GET", "OPTIONS"])
@app.route("/api/history", methods=["GET", "OPTIONS"])
def history():
    if request.method == "OPTIONS":
        return jsonify([])

    history_data = []
    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()
        cursor.execute("""
            SELECT id, message, classification, probability, timestamp
            FROM predictions
            ORDER BY id DESC
        """)
        rows = cursor.fetchall()
        connection.close()
        for row in rows:
            history_data.append({
                "id": row[0],
                "message": row[1],
                "classification": row[2],
                "probability": row[3],
                "timestamp": row[4]
            })
    except Exception as e:
        print("Database history note:", e)

    return jsonify(history_data)


@app.route("/stats", methods=["GET", "OPTIONS"])
@app.route("/api/stats", methods=["GET", "OPTIONS"])
def stats():
    if request.method == "OPTIONS":
        return jsonify({"total": 0, "spam": 0, "ham": 0, "average_probability": 0})

    total = 0
    spam = 0
    ham = 0
    average_probability = 0

    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE classification = 'SPAM'")
        spam = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predictions WHERE classification = 'HAM'")
        ham = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(probability) FROM predictions")
        avg = cursor.fetchone()[0]
        average_probability = round(avg, 2) if avg is not None else 0
        connection.close()
    except Exception as e:
        print("Database stats note:", e)

    return jsonify({
        "total": total,
        "spam": spam,
        "ham": ham,
        "average_probability": average_probability
    })


@app.route("/clear-history", methods=["DELETE", "OPTIONS"])
@app.route("/api/clear-history", methods=["DELETE", "OPTIONS"])
def clear_history():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    try:
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()
        cursor.execute("DELETE FROM predictions")
        connection.commit()
        connection.close()
    except Exception as e:
        print("Database clear note:", e)

    return jsonify({"message": "Prediction history cleared successfully."})


# Vercel catch-all router for rewritten requests targeting /api/index
@app.route("/api/index", methods=["GET", "POST", "DELETE", "OPTIONS"])
@app.route("/api/index/", methods=["GET", "POST", "DELETE", "OPTIONS"])
def vercel_index_catchall():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    action = request.args.get("action", "").strip().lower()
    target = (request.headers.get("x-matched-path", "") or request.path).lower()

    if action == "predict" or "predict" in target or request.method == "POST":
        return predict()
    elif action == "clear-history" or "clear-history" in target or request.method == "DELETE":
        return clear_history()
    elif action == "history" or "history" in target or "hist" in target:
        return history()
    elif action == "stats" or "stats" in target:
        return stats()
    elif action == "health" or "health" in target:
        return health()
    elif action == "dashboard" or "admin/dashboard" in target:
        return render_template("dashboard.html")
    elif action == "admin" or "admin" in target:
        return render_template("admin.html")
    elif target in ["/", "", "/api", "/api/"]:
        return render_template("index.html")

    # Default fallback for GET
    return stats()


if __name__ == "__main__":
    app.run(debug=True)