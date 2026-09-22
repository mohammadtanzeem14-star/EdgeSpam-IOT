import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "Templates"))

# In Vercel serverless environments, root is read-only; /tmp is writable
if os.environ.get("VERCEL"):
    DATABASE = "/tmp/predictions.db"
else:
    DATABASE = os.path.join(BASE_DIR, "predictions.db")


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def init_database():

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


init_database()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

DATASET_PATH = os.path.join(BASE_DIR, "SMSSpamCollection")

df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    names=["label", "text"]
)

df["target"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


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

print("ML model loaded successfully!")
print("Database initialized successfully!")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# --------------------------------------------------
# PREDICT SMS
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "message" not in data:

        return jsonify({
            "error": "Please provide an SMS message."
        }), 400

    message = data["message"]

    if not message.strip():

        return jsonify({
            "error": "SMS message cannot be empty."
        }), 400

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1] * 100

    if prediction == 1:

        classification = "SPAM"

    else:

        classification = "HAM"

    probability = round(probability, 2)

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # Save prediction

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (message, classification, probability, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        message,
        classification,
        probability,
        timestamp
    ))

    connection.commit()
    connection.close()


    return jsonify({

        "message": message,

        "classification": classification,

        "spam_probability": probability,

        "timestamp": timestamp

    })


# --------------------------------------------------
# GET PREDICTION HISTORY
# --------------------------------------------------

@app.route("/history", methods=["GET"])
def history():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, message, classification, probability, timestamp
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    history_data = []

    for row in rows:

        history_data.append({

            "id": row[0],

            "message": row[1],

            "classification": row[2],

            "probability": row[3],

            "timestamp": row[4]

        })

    return jsonify(history_data)


# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------

@app.route("/stats", methods=["GET"])
def stats():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()


    # Total predictions

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
    """)

    total = cursor.fetchone()[0]


    # Total SPAM

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE classification = 'SPAM'
    """)

    spam = cursor.fetchone()[0]


    # Total HAM

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE classification = 'HAM'
    """)

    ham = cursor.fetchone()[0]


    # Average spam probability

    cursor.execute("""
        SELECT AVG(probability)
        FROM predictions
    """)

    average_probability = cursor.fetchone()[0]


    connection.close()


    if average_probability is None:

        average_probability = 0

    else:

        average_probability = round(
            average_probability,
            2
        )


    return jsonify({

        "total": total,

        "spam": spam,

        "ham": ham,

        "average_probability": average_probability

    })


# --------------------------------------------------
# CLEAR HISTORY
# --------------------------------------------------

@app.route("/clear-history", methods=["DELETE"])
def clear_history():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM predictions
    """)

    connection.commit()
    connection.close()

    return jsonify({

        "message":
        "Prediction history cleared successfully."

    })


# --------------------------------------------------
# RUN FLASK
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)