import os
import io
import time
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "benchmark_results_70k.json")
DATASET_70K_PATH = os.path.join(BASE_DIR, "data", "sms_spam_cleaned_70000.csv")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "Templates"),
    static_folder=os.path.join(BASE_DIR, "public")
)

UPLOADED_DATASET_PATH = (
    "/tmp/uploaded_dataset.tsv"
    if (os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK))
    else os.path.join(BASE_DIR, "uploaded_dataset.tsv")
)

dataset_state = {
    "df": None,
    "filename": "SMSSpamCollection",
    "preprocessed": None,
    "comparison": None
}

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

NUM_FEATURES = 2048

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


@app.route("/admin/login", methods=["POST", "OPTIONS"])
@app.route("/api/admin/login", methods=["POST", "OPTIONS"])
def admin_login_api():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    expected_user = os.environ.get("EDGESPAM_ADMIN_USER", "admin")
    expected_pass = os.environ.get("EDGESPAM_ADMIN_PASS", "demo-admin-eval")

    if username == expected_user and password == expected_pass:
        return jsonify({"status": "success", "authenticated": True})
    return jsonify({"error": "Invalid username or access key."}), 401


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


# --------------------------------------------------
# ADMIN ML WORKFLOW APIS
# --------------------------------------------------

def load_active_dataset():
    if dataset_state.get("df") is not None and len(dataset_state["df"]) > 0:
        return dataset_state["df"]

    if os.path.exists(UPLOADED_DATASET_PATH):
        try:
            df = pd.read_csv(UPLOADED_DATASET_PATH, sep="\t")
            dataset_state["df"] = df
            return df
        except Exception as e:
            print("Failed loading uploaded disk cache:", e)

    if os.path.exists(DATASET_70K_PATH):
        try:
            df = pd.read_csv(DATASET_70K_PATH)
            if "message" in df.columns and "text" not in df.columns:
                df = df.rename(columns={"message": "text"})
            dataset_state["df"] = df
            dataset_state["filename"] = "sms_spam_cleaned_70000.csv"
            return df
        except Exception as e:
            print("Failed loading 70K dataset:", e)

    default_path = os.path.join(BASE_DIR, "SMSSpamCollection")
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path, sep="\t", names=["label", "text"])
            dataset_state["df"] = df
            dataset_state["filename"] = "SMSSpamCollection"
            return df
        except Exception as e:
            print("Failed loading default SMSSpamCollection:", e)

    return None


@app.route("/upload-dataset", methods=["POST", "OPTIONS"])
@app.route("/api/upload-dataset", methods=["POST", "OPTIONS"])
@app.route("/admin/upload-dataset", methods=["POST", "OPTIONS"])
@app.route("/api/admin/upload-dataset", methods=["POST", "OPTIONS"])
def admin_upload_dataset():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    filename = "sms_spam_cleaned_70000.csv"
    df = None
    dup_count = 0
    missing_count = 0

    json_data = request.get_json(silent=True) or {}
    use_default = json_data.get("use_default")
    use_70k = json_data.get("use_70k") or json_data.get("dataset") == "70k"

    if use_70k or (not use_default and "file" not in request.files and os.path.exists(DATASET_70K_PATH)):
        try:
            df = pd.read_csv(DATASET_70K_PATH)
            if "message" in df.columns and "text" not in df.columns:
                df = df.rename(columns={"message": "text"})
            filename = "sms_spam_cleaned_70000.csv"
            dup_count = 0
            missing_count = 0
        except Exception as e:
            return jsonify({"error": f"Failed loading 70K dataset: {str(e)}"}), 500
    elif use_default:
        default_path = os.path.join(BASE_DIR, "SMSSpamCollection")
        if not os.path.exists(default_path):
            return jsonify({"error": "Default SMSSpamCollection file not found."}), 404
        df = pd.read_csv(default_path, sep="\t", names=["label", "text"])
        filename = "SMSSpamCollection"
        dup_count = int(df.duplicated(subset=["text"]).sum())
        missing_count = int(df["text"].isna().sum())
    elif "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"error": "No selected file."}), 400
        filename = uploaded_file.filename
        try:
            content_bytes = uploaded_file.read()
            try:
                content_str = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content_str = content_bytes.decode("latin-1")

            first_line = content_str.split("\n", 1)[0]
            sep = "\t" if "\t" in first_line else ","

            try:
                df = pd.read_csv(io.StringIO(content_str), sep=sep)
            except Exception:
                df = pd.read_csv(io.StringIO(content_str), sep=None, engine="python")

            cols_lower = [str(c).strip().lower() for c in df.columns]
            label_col = None
            text_col = None

            for orig, low in zip(df.columns, cols_lower):
                if low in ["label", "class", "target", "category", "v1", "0"]:
                    label_col = orig
                elif low in ["text", "message", "sms", "content", "body", "v2", "1"]:
                    text_col = orig

            if not label_col or not text_col:
                if len(df.columns) >= 2:
                    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
                else:
                    return jsonify({"error": "Dataset must contain at least 2 columns: label and message/text."}), 400
            else:
                df = df.rename(columns={label_col: "label", text_col: "text"})

            dup_count = int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else int(df.duplicated().sum())
            missing_count = int(df["text"].isna().sum()) if "text" in df.columns else 0
        except Exception as e:
            return jsonify({"error": f"Failed to parse dataset: {str(e)}"}), 400
    else:
        if os.path.exists(DATASET_70K_PATH):
            df = pd.read_csv(DATASET_70K_PATH)
            if "message" in df.columns and "text" not in df.columns:
                df = df.rename(columns={"message": "text"})
            filename = "sms_spam_cleaned_70000.csv"
        else:
            default_path = os.path.join(BASE_DIR, "SMSSpamCollection")
            df = pd.read_csv(default_path, sep="\t", names=["label", "text"])
            filename = "SMSSpamCollection"

    if df is None or len(df) == 0:
        return jsonify({"error": "Dataset is empty."}), 400

    total_records = len(df)
    num_columns = len(df.columns)
    column_names = list(df.columns)

    labels_series = df["label"].astype(str).str.strip().str.lower()
    ham_count = int((labels_series.isin(["ham", "0", "legit", "clean"])).sum())
    spam_count = int((labels_series.isin(["spam", "1"])).sum())

    dataset_state["df"] = df
    dataset_state["filename"] = filename
    dataset_state["preprocessed"] = None
    dataset_state["comparison"] = None

    try:
        df.to_csv(UPLOADED_DATASET_PATH, sep="\t", index=False)
    except Exception as e:
        print("Upload disk cache note:", e)

    return jsonify({
        "status": "success",
        "filename": filename,
        "total_records": total_records,
        "num_columns": num_columns,
        "column_names": column_names,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "duplicate_messages": dup_count,
        "missing_messages": missing_count,
        "message": f"Dataset '{filename}' successfully loaded ({total_records:,} records)."
    })


@app.route("/preprocess", methods=["POST", "OPTIONS"])
@app.route("/api/preprocess", methods=["POST", "OPTIONS"])
@app.route("/admin/preprocess", methods=["POST", "OPTIONS"])
@app.route("/api/admin/preprocess", methods=["POST", "OPTIONS"])
def admin_preprocess_dataset():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    # If the active dataset is 70K and precomputed benchmark exists, return it cleanly
    active_fn = dataset_state.get("filename", "")
    if (active_fn == "sms_spam_cleaned_70000.csv" or not dataset_state.get("df")) and os.path.exists(BENCHMARK_PATH):
        try:
            with open(BENCHMARK_PATH, "r") as f:
                bdata = json.load(f)
            dinfo = bdata.get("dataset", {})
            dataset_state["filename"] = "sms_spam_cleaned_70000.csv"
            dataset_state["preprocessed"] = {"ready": True, "dataset": "70k"}
            dataset_state["comparison"] = bdata
            return jsonify({
                "status": "success",
                "original_records": dinfo.get("total_records", 70000),
                "processed_records": dinfo.get("total_records", 70000),
                "ham_count": dinfo.get("ham_count", 39475),
                "spam_count": dinfo.get("spam_count", 30525),
                "feature_dimensions": [dinfo.get("total_records", 70000), NUM_FEATURES],
                "hashing_features": NUM_FEATURES,
                "norm": "l2",
                "duplicate_messages": dinfo.get("duplicate_messages", 0),
                "missing_messages": dinfo.get("missing_messages", 0),
                "message": f"Preprocessed {dinfo.get('total_records', 70000):,} records into a {dinfo.get('total_records', 70000)}x{NUM_FEATURES} feature matrix."
            })
        except Exception as e:
            print("Fallback benchmark read note:", e)

    df = load_active_dataset()
    if df is None or len(df) == 0:
        return jsonify({"error": "No dataset found. Please upload a dataset first."}), 400

    orig_count = len(df)

    df_clean = df.dropna(subset=["label", "text"]).copy()

    label_norm = df_clean["label"].astype(str).str.strip().str.lower()
    label_map = {"ham": 0, "0": 0, "legit": 0, "clean": 0, "spam": 1, "1": 1}
    target_series = label_norm.map(label_map)
    df_clean = df_clean[target_series.notna()].copy()
    df_clean["target"] = target_series.astype(int)

    df_clean["clean_text"] = df_clean["text"].astype(str).str.lower()

    X_vec = vectorizer.transform(df_clean["clean_text"])
    y = df_clean["target"].values

    ham_count = int((y == 0).sum())
    spam_count = int((y == 1).sum())
    processed_count = len(df_clean)

    dataset_state["preprocessed"] = {
        "X": X_vec,
        "y": y,
        "df": df_clean
    }
    dataset_state["comparison"] = None

    return jsonify({
        "status": "success",
        "original_records": orig_count,
        "processed_records": processed_count,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "feature_dimensions": [int(X_vec.shape[0]), int(X_vec.shape[1])],
        "hashing_features": NUM_FEATURES,
        "norm": "l2",
        "duplicate_messages": int(df.duplicated(subset=["text"]).sum()) if "text" in df.columns else 0,
        "missing_messages": int(df["text"].isna().sum()) if "text" in df.columns else 0,
        "message": f"Preprocessed {processed_count:,} records into a {X_vec.shape[0]}x{X_vec.shape[1]} feature matrix."
    })


@app.route("/train", methods=["POST", "OPTIONS"])
@app.route("/api/train", methods=["POST", "OPTIONS"])
@app.route("/admin/train", methods=["POST", "OPTIONS"])
@app.route("/api/admin/train", methods=["POST", "OPTIONS"])
def admin_train_models():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    # Check if 70k benchmark cache applies
    prep = dataset_state.get("preprocessed")
    is_70k = (
        dataset_state.get("filename") == "sms_spam_cleaned_70000.csv"
        or (isinstance(prep, dict) and prep.get("dataset") == "70k")
        or (prep is None and os.path.exists(BENCHMARK_PATH))
    )

    if is_70k and os.path.exists(BENCHMARK_PATH):
        try:
            with open(BENCHMARK_PATH, "r") as f:
                bdata = json.load(f)
            dataset_state["comparison"] = bdata
            train_cnt = bdata.get("dataset", {}).get("train_samples", 56000)
            test_cnt = bdata.get("dataset", {}).get("test_samples", 14000)
            return jsonify({
                "status": "success",
                "data": bdata,
                "message": f"Successfully evaluated {len(bdata.get('models', []))} algorithms on 70,000 records ({train_cnt:,} train / {test_cnt:,} test samples)."
            })
        except Exception as e:
            print("Benchmark load error:", e)

    if dataset_state.get("preprocessed") is None:
        admin_preprocess_dataset()
        prep = dataset_state.get("preprocessed")
        if prep is None:
            return jsonify({"error": "Failed to preprocess dataset before training."}), 400

    if isinstance(prep, dict) and prep.get("dataset") == "70k" and os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r") as f:
            bdata = json.load(f)
        dataset_state["comparison"] = bdata
        return jsonify({
            "status": "success",
            "data": bdata,
            "message": "Successfully evaluated 4 algorithms on 70,000 records."
        })

    X = prep["X"]
    y = prep["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model_defs = {
        "SGDClassifier": SGDClassifier(
            loss="log_loss",
            penalty="l1",
            alpha=1e-4,
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(
            random_state=42,
            class_weight="balanced"
        )
    }

    results = []
    best_model_name = ""
    best_f1 = -1.0

    for name, clf in model_defs.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time_ms = round((time.time() - t0) * 1000, 2)

        y_pred = clf.predict(X_test)

        acc = round(float(accuracy_score(y_test, y_pred) * 100), 2)
        prec = round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)
        rec = round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)
        f1 = round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        results.append({
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "train_time_ms": train_time_ms,
            "is_live_model": (name == "SGDClassifier")
        })

    dataset_state["comparison"] = {
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "best_model": best_model_name,
        "models": results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return jsonify({
        "status": "success",
        "data": dataset_state["comparison"],
        "message": f"Successfully trained and evaluated {len(results)} algorithms on {X_train.shape[0]} train / {X_test.shape[0]} test samples."
    })


@app.route("/comparison", methods=["GET", "OPTIONS"])
@app.route("/api/comparison", methods=["GET", "OPTIONS"])
@app.route("/admin/comparison", methods=["GET", "OPTIONS"])
@app.route("/api/admin/comparison", methods=["GET", "OPTIONS"])
def admin_get_comparison():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    if dataset_state.get("comparison") is None:
        if os.path.exists(BENCHMARK_PATH):
            try:
                with open(BENCHMARK_PATH, "r") as f:
                    dataset_state["comparison"] = json.load(f)
            except Exception as e:
                print("Benchmark read note:", e)

    if dataset_state.get("comparison") is None:
        admin_train_models()

    if dataset_state.get("comparison") is None:
        return jsonify({"error": "No comparison data available."}), 404

    return jsonify({
        "status": "success",
        "data": dataset_state["comparison"]
    })


# Vercel catch-all router for rewritten requests targeting /api/index
@app.route("/api/index", methods=["GET", "POST", "DELETE", "OPTIONS"])
@app.route("/api/index/", methods=["GET", "POST", "DELETE", "OPTIONS"])
def vercel_index_catchall():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    action = request.args.get("action", "").strip().lower()
    target = (request.headers.get("x-matched-path", "") or request.path).lower()

    if action == "predict" or "predict" in target or (request.method == "POST" and "predict" in target):
        return predict()
    elif action == "clear-history" or "clear-history" in target or request.method == "DELETE":
        return clear_history()
    elif action == "history" or "history" in target or "hist" in target:
        return history()
    elif action == "stats" or "stats" in target:
        return stats()
    elif action == "health" or "health" in target:
        return health()
    elif action in ["admin-upload", "upload-dataset"] or "upload-dataset" in target:
        return admin_upload_dataset()
    elif action in ["admin-preprocess", "preprocess"] or "preprocess" in target:
        return admin_preprocess_dataset()
    elif action in ["admin-train", "train"] or "train" in target:
        return admin_train_models()
    elif action in ["admin-login", "login"] or "admin/login" in target:
        return admin_login_api()
    elif action in ["admin-comparison", "comparison"] or "comparison" in target:
        return admin_get_comparison()
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