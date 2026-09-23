import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam_cleaned_70000.csv")

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print("\n--- 1. DATASET VALIDATION & PREPROCESSING ---")
print(f"Initial shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Validate required columns
assert "message" in df.columns and "label" in df.columns, "Columns 'message' and 'label' required!"

# Check nulls/empty
null_count = df.isnull().sum().to_dict()
empty_msgs = int((df["message"].astype(str).str.strip() == "").sum())
print(f"Null counts: {null_count}")
print(f"Empty/whitespace messages: {empty_msgs}")

# Remove empty/null if unexpectedly present
df = df.dropna(subset=["message", "label"]).copy()
df = df[df["message"].astype(str).str.strip() != ""].copy()

# Normalize labels
df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
valid_labels = df["label_norm"].isin(["ham", "spam"])
print(f"Valid label count: {int(valid_labels.sum())} / {len(df)}")
df = df[valid_labels].copy()

df["target"] = df["label_norm"].map({"ham": 0, "spam": 1}).astype(int)

# Check duplicates
dup_rows = int(df.duplicated().sum())
dup_msgs = int(df.duplicated(subset=["message"]).sum())
print(f"Duplicate rows: {dup_rows}")
print(f"Duplicate messages: {dup_msgs}")

ham_count = int((df["target"] == 0).sum())
spam_count = int((df["target"] == 1).sum())
total_records = int(len(df))
print(f"Final records: {total_records} (HAM: {ham_count}, SPAM: {spam_count})")

# Feature Vectorization
NUM_FEATURES = 512
vectorizer = HashingVectorizer(
    n_features=NUM_FEATURES,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b"
)

# Stratified 80/20 train-test split
print("\n--- 2. STRATIFIED 80/20 SPLIT (random_state=42) ---")
X_text = df["message"].astype(str).str.lower()
y = df["target"].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train samples: {len(y_train)} (HAM: {int((y_train == 0).sum())}, SPAM: {int((y_train == 1).sum())})")
print(f"Test samples : {len(y_test)} (HAM: {int((y_test == 0).sum())}, SPAM: {int((y_test == 1).sum())})")

print("\n--- 3. FEATURE EXTRACTION ---")
t_vec_start = time.time()
X_train_vec = vectorizer.transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)
vec_time_ms = round((time.time() - t_vec_start) * 1000, 2)
print(f"HashingVectorizer finished in {vec_time_ms} ms. Shape: {X_train_vec.shape}")

# 4 Algorithms
model_configs = {
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
trained_models = {}

print("\n--- 4. TRAINING & EVALUATING 4 ALGORITHMS ---")
for name, clf in model_configs.items():
    print(f"\nTraining {name}...")
    t0 = time.time()
    clf.fit(X_train_vec, y_train)
    train_time_ms = round((time.time() - t0) * 1000, 2)
    trained_models[name] = clf
    
    # Predict on test set
    y_pred = clf.predict(X_test_vec)
    
    acc = round(float(accuracy_score(y_test, y_pred) * 100), 2)
    prec = round(float(precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)
    rec = round(float(recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)
    f1 = round(float(f1_score(y_test, y_pred, pos_label=1, zero_division=0) * 100), 2)
    cm = confusion_matrix(y_test, y_pred) # [[TN, FP], [FN, TP]]
    
    res = {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "train_time_ms": train_time_ms,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        },
        "is_live_model": (name == "SGDClassifier")
    }
    results.append(res)
    
    print(f"Results for {name}:")
    print(f"  Accuracy       : {acc}%")
    print(f"  Precision(Spam): {prec}%")
    print(f"  Recall(Spam)   : {rec}%")
    print(f"  F1-Score       : {f1}%")
    print(f"  ConfusionMatrix: TN={int(cm[0, 0])}, FP={int(cm[0, 1])}, FN={int(cm[1, 0])}, TP={int(cm[1, 1])}")
    print(f"  Training Time  : {train_time_ms} ms")

# Determine best model by F1-score
best_model = max(results, key=lambda x: x["f1_score"])
print(f"\nBest Model by F1-Score: {best_model['name']} ({best_model['f1_score']}%)")

# Save benchmark results JSON
benchmark_data = {
    "dataset": {
        "filename": "sms_spam_cleaned_70000.csv",
        "total_records": int(total_records),
        "ham_count": int(ham_count),
        "spam_count": int(spam_count),
        "num_features": int(NUM_FEATURES),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "duplicate_messages": int(dup_msgs),
        "missing_messages": 0
    },
    "best_model": best_model["name"],
    "models": results,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

benchmark_json_path = os.path.join(BASE_DIR, "data", "benchmark_results_70k.json")
with open(benchmark_json_path, "w") as f:
    json.dump(benchmark_data, f, indent=2)
print(f"\nSaved benchmark results to: {benchmark_json_path}")

# Save candidate model (SGDClassifier is the live edge-compatible model)
candidate_model_path = os.path.join(BASE_DIR, "model_70k_candidate.joblib")
joblib.dump(trained_models["SGDClassifier"], candidate_model_path)
print(f"Saved candidate SGDClassifier model to: {candidate_model_path}")

print("\n--- 5. CHECKING WEIGHT SPARSITY FOR SGD (EDGE COMPATIBILITY) ---")
sgd = trained_models["SGDClassifier"]
weights = sgd.coef_[0]
bias = float(sgd.intercept_[0])
zero_weights = int(np.sum(weights == 0))
active_weights = int(NUM_FEATURES - zero_weights)
print(f"Total Features : {NUM_FEATURES}")
print(f"Zero Weights   : {zero_weights} ({round(zero_weights / NUM_FEATURES * 100, 1)}%)")
print(f"Active Weights : {active_weights}")
print(f"Model Bias     : {bias}")
print("\nTraining and evaluation successfully completed!")
