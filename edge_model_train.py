import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "text"]
)

df["target"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# -----------------------------
# 2. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["target"],
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

# -----------------------------
# 3. Hashing configuration
# -----------------------------
NUM_FEATURES = 512

vectorizer = HashingVectorizer(
    n_features=NUM_FEATURES,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b"
)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 4. Train model
# -----------------------------
model = SGDClassifier(
    loss="log_loss",
    penalty="l1",
    alpha=1e-4,
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\n========== EDGE MODEL RESULTS ==========")

print(classification_report(
    y_test,
    y_pred,
    target_names=["HAM", "SPAM"]
))

# -----------------------------
# 6. Model weights
# -----------------------------
weights = model.coef_[0]
bias = model.intercept_[0]

zero_weights = np.sum(weights == 0)
active_weights = NUM_FEATURES - zero_weights

print("Total Features :", NUM_FEATURES)
print("Zero Weights   :", zero_weights)
print("Active Weights :", active_weights)
print("Model Bias     :", bias)

# -----------------------------
# 7. Export C header
# -----------------------------
header_filename = "model_weights.h"

with open(header_filename, "w") as f:

    f.write("// Auto-generated EdgeSpam-IoT model\n")
    f.write("// Compatible with ESP32 deployment\n\n")

    f.write("#pragma once\n\n")

    f.write(f"#define NUM_FEATURES {NUM_FEATURES}\n\n")

    f.write(
        f"static const float MODEL_BIAS = {bias:.9f}f;\n\n"
    )

    f.write(
        "static const float MODEL_WEIGHTS[NUM_FEATURES] = {\n"
    )

    for i, w in enumerate(weights):

        f.write(f"    {w:.9f}f")

        if i < len(weights) - 1:
            f.write(",")

        if (i + 1) % 4 == 0:
            f.write("\n")
        else:
            f.write(" ")

    f.write("};\n")

print("\n========================================")
print("SUCCESS!")
print("Generated file: model_weights.h")
print("========================================")