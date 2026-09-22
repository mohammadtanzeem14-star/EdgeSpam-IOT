import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    names=["label", "text"]
)

# Convert labels to numbers
df["target"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["target"],
    test_size=0.2,
    random_state=42,
    stratify=df["target"]
)

# Feature hashing
NUM_FEATURES = 512

vectorizer = HashingVectorizer(
    n_features=NUM_FEATURES,
    alternate_sign=False
)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = SGDClassifier(
    loss="log_loss",
    penalty="l1",
    alpha=1e-4,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

print("\n--- MODEL RESULTS ---")
print(classification_report(
    y_test,
    y_pred,
    target_names=["HAM", "SPAM"]
))

# Check sparsity
weights = model.coef_[0]

zero_weights = np.sum(weights == 0)

print("\nTotal Features:", NUM_FEATURES)
print("Zero Weights:", zero_weights)
print("Active Weights:", NUM_FEATURES - zero_weights)