import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

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
# 2. Hashing configuration
# -----------------------------
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

# -----------------------------
# 3. Train model
# -----------------------------
model = SGDClassifier(
    loss="log_loss",
    penalty="l1",
    alpha=1e-4,
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X, y)

# -----------------------------
# 4. SMS Demo
# -----------------------------
print("\n================================")
print("     EdgeSpam-IoT SMS Demo")
print("================================")
print("Type an SMS message.")
print("Type 'exit' to stop.\n")

while True:

    message = input("Enter SMS: ")

    if message.lower() == "exit":
        print("\nDemo stopped.")
        break

    features = vectorizer.transform([message])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1] * 100

    print("\n--- RESULT ---")

    if prediction == 1:
        print("Classification : SPAM")
    else:
        print("Classification : HAM")

    print(f"Spam Probability : {probability:.2f}%")
    print("---------------\n")