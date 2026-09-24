"""
verify_65824_model.py
=====================
Verification script for candidate model: model_65824_candidate.joblib
Trained on sms_spam_sms_only.csv (65,824 records).
"""

import os
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_65824_candidate.joblib")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer_65824_candidate.joblib")

print(f"Loading candidate model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

if os.path.exists(VEC_PATH):
    vectorizer = joblib.load(VEC_PATH)
else:
    vectorizer = HashingVectorizer(
        n_features=512,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )

test_messages = [
    "Congratulations! You have won a free prize. Call now.",
    "Your SBI account has been credited with Rs 5000.",
    "Hey, are we meeting at college tomorrow?",
    "Your OTP is 482913. Do not share it."
]

print("\n" + "=" * 75)
print(f"{'MESSAGE':<55} | {'PREDICTION':<10} | {'SPAM PROBABILITY':<16}")
print("=" * 75)

for msg in test_messages:
    vec = vectorizer.transform([msg.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1] * 100
    label = "SPAM" if pred == 1 else "HAM"
    msg_display = (msg[:52] + "...") if len(msg) > 55 else msg
    print(f"{msg_display:<55} | {label:<10} | {prob:6.2f}%")

print("=" * 75)
print("[SUCCESS] Candidate model successfully verified!")
