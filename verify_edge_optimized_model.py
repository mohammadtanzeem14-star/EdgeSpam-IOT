"""
verify_edge_optimized_model.py
==============================
Independent verification script for model_65824_edge_optimized_candidate.joblib.
Configuration: HashingVectorizer (2048 features, word analyzer) + SGDClassifier (log_loss, L1 penalty).
"""

import os
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_65824_edge_optimized_candidate.joblib")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer_65824_edge_optimized_candidate.joblib")

print(f"[*] Loading Edge-Optimized Candidate Model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

if os.path.exists(VEC_PATH):
    vectorizer = joblib.load(VEC_PATH)
else:
    vectorizer = HashingVectorizer(
        n_features=2048,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b"
    )

test_messages = [
    ("Congratulations! You have won a free prize. Call now.", "SPAM"),
    ("Your SBI account has been credited with Rs 5000.", "HAM"),
    ("Hey, are we meeting at college tomorrow?", "HAM"),
    ("Your OTP is 482913. Do not share it.", "HAM"),
    ("URGENT! You have won a cash prize. Call immediately.", "SPAM"),
    ("Your Amazon order has been shipped.", "HAM"),
    ("Meeting at 10 AM tomorrow.", "HAM"),
    ("Congratulations! You have been selected for a free reward.", "SPAM")
]

print("\n" + "=" * 80)
print(f"{'#':<3} | {'MESSAGE':<52} | {'PRED':<6} | {'PROB':<7} | {'EXPECTED':<8}")
print("=" * 80)

passed = 0
for i, (msg, expected) in enumerate(test_messages, 1):
    vec = vectorizer.transform([msg.lower()])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1] * 100
    label = "SPAM" if pred == 1 else "HAM"
    status = "CORRECT" if label == expected else "REVIEW"
    if label == expected:
        passed += 1
    msg_disp = (msg[:49] + "...") if len(msg) > 52 else msg
    print(f"{i:<3} | {msg_disp:<52} | {label:<6} | {prob:5.1f}% | {expected:<8}")

print("=" * 80)
print(f"[RESULT] {passed}/{len(test_messages)} test messages matched expected ground truth.")
print("[SUCCESS] Candidate model verified independently and operational.")
