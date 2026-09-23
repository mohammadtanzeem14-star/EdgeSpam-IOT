import os
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorizer = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    token_pattern=r"(?u)\b\w\w+\b"
)

candidate_path = os.path.join(BASE_DIR, "model_70k_candidate.joblib")
backup_5k_path = os.path.join(BASE_DIR, "model.joblib.bak_5k")

print(f"Loading candidate model: {candidate_path}")
m_candidate = joblib.load(candidate_path)

print(f"Loading baseline 5k model: {backup_5k_path}")
m_5k = joblib.load(backup_5k_path)

test_messages = [
    # SPAM cases
    ("WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.", "SPAM"),
    ("URGENT! Your Mobile number has been awarded with a £2000 prize. Call 09066362206 from landline now!", "SPAM"),
    ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question", "SPAM"),
    ("Congratulations! You have won a $1000 Walmart Gift Card. Click here: http://bit.ly/claim-prize now!", "SPAM"),
    ("Your account has been suspended due to unusual activity. Click http://bank-verify.com to verify your identity immediately.", "SPAM"),
    # HAM cases
    ("Hey mom, are you free for dinner tonight? I can pick up groceries on the way.", "HAM"),
    ("Sure thing, let's meet at the coffee shop at 4pm.", "HAM"),
    ("Can you please send me the report when you get a chance? Thanks!", "HAM"),
    ("Your verification code is 482910. It will expire in 10 minutes. Do not share this code.", "HAM"),
    ("Good morning! Hope you have a wonderful day ahead.", "HAM"),
    ("I am on my way home now, see you soon.", "HAM")
]

print("\n" + "="*80)
print(f"{'MESSAGE':<50} | {'EXPECTED':<8} | {'70K CANDIDATE':<15} | {'5K BASELINE':<15}")
print("="*80)

passed = 0
total = len(test_messages)

for msg, expected in test_messages:
    vec = vectorizer.transform([msg.lower()])
    
    # Candidate
    pred_cand = m_candidate.predict(vec)[0]
    prob_cand = m_candidate.predict_proba(vec)[0][1] * 100
    label_cand = "SPAM" if pred_cand == 1 else "HAM"
    
    # 5K Baseline
    pred_5k = m_5k.predict(vec)[0]
    prob_5k = m_5k.predict_proba(vec)[0][1] * 100
    label_5k = "SPAM" if pred_5k == 1 else "HAM"
    
    is_correct = (label_cand == expected)
    if is_correct:
        passed += 1
    
    msg_short = (msg[:47] + "...") if len(msg) > 50 else msg
    print(f"{msg_short:<50} | {expected:<8} | {label_cand} ({prob_cand:5.1f}%) | {label_5k} ({prob_5k:5.1f}%)")

print("="*80)
print(f"Candidate Model Accuracy on test samples: {passed}/{total} ({passed/total*100:.1f}%)")
assert passed >= total * 0.8, "Candidate model failed too many test samples!"
print("Verification passed! The candidate model is working properly.")
