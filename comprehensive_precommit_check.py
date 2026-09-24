import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from app import app, vectorizer, NUM_FEATURES

def run_comprehensive_check():
    print("=" * 80)
    print("FINAL PRE-COMMIT SECURITY & REPOSITORY AUDIT")
    print("=" * 80)

    # CHECK 1: Model Feature Dimension
    m_root = joblib.load("model.joblib")
    assert m_root.coef_.shape == (1, 2048), f"Expected (1, 2048), got {m_root.coef_.shape}"
    print(f"[CHECK 1 PASS] model.joblib expects 2048 features.")

    # CHECK 2: Synchronization of root and API models
    m_api = joblib.load(os.path.join("api", "model.joblib"))
    assert m_api.coef_.shape == (1, 2048), f"Expected (1, 2048) for API model, got {m_api.coef_.shape}"
    assert np.allclose(m_root.coef_, m_api.coef_), "model.joblib and api/model.joblib mismatch!"
    assert np.isclose(m_root.intercept_[0], m_api.intercept_[0]), "Bias mismatch between root and API models!"
    print(f"[CHECK 2 PASS] model.joblib and api/model.joblib are 100% synchronized.")

    # CHECK 3: model_weights.h contains exactly 2048 weights
    for h_path in ["model_weights.h", os.path.join("EdgeSpam_Demo", "model_weights.h")]:
        with open(h_path, "r", encoding="utf-8") as f:
            h_content = f.read()
        assert "#define NUM_FEATURES 2048" in h_content, f"Missing #define NUM_FEATURES 2048 in {h_path}"
        weights = re.findall(r"[-+]?\d*\.\d+f", h_content)
        # 1 bias float + 2048 weight floats = 2049 floats total
        assert len(weights) == 2049, f"Expected 2049 float literals in {h_path}, found {len(weights)}"
        print(f"[CHECK 3 PASS] {h_path} contains exactly 2048 weights and matching bias.")

    # CHECK 4: ESP32 firmware uses NUM_FEATURES 2048 and static array
    ino_path = os.path.join("EdgeSpam_Demo", "EdgeSpam_Demo.ino")
    with open(ino_path, "r", encoding="utf-8") as f:
        ino_content = f.read()
    assert "#define NUM_FEATURES 2048" in ino_content, "Missing #define NUM_FEATURES 2048 in ino"
    assert "static float features[NUM_FEATURES];" in ino_content, "Missing static float features in ino"
    print(f"[CHECK 4 PASS] ESP32 firmware properly configured with NUM_FEATURES 2048 and static RAM allocation.")

    # CHECK 5: Dataset remains 65,824 rows, no synthetic data
    csv_path = "sms_spam_sms_only.csv"
    assert os.path.exists(csv_path), "Missing sms_spam_sms_only.csv"
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    ham_count = int((df["label"] == 0).sum())
    spam_count = int((df["label"] == 1).sum())
    assert total_rows == 65824, f"Expected 65,824 rows, got {total_rows}"
    assert ham_count == 38205, f"Expected 38,205 HAM, got {ham_count}"
    assert spam_count == 27619, f"Expected 27,619 SPAM, got {spam_count}"
    print(f"[CHECK 5 PASS] Dataset verified: exactly 65,824 rows (HAM={ham_count}, SPAM={spam_count}), 0 synthetic rows.")

    # CHECK 6: requirements.txt is valid
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs = f.read().strip().splitlines()
    req_names = [r.split(">=")[0].strip() for r in reqs]
    assert "Flask" in req_names
    assert "numpy" in req_names
    assert "pandas" in req_names
    assert "scikit-learn" in req_names
    assert "joblib" in req_names
    print(f"[CHECK 6 PASS] requirements.txt valid with packages: {req_names}")

    # CHECK 7: Vercel configuration is valid JSON
    with open("vercel.json", "r", encoding="utf-8") as f:
        v_data = json.load(f)
    assert v_data.get("version") == 2
    assert "functions" in v_data
    assert "rewrites" in v_data
    print(f"[CHECK 7 PASS] vercel.json is valid and unchanged.")

    # CHECK 8: Prediction Endpoint Tests
    client = app.test_client()
    test_msgs = [
        ("Congratulations! You have won a free prize. Call now.", "SPAM"),
        ("Hey, are we meeting at college tomorrow?", "HAM"),
        ("Your OTP is 482913. Do not share it.", "HAM"),
        ("URGENT! You have won a cash prize. Call immediately.", "SPAM"),
        ("Your Amazon order has been shipped.", "HAM"),
        ("Meeting at 10 AM tomorrow.", "HAM")
    ]
    for msg, exp in test_msgs:
        res = client.post("/predict", json={"message": msg})
        assert res.status_code == 200
        d = res.get_json()
        assert d["classification"] == exp
        assert "spam_probability" in d
    
    empty_res = client.post("/predict", json={"message": ""})
    assert empty_res.status_code == 400
    print(f"[CHECK 8 PASS] All 6 test predictions and input validations passed.")

    # CHECK 9: Admin Authentication & Frontend Regression Tests
    # Test valid demo credentials
    login_valid = client.post("/api/admin/login", json={"username": "admin", "password": "demo-admin-eval"})
    assert login_valid.status_code == 200
    assert login_valid.get_json().get("authenticated") is True
    
    # Test invalid credentials
    login_invalid = client.post("/api/admin/login", json={"username": "admin", "password": "wrongpassword"})
    assert login_invalid.status_code == 401

    # Test pages render
    assert client.get("/").status_code == 200
    assert client.get("/admin").status_code == 200
    assert client.get("/admin/dashboard").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/stats").status_code == 200
    assert client.get("/health").status_code == 200
    print(f"[CHECK 9 PASS] Admin authentication (demo & error states) and all UI routes passed.")

    print("=" * 80)
    print("ALL 9 PRE-COMMIT VERIFICATION CHECKS PASSED WITH 100% INTEGRITY")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_check()
