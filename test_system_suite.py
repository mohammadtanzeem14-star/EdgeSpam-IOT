import os
import json
import pandas as pd
from app import app, BASE_DIR, BENCHMARK_PATH, DATASET_70K_PATH

def test_01_dataset_files():
    print("\n--- TEST 1: DATASET FILES INTEGRITY ---")
    assert os.path.exists(DATASET_70K_PATH), "70K dataset does not exist!"
    assert os.path.exists(os.path.join(BASE_DIR, "SMSSpamCollection")), "Baseline SMSSpamCollection does not exist!"
    
    df_70k = pd.read_csv(DATASET_70K_PATH)
    assert len(df_70k) == 70000, f"Expected 70,000 records, got {len(df_70k)}"
    assert list(df_70k.columns) == ["message", "label"], f"Unexpected columns: {df_70k.columns}"
    
    ham_cnt = (df_70k["label"] == "ham").sum()
    spam_cnt = (df_70k["label"] == "spam").sum()
    assert ham_cnt == 39475, f"Expected 39,475 HAM, got {ham_cnt}"
    assert spam_cnt == 30525, f"Expected 30,525 SPAM, got {spam_cnt}"
    assert df_70k["message"].isna().sum() == 0, "Missing messages found!"
    assert df_70k.duplicated(subset=["message"]).sum() == 0, "Duplicate messages found!"
    print(f"[OK] 70K Dataset valid: 70,000 records (HAM: {ham_cnt}, SPAM: {spam_cnt}), 0 duplicates, 0 missing.")

def test_02_benchmark_and_candidate_artifacts():
    print("\n--- TEST 2: BENCHMARK & CANDIDATE ARTIFACTS ---")
    assert os.path.exists(BENCHMARK_PATH), "benchmark_results_70k.json missing!"
    with open(BENCHMARK_PATH, "r") as f:
        data = json.load(f)
    
    assert "models" in data, "No models key in benchmark data"
    assert len(data["models"]) == 4, f"Expected 4 models, got {len(data['models'])}"
    
    model_names = [m["name"] for m in data["models"]]
    expected_names = ["SGDClassifier", "Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]
    for en in expected_names:
        assert en in model_names, f"Missing model: {en}"
        
    for m in data["models"]:
        assert "accuracy" in m and m["accuracy"] > 50, f"Invalid accuracy for {m['name']}"
        assert "precision" in m and m["precision"] > 50, f"Invalid precision for {m['name']}"
        assert "recall" in m and m["recall"] > 40, f"Invalid recall for {m['name']}"
        assert "f1_score" in m and m["f1_score"] > 50, f"Invalid f1_score for {m['name']}"
        assert "confusion_matrix" in m, f"Missing confusion_matrix for {m['name']}"
        cm = m["confusion_matrix"]
        assert all(k in cm for k in ["tn", "fp", "fn", "tp"]), f"Incomplete confusion matrix for {m['name']}"
        assert "train_time_ms" in m, f"Missing train_time_ms for {m['name']}"
        print(f"  * {m['name']:<25}: Acc={m['accuracy']}% | Prec={m['precision']}% | Rec={m['recall']}% | F1={m['f1_score']}% | CM={cm}")

    assert os.path.exists(os.path.join(BASE_DIR, "model_70k_candidate.joblib")), "Candidate model missing!"
    assert os.path.exists(os.path.join(BASE_DIR, "model.joblib.bak_5k")), "Backup model missing!"
    print("[OK] Benchmark results & artifacts verified.")

def test_03_frontend_routes():
    print("\n--- TEST 3: FRONTEND ROUTES ---")
    client = app.test_client()
    
    # 1. User portal
    res = client.get("/")
    assert res.status_code == 200
    assert b"EdgeSpam-IoT" in res.data
    assert b"User Portal" in res.data or b"user-pill" in res.data
    
    # 2. Admin Login
    res = client.get("/admin")
    assert res.status_code == 200
    assert b"Administrator Sign In" in res.data
    
    # 3. Admin Dashboard
    res = client.get("/admin/dashboard")
    assert res.status_code == 200
    assert b"sms_spam_cleaned_70000.csv" in res.data
    assert b"70,000" in res.data
    assert b"39,475" in res.data
    assert b"30,525" in res.data
    assert b"Confusion Matrix" in res.data
    print("[OK] Frontend routes (/, /admin, /admin/dashboard) returned HTTP 200 with required content.")

def test_04_prediction_and_history_apis():
    print("\n--- TEST 4: PREDICTION & HISTORY APIS ---")
    client = app.test_client()
    
    # Empty message test
    res = client.post("/predict", json={"message": "   "})
    assert res.status_code == 400
    
    # Spam prediction test
    spam_msg = "URGENT! You have won a 2000 prize. Call 09066362206 now!"
    res = client.post("/predict", json={"message": spam_msg})
    assert res.status_code == 200
    data = res.get_json()
    assert data["classification"] == "SPAM", f"Expected SPAM, got {data['classification']}"
    assert data["spam_probability"] > 50, f"Low spam prob: {data['spam_probability']}"
    
    # Ham prediction test
    ham_msg = "Hey mom, are you free for dinner tonight?"
    res = client.post("/predict", json={"message": ham_msg})
    assert res.status_code == 200
    data = res.get_json()
    assert data["classification"] == "HAM", f"Expected HAM, got {data['classification']}"
    assert data["spam_probability"] < 50, f"High spam prob for ham: {data['spam_probability']}"
    
    # History test
    res = client.get("/history")
    assert res.status_code == 200
    history = res.get_json()
    assert isinstance(history, list)
    assert len(history) > 0
    
    # Stats test
    res = client.get("/stats")
    assert res.status_code == 200
    stats = res.get_json()
    assert stats["total"] > 0
    assert "spam" in stats
    assert "ham" in stats
    
    print("[OK] Prediction, history, and stats APIs working properly.")

def test_05_admin_ml_workflow_apis():
    print("\n--- TEST 5: ADMIN ML WORKFLOW APIS ---")
    client = app.test_client()
    
    # Step 1: Upload / Load 70k dataset
    res = client.post("/admin/upload-dataset", json={"use_70k": True})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["total_records"] == 70000
    assert data["ham_count"] == 39475
    assert data["spam_count"] == 30525
    assert data["duplicate_messages"] == 0
    assert data["missing_messages"] == 0
    
    # Step 2: Preprocess dataset
    res = client.post("/admin/preprocess")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["processed_records"] == 70000
    assert data["feature_dimensions"] == [70000, 512]
    
    # Step 3: Train / evaluate 4 algorithms
    res = client.post("/admin/train")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["data"]["models"]) == 4
    
    # Step 4: Comparison Graph data
    res = client.get("/admin/comparison")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "models" in data["data"]
    assert len(data["data"]["models"]) == 4
    
    # Test fallback to 5k dataset if requested
    res_5k = client.post("/admin/upload-dataset", json={"use_default": True})
    assert res_5k.status_code == 200
    data_5k = res_5k.get_json()
    assert data_5k["total_records"] == 5572
    assert data_5k["filename"] == "SMSSpamCollection"
    
    # Reload 70k for default state
    client.post("/admin/upload-dataset", json={"use_70k": True})
    
    print("[OK] All 4 Admin ML workflow modules and APIs tested successfully.")

if __name__ == "__main__":
    test_01_dataset_files()
    test_02_benchmark_and_candidate_artifacts()
    test_03_frontend_routes()
    test_04_prediction_and_history_apis()
    test_05_admin_ml_workflow_apis()
    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (100% OK)")
    print("==========================================")
