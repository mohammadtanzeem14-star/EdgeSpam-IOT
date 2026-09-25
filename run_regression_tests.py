import os
import sys
import json
import unittest
import pandas as pd

# Ensure EdgeSpam_IOT root is in sys.path
BASE_DIR = r"C:\Users\Dell\EdgeSpam_IOT"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, model, DATABASE, DATASET_VERIFIED_PATH, DATASET_65K_PATH

class EdgeSpamRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    def test_01_model_loaded(self):
        """Verify the 98.80% LinearSVC pipeline is loaded."""
        self.assertIsNotNone(model, "Primary model must not be None")
        self.assertTrue(hasattr(model, "predict"), "Model must have predict method")
        self.assertTrue(hasattr(model, "decision_function"), "LinearSVC model must have decision_function")
        clf_step = model.named_steps.get("clf")
        self.assertIsNotNone(clf_step, "Pipeline must have a 'clf' step")
        self.assertEqual(clf_step.__class__.__name__, "LinearSVC", "Classifier must be LinearSVC")
        print("[PASS] Test 1: LinearSVC Primary Model is cleanly loaded")

    def test_02_get_user_portal(self):
        """Verify GET / loads user portal with Confidence / Score."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("SMS Spam Detection", html)
        self.assertIn("Confidence / Score", html)
        self.assertIn("98.80%", html)
        print("[PASS] Test 2: User portal (GET /) loads with verified accuracy & score headers")

    def test_03_admin_login(self):
        """Verify admin login with demo credentials."""
        res_ok = self.client.post("/admin/login", json={"username": "admin", "password": "demo-admin-eval"})
        self.assertEqual(res_ok.status_code, 200)
        data = res_ok.get_json()
        self.assertEqual(data.get("status"), "success")

        res_fail = self.client.post("/admin/login", json={"username": "admin", "password": "wrongpassword"})
        self.assertEqual(res_fail.status_code, 401)
        print("[PASS] Test 3: Admin authentication works correctly (200 on success, 401 on failure)")

    def test_04_admin_dashboard(self):
        """Verify admin dashboard loads verified metrics."""
        response = self.client.get("/admin/dashboard")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("98.80%", html)
        self.assertIn("9,203", html)
        self.assertIn("sms_spam_verified.csv", html)
        self.assertIn("Historical Scraped Benchmark Reference", html)
        self.assertIn("73.05%", html)
        print("[PASS] Test 4: Admin dashboard contains 98.80% primary model and preserved historical 73.05% callout")

    def test_05_health_endpoints(self):
        """Verify /health and /api/health."""
        for path in ["/health", "/api/health"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get("status"), "healthy")
            self.assertTrue(data.get("model_loaded"))
        print("[PASS] Test 5: /health and /api/health return 200 and healthy status")

    def test_06_predict_ham_personal(self):
        """Test legitimate personal HAM SMS."""
        msg = "Hey bro, are we meeting at college tomorrow for the final seminar presentation?"
        res = self.client.post("/predict", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("classification"), "HAM")
        self.assertLess(data.get("decision_score"), 0.0)
        self.assertLess(data.get("spam_probability"), 50.0)
        self.assertIn("confidence", data)
        print(f"[PASS] Test 6: HAM SMS correctly classified (score={data['decision_score']:.2f}, prob={data['spam_probability']}%, conf={data['confidence']}%)")

    def test_07_predict_ham_errand(self):
        """Test another legitimate conversational HAM SMS."""
        msg = "Hi Mum, I will be home by 6pm for dinner. Can you please keep my keys on the table?"
        res = self.client.post("/predict", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("classification"), "HAM")
        self.assertLess(data.get("decision_score"), 0.0)
        print(f"[PASS] Test 7: Conversational HAM correctly classified (score={data['decision_score']:.2f})")

    def test_08_predict_spam_lottery(self):
        """Test obvious lottery SPAM SMS."""
        msg = "URGENT! You have won a £2,000 cash prize or a brand new car! Claim now by calling 09061743811. Offer expires in 24 hours."
        res = self.client.post("/predict", json={"message": msg})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("classification"), "SPAM")
        self.assertGreater(data.get("decision_score"), 0.0)
        self.assertGreater(data.get("spam_probability"), 50.0)
        print(f"[PASS] Test 8: Lottery SPAM correctly classified (score=+{data['decision_score']:.2f}, prob={data['spam_probability']}%)")

    def test_09_predict_empty_and_whitespace(self):
        """Test empty, whitespace, and missing key validation."""
        cases = [
            {"message": ""},
            {"message": "    "},
            {},
            {"text": "wrong key"}
        ]
        for c in cases:
            res = self.client.post("/predict", json=c)
            self.assertEqual(res.status_code, 400)
            data = res.get_json()
            self.assertIn("error", data)
        print("[PASS] Test 9: Input validation correctly rejects empty/whitespace messages with 400")

    def test_10_history_and_stats(self):
        """Test /history and /stats endpoints."""
        res_h = self.client.get("/history")
        self.assertEqual(res_h.status_code, 200)
        history = res_h.get_json()
        self.assertIsInstance(history, list)
        self.assertGreaterEqual(len(history), 3)

        res_s = self.client.get("/stats")
        self.assertEqual(res_s.status_code, 200)
        stats = res_s.get_json()
        self.assertIn("total", stats)
        self.assertIn("spam", stats)
        self.assertIn("ham", stats)
        self.assertIn("average_probability", stats)
        print(f"[PASS] Test 10: /history and /stats return valid runtime records (total={stats['total']})")

    def test_11_admin_preprocess_and_train(self):
        """Test /admin/preprocess and /admin/train routes."""
        res_prep = self.client.post("/admin/preprocess")
        self.assertEqual(res_prep.status_code, 200)
        prep_data = res_prep.get_json()
        self.assertEqual(prep_data.get("processed_records"), 9203)
        self.assertEqual(prep_data.get("ham_count"), 6827)
        self.assertEqual(prep_data.get("spam_count"), 2376)

        res_train = self.client.post("/admin/train")
        self.assertEqual(res_train.status_code, 200)
        train_data = res_train.get_json()
        models = train_data.get("data", {}).get("models", [])
        self.assertGreaterEqual(len(models), 4)
        top_acc = max(m["accuracy"] for m in models)
        self.assertEqual(top_acc, 98.80)
        print(f"[PASS] Test 11: /admin/preprocess and /admin/train return verified stats (top accuracy: {top_acc}%)")

    def test_12_dataset_and_model_artifacts_preservation(self):
        """Verify historical files are preserved and new verified files exist."""
        # 1. New verified dataset
        self.assertTrue(os.path.exists(DATASET_VERIFIED_PATH), "sms_spam_verified.csv must exist")
        df = pd.read_csv(DATASET_VERIFIED_PATH)
        self.assertEqual(df.shape[0], 9203, "sms_spam_verified.csv must have exactly 9,203 data rows")
        self.assertEqual(int((df['label'] == 0).sum()), 6827, "Must have exactly 6,827 HAM")
        self.assertEqual(int((df['label'] == 1).sum()), 2376, "Must have exactly 2,376 SPAM")

        # 2. Historical dataset preserved
        self.assertTrue(os.path.exists(DATASET_65K_PATH), "sms_spam_sms_only.csv must be preserved")

        # 3. Model weights for ESP32
        weights_path = os.path.join(BASE_DIR, "model_weights.h")
        self.assertTrue(os.path.exists(weights_path), "model_weights.h must exist")
        with open(weights_path, "r", encoding="utf-8") as f:
            weights_content = f.read()
        self.assertIn("MODEL_WEIGHTS", weights_content)
        self.assertIn("95.44%", weights_content)

        # 4. Old model.joblib preserved
        old_model_path = os.path.join(BASE_DIR, "model.joblib")
        self.assertTrue(os.path.exists(old_model_path), "model.joblib must be preserved")

        print("[PASS] Test 12: All verified artifacts and historical files verified intact")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(EdgeSpamRegressionTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
