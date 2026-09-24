import json
from app import app

def test_frontend_regression():
    client = app.test_client()
    print("=" * 80)
    print("FRONTEND REGRESSION TEST SUITE")
    print("=" * 80)
    
    # 1. User Portal (index.html)
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200 for /, got {resp.status_code}"
    html = resp.data.decode("utf-8")
    assert "EdgeSpam" in html, "Missing EdgeSpam branding in index.html"
    assert "message" in html, "Missing SMS message input field"
    assert "Check SMS" in html, "Missing Check SMS button"
    assert "classification" in html, "Missing classification element"
    assert "probability" in html, "Missing probability element"
    print("[PASS] User Portal (/) loads successfully with all interactive elements.")
    
    # 2. Prediction API and Result Display compatibility
    ham_resp = client.post("/predict",
                           data=json.dumps({"message": "Hello friend, see you at school"}),
                           content_type="application/json")
    assert ham_resp.status_code == 200
    ham_data = ham_resp.get_json()
    assert ham_data["classification"] == "HAM"
    assert "spam_probability" in ham_data
    assert "id" in ham_data
    assert "timestamp" in ham_data
    print(f"[PASS] HAM Prediction format valid: {ham_data['classification']} ({ham_data['spam_probability']}%)")
    
    spam_resp = client.post("/predict",
                            data=json.dumps({"message": "WINNER! You won $10,000 cash. Call now!"}),
                            content_type="application/json")
    assert spam_resp.status_code == 200
    spam_data = spam_resp.get_json()
    assert spam_data["classification"] == "SPAM"
    assert "spam_probability" in spam_data
    print(f"[PASS] SPAM Prediction format valid: {spam_data['classification']} ({spam_data['spam_probability']}%)")
    
    # 3. Prediction History endpoint (/history)
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    history = hist_resp.get_json()
    assert isinstance(history, list)
    assert len(history) > 0
    latest = history[0]
    assert "id" in latest and "message" in latest and "classification" in latest and "probability" in latest
    print(f"[PASS] Prediction history (/history) returns {len(history)} items with complete fields.")
    
    # 4. Dashboard Statistics endpoint (/stats)
    stats_resp = client.get("/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.get_json()
    assert "total" in stats_data
    assert "spam" in stats_data
    assert "ham" in stats_data
    assert "average_probability" in stats_data
    print(f"[PASS] Dashboard statistics (/stats): Total={stats_data['total']}, SPAM={stats_data['spam']}, HAM={stats_data['ham']}, AvgProb={stats_data['average_probability']}%")
    
    # 5. Admin Sign In page (/admin)
    admin_resp = client.get("/admin")
    assert admin_resp.status_code == 200
    admin_html = admin_resp.data.decode("utf-8")
    assert "Admin Sign In" in admin_html
    assert "loginUsername" in admin_html
    assert "loginPassword" in admin_html
    assert "loginBtn" in admin_html
    print("[PASS] Admin Sign In page (/admin) renders successfully with credentials inputs.")
    
    # 6. Admin Dashboard page (/admin/dashboard)
    dash_resp = client.get("/admin/dashboard")
    assert dash_resp.status_code == 200
    dash_html = dash_resp.data.decode("utf-8")
    assert "Admin Dashboard" in dash_html or "Dashboard" in dash_html
    print("[PASS] Admin Dashboard (/admin/dashboard) loads successfully.")
    
    # 7. Health check endpoint (/health)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json().get("status") == "ok"
    print("[PASS] Backend health check (/health) returns status 'ok'.")
    
    print("=" * 80)
    print("ALL FRONTEND REGRESSION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_frontend_regression()
