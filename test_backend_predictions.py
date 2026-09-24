import json
from app import app

def test_predictions():
    client = app.test_client()
    
    test_cases = [
        "Congratulations! You have won a free prize. Call now.",
        "Hey, are we meeting at college tomorrow?",
        "Your OTP is 482913. Do not share it.",
        "URGENT! You have won a cash prize. Call immediately.",
        "Your Amazon order has been shipped.",
        "Meeting at 10 AM tomorrow."
    ]
    
    print("=" * 80)
    print("TESTING /predict ENDPOINT WITH CANDIDATE 2048-FEATURE MODEL")
    print("=" * 80)
    
    all_passed = True
    results = []
    
    for idx, msg in enumerate(test_cases, 1):
        response = client.post("/predict",
                               data=json.dumps({"message": msg}),
                               content_type="application/json")
        status_code = response.status_code
        data = response.get_json()
        
        print(f"\n[Test Case {idx}]")
        print(f"  Message        : {msg}")
        print(f"  HTTP Status    : {status_code}")
        
        if status_code != 200:
            print(f"  ERROR: Expected 200, got {status_code}")
            all_passed = False
            continue
            
        cls = data.get("classification")
        prob = data.get("spam_probability")
        mid = data.get("id")
        ts = data.get("timestamp")
        
        print(f"  Classification : {cls}")
        print(f"  Spam Prob (%)  : {prob}%")
        print(f"  ID             : {mid}")
        print(f"  Timestamp      : {ts}")
        
        # Verify schema
        assert cls in ["HAM", "SPAM"], f"Invalid classification: {cls}"
        assert isinstance(prob, (int, float)), f"Invalid probability: {prob}"
        assert mid is not None, "Missing id"
        assert ts is not None, "Missing timestamp"
        
        results.append({
            "index": idx,
            "message": msg,
            "status": status_code,
            "classification": cls,
            "probability": prob
        })
        
    print("\n" + "=" * 80)
    print("TESTING EMPTY / INVALID MESSAGE VALIDATION")
    print("=" * 80)
    
    empty_resp = client.post("/predict",
                             data=json.dumps({"message": ""}),
                             content_type="application/json")
    print(f"Empty string: HTTP {empty_resp.status_code}, Response: {empty_resp.get_json()}")
    assert empty_resp.status_code == 400
    assert "SMS message cannot be empty." in empty_resp.get_json().get("error", "")
    
    whitespace_resp = client.post("/predict",
                                  data=json.dumps({"message": "   "}),
                                  content_type="application/json")
    print(f"Whitespace: HTTP {whitespace_resp.status_code}, Response: {whitespace_resp.get_json()}")
    assert whitespace_resp.status_code == 400
    assert "SMS message cannot be empty." in whitespace_resp.get_json().get("error", "")
    
    missing_resp = client.post("/predict",
                               data=json.dumps({}),
                               content_type="application/json")
    print(f"Missing field: HTTP {missing_resp.status_code}, Response: {missing_resp.get_json()}")
    assert missing_resp.status_code == 400
    
    print("\n" + "=" * 80)
    print("TESTING /api/predict ENDPOINT ALIAS")
    print("=" * 80)
    api_resp = client.post("/api/predict",
                           data=json.dumps({"message": "Test API alias call"}),
                           content_type="application/json")
    print(f"/api/predict: HTTP {api_resp.status_code}, Classification: {api_resp.get_json().get('classification')}")
    assert api_resp.status_code == 200
    
    print("\n" + "=" * 80)
    print("ALL BACKEND PREDICTION TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_predictions()
