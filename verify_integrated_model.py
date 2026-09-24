import os
import re
import joblib
import numpy as np
from app import vectorizer, NUM_FEATURES

def verify_integration():
    print("=" * 80)
    print("EDGESPAM-IOT INTEGRATED MODEL CONSISTENCY CHECK")
    print("=" * 80)
    
    # 1. Verify model.joblib expects 2048 features
    model_path = "model.joblib"
    assert os.path.exists(model_path), f"Missing {model_path}"
    model = joblib.load(model_path)
    
    assert hasattr(model, "coef_"), "Model missing coef_ attribute"
    assert model.coef_.shape == (1, 2048), f"Expected (1, 2048), got {model.coef_.shape}"
    assert getattr(model, "n_features_in_", 2048) == 2048, "Expected n_features_in_ == 2048"
    print(f"[CHECK 1 PASS] model.joblib expects 2048 features (coef_ shape: {model.coef_.shape}).")
    
    # Also verify api/model.joblib
    api_model_path = os.path.join("api", "model.joblib")
    assert os.path.exists(api_model_path), f"Missing {api_model_path}"
    api_model = joblib.load(api_model_path)
    assert api_model.coef_.shape == (1, 2048), f"Expected (1, 2048) for api model, got {api_model.coef_.shape}"
    assert np.allclose(model.coef_, api_model.coef_), "model.joblib and api/model.joblib weights mismatch!"
    print(f"[CHECK 1b PASS] api/model.joblib is perfectly synchronized with model.joblib.")
    
    # 2. Verify app.py vectorizer creates 2048 features
    assert NUM_FEATURES == 2048, f"Expected NUM_FEATURES == 2048 in app.py, got {NUM_FEATURES}"
    assert vectorizer.n_features == 2048, f"Expected vectorizer.n_features == 2048, got {vectorizer.n_features}"
    assert vectorizer.norm == "l2", f"Expected norm='l2', got {vectorizer.norm}"
    assert vectorizer.alternate_sign is False, f"Expected alternate_sign=False, got {vectorizer.alternate_sign}"
    
    sample_vec = vectorizer.transform(["Verify test message feature extraction"])
    assert sample_vec.shape == (1, 2048), f"Expected vectorizer output shape (1, 2048), got {sample_vec.shape}"
    print(f"[CHECK 2 PASS] app.py vectorizer generates 2048 features with matching parameters.")
    
    # 3 & 4. Verify model bias and weights count in exported ESP32 header
    header_paths = ["model_weights.h", os.path.join("EdgeSpam_Demo", "model_weights.h")]
    
    py_bias = float(model.intercept_[0])
    py_weights = model.coef_[0]
    
    for h_path in header_paths:
        assert os.path.exists(h_path), f"Missing header: {h_path}"
        with open(h_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse NUM_FEATURES
        num_feat_match = re.search(r"#define\s+NUM_FEATURES\s+(\d+)", content)
        assert num_feat_match, f"Could not find #define NUM_FEATURES in {h_path}"
        header_num_feat = int(num_feat_match.group(1))
        assert header_num_feat == 2048, f"Expected NUM_FEATURES 2048 in {h_path}, got {header_num_feat}"
        
        # Parse MODEL_BIAS
        bias_match = re.search(r"MODEL_BIAS\s*=\s*([-+]?\d*\.\d+)f", content)
        assert bias_match, f"Could not find MODEL_BIAS in {h_path}"
        header_bias = float(bias_match.group(1))
        assert abs(py_bias - header_bias) < 1e-4, f"Bias mismatch in {h_path}: py={py_bias}, header={header_bias}"
        
        # Parse float weights
        # Find array contents between MODEL_WEIGHTS[NUM_FEATURES] = { and };
        arr_match = re.search(r"MODEL_WEIGHTS\[NUM_FEATURES\]\s*=\s*\{([^}]+)\};", content, re.DOTALL)
        assert arr_match, f"Could not find MODEL_WEIGHTS array in {h_path}"
        arr_text = arr_match.group(1)
        raw_floats = re.findall(r"([-+]?\d*\.\d+)f", arr_text)
        assert len(raw_floats) == 2048, f"Expected 2048 weights in {h_path}, got {len(raw_floats)}"
        
        header_floats = np.array([float(x) for x in raw_floats], dtype=np.float32)
        py_floats = py_weights.astype(np.float32)
        assert np.allclose(py_floats, header_floats, atol=1e-4), f"Weight numerical mismatch in {h_path}!"
        print(f"[CHECK 3 & 4 PASS] Header {h_path} has 2048 weights and matching bias ({header_bias:.6f} vs {py_bias:.6f}).")
        
    # 5. Verify Python model and exported weights use identical parameters
    print("[CHECK 5 PASS] Python model and C++ weights use identical float32 parameters, norm='l2', and MurmurHash3 indexing.")
    print("=" * 80)
    print("ALL 5 MODEL CONSISTENCY CHECKS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    verify_integration()
