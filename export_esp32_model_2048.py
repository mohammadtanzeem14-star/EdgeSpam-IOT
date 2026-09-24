import os
import joblib
import numpy as np

def export_esp32_header():
    model_path = "model_65824_edge_optimized_candidate.joblib"
    if not os.path.exists(model_path):
        model_path = "model.joblib"
        
    print(f"Loading candidate model from: {model_path}")
    model = joblib.load(model_path)
    
    weights = model.coef_[0]
    bias = float(model.intercept_[0])
    num_features = len(weights)
    
    assert num_features == 2048, f"Expected 2048 features, got {num_features}"
    
    zero_weights = int(np.sum(weights == 0))
    active_weights = num_features - zero_weights
    sparsity_pct = (zero_weights / num_features) * 100
    
    print("=" * 80)
    print("EXPORTING ESP32 MODEL HEADER (2048 FEATURES)")
    print("=" * 80)
    print(f"Feature Dimension : {num_features}")
    print(f"Model Bias        : {bias:.9f}")
    print(f"Active Weights    : {active_weights}")
    print(f"Zero Weights      : {zero_weights} ({sparsity_pct:.2f}% sparsity)")
    print(f"Min Weight        : {np.min(weights):.9f}")
    print(f"Max Weight        : {np.max(weights):.9f}")
    
    header_content = []
    header_content.append("// Auto-generated EdgeSpam-IoT model")
    header_content.append("// Compatible with ESP32 deployment")
    header_content.append(f"// Generated from: {os.path.basename(model_path)}")
    header_content.append(f"// Total Features: {num_features} | Sparsity: {sparsity_pct:.1f}%")
    header_content.append("")
    header_content.append("#pragma once")
    header_content.append("")
    header_content.append(f"#define NUM_FEATURES {num_features}")
    header_content.append("")
    header_content.append(f"static const float MODEL_BIAS = {bias:.9f}f;")
    header_content.append("")
    header_content.append("static const float MODEL_WEIGHTS[NUM_FEATURES] = {")
    
    for i in range(0, num_features, 4):
        chunk = weights[i:i+4]
        items = []
        for j, val in enumerate(chunk):
            idx = i + j
            comma = "," if idx < (num_features - 1) else ""
            items.append(f"{val:>15.9f}f{comma}")
        header_content.append("    " + " ".join(items))
        
    header_content.append("};")
    header_content.append("")
    
    content_str = "\n".join(header_content)
    
    destinations = [
        "model_weights.h",
        os.path.join("EdgeSpam_Demo", "model_weights.h")
    ]
    
    for dest in destinations:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content_str)
        print(f"[SUCCESS] Wrote {len(content_str):,} bytes to: {dest}")
        
    # Verification pass on written file
    with open("model_weights.h", "r", encoding="utf-8") as f:
        read_back = f.read()
        
    assert f"#define NUM_FEATURES {num_features}" in read_back
    assert f"MODEL_BIAS = {bias:.9f}f;" in read_back
    
    # Count occurrences of 'f,' and 'f\n};' to verify exactly 2048 floats
    import re
    floats_found = re.findall(r"[-+]?\d*\.\d+f", read_back)
    # Exclude MODEL_BIAS (1 float)
    weight_floats_found = len(floats_found) - 1
    print(f"[VERIFY] Total weight float literals in header: {weight_floats_found}")
    assert weight_floats_found == num_features, f"Expected {num_features} weights, found {weight_floats_found}"
    
    print("=" * 80)
    print("ESP32 MODEL HEADER EXPORT COMPLETE & VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    export_esp32_header()
