import subprocess
import sys

def run_tests():
    print("=" * 80)
    print("EDGESPAM-IOT COMPREHENSIVE FINAL REGRESSION SUITE")
    print("=" * 80)
    
    scripts = [
        ("verify_integrated_model.py", "Model Dimension & Consistency Check"),
        ("test_backend_predictions.py", "Backend Prediction & API Contract Check"),
        ("test_frontend_regression.py", "Frontend & UI Endpoint Regression Check")
    ]
    
    python_exe = sys.executable
    
    all_success = True
    for script, desc in scripts:
        print(f"\n---> Running: {desc} ({script})")
        res = subprocess.run([python_exe, script], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[PASS] {desc}")
            print(res.stdout.strip())
        else:
            print(f"[FAIL] {desc} (Exit code {res.returncode})")
            print(res.stderr.strip())
            all_success = False
            
    print("\n" + "=" * 80)
    if all_success:
        print("MASTER REGRESSION RESULT: 100% OF TESTS PASSED")
    else:
        print("MASTER REGRESSION RESULT: FAILURES DETECTED")
    print("=" * 80)
    return all_success

if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
