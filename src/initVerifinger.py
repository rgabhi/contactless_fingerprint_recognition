import sys
import os
import cv2
import numpy as np
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus, NFPosition
from pynsdk.licensing import NLicense, NLicenseManager

def init_sdk():
    """Initializes the SDK and returns the Engine object."""
    # 1. Initialize License
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    
    print("Initializing VeriFinger SDK License...")
    license_name = "FingerClient"
    if not NLicense.obtain("/local", 5000, license_name):
        print(f"Failed to obtain license: {license_name}")
        # Fallback to FingerExtractor if Client not available (often bundle)
        if not NLicense.obtain("/local", 5000, "FingerExtractor"):
            sys.exit(1)

    # 2. Setup the Biometric Engine
    engine = NBiometricEngine()
    # We do NOT set fingers_return_binarized_image = True because we want grayscale for Step 3
    engine.fingers_return_binarized_image = False
    return engine


if __name__ == "__main__":
    eng = init_sdk()
    try:
        process_roi(eng, args_.input, args_.output)
    finally:
        # CRITICAL FIX: Explicitly delete the engine before script exit
        # This prevents the 'invalid_operation' error during garbage collection
        del eng 