import argparse
from pynsdk.media import NImage
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NFinger, NSubject, NBiometricStatus
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    ##=========================================================================

    license = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    engine = NBiometricEngine()
    engine.fingers_return_binarized_image = True
    image = NImage(file_name=args.file_name)

    subject = NSubject()
    finger = NFinger()
    finger.image = image
    subject.fingers.add(finger)
    status = engine.perform_operation(subject, NBiometricOperations.create_template)

    if status is not NBiometricStatus.ok:
        print(f"Failed with '{status.name}' status. exiting...")
        exit()

    binarized_image = finger.binarized_image
    binarized_image.save_to_file(args.save_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finger image binarization')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    parser.add_argument('--save_file_name', type=str, default="", help='file name to save binarized image')
    args_ = parser.parse_args()
    main(args_)

