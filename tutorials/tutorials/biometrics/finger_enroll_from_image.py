import argparse
from nsdk.media import NImage
from nsdk.biometrics import FingerEngine
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    if not args.file_name:
        print ("File name is not provided")
        return 
    if not args.template_name:
        print ("Template name is not provided")
        return 

    ##=========================================================================
    ## TRIAL MODE
    ##=========================================================================
    ## By default trial is disabled - you have to explicitly set it to enable it.

    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    ##=========================================================================

    license = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    engine = FingerEngine()
    nimage = NImage(args.file_name)
    _, finger_templates = engine.extract_finger(nimage)
    finger_templates[0].to_buffer().to_file(args.template_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enroll finger from image')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    parser.add_argument('--template_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)