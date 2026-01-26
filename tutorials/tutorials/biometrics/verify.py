
import argparse
from pynsdk.biometrics import NSubject, NBiometricEngine
from pynsdk.core import NBuffer
from pynsdk.licensing import NLicense, NLicenseManager

def main(args):
    if not args.probe_template:
        print ("Probe template name is not provided")
        return 
    if not args.gallery_template:
        print ("Gallery template name is not provided")
        return 

    ##=========================================================================
    ## TRIAL MODE
    ##=========================================================================
    ## By default trial is disabled - you have to explicitly set it to enable it.

    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    ##=========================================================================

    licenses = "FingerMatcher,FaceMatcher,IrisMatcher,PalmMatcher,VoiceMatcher"
    failed = False
    for license in licenses.split(','):
        if not NLicense.obtain("/local", 5000, license):
            print(f"Failed to obtain license: {license}")
            failed = True
        else:
            print(f"License obtained successfully: {license}")

    if failed:
        return
    
    engine = NBiometricEngine()
    probe_subject = NSubject()
    probe_subject.template_buffer = NBuffer.from_file(args.probe_template)
    probe_subject.id = "1"
    gallery_subject = NSubject()
    gallery_subject.template_buffer = NBuffer.from_file(args.gallery_template)
    gallery_subject.id = "2"
    verification_result = engine.verify_offline(probe_subject, gallery_subject)
    print (verification_result)
    matching_results = probe_subject.matching_results
    print ("Matching score: ", matching_results[0].score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify templates')
    parser.add_argument('--probe_template', type=str, default="", help='file name')
    parser.add_argument('--gallery_template', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)