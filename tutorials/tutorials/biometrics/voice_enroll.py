import argparse
from pynsdk.media import NSoundBuffer
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NVoice, NSubject, NBiometricStatus
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

    license = "VoiceExtractor"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    engine = NBiometricEngine()
    voice = NVoice()
    voice.sound_buffer = NSoundBuffer.from_file(args.file_name)
    subject = NSubject()
    subject.add(voice)    
    status = engine.perform_operation(subject, NBiometricOperations.create_template)

    if not status == NBiometricStatus.ok:
        print(f"Failed to create template reason: {status}")
        return 
    
    subject.template.to_buffer().to_file(args.template_name)
    print(f"Template succesfully created and saved: {args.template_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enroll voice from file')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    parser.add_argument('--template_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)
