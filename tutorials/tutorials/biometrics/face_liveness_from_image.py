import argparse
from pynsdk.biometrics import NSubject, NFace, NBiometricOperations, NBiometricStatus
from pynsdk.biometric_client import NBiometricClient
from pynsdk.licensing import NLicense, NLicenseManager
from nsdk.media import NImage

def main(args):
    ##=========================================================================
    ## TRIAL MODE
    ##=========================================================================
    ## By default trial is disabled - you have to explicitly set it to enable it.

    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    ##=========================================================================

    license = "FaceClient"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    client = NBiometricClient()
    client.initialize()
    client.set_property("Faces.DetectLiveness", True)  # Turns on liveness from single frame(performed on template extraction)

    subject = NSubject()
    face = NFace()
    image = NImage(args.file_name)
    face.image = image
    subject.faces.add(face)

    task = client.create_task(subject, NBiometricOperations.create_template)
    client.perform_task(task)
    status = task.get_status()

    if(status != NBiometricStatus.ok):
        print(f"Failed with status: {status.name}")
        return

    print("Success(liveness check passed).")

    if args.template_name:
        subject.template_buffer.to_file(args.template_name)
        print("Saved template.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face liveness from image sample')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    parser.add_argument('--template_name', type=str, default="", help='(optional) file name')
    args_ = parser.parse_args()
    main(args_)