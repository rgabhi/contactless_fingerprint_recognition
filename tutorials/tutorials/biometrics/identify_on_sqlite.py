from pynsdk.biometric_client import NBiometricClient, NClusterBiometricConnection
from pynsdk.biometrics import NBiometricOperations, NSubject, NBiometricStatus
from pynsdk.core import NBuffer
from pynsdk.licensing import NLicense, NLicenseManager
import argparse

def main(args):
    if not args.template:
        print ("Template name is not provided")
        return 
    if not args.sqlite_file_name:
        print ("SQLite db name is not provided")
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

    client = NBiometricClient()
    client.set_database_connection_to_sqlite(args.sqlite_file_name)

    template_buffer = NBuffer.from_file(args.template)
    subject = NSubject()
    subject.template_buffer = template_buffer
    task = client.create_task(subject, NBiometricOperations.identify)
    client.perform_task(task)
    status = task.get_status()
    if status != NBiometricStatus.ok:
        print (f"Enrollment was unsuccessful. Status: {status}")
    else:
        for result in subject.matching_results:
            print (f"{result.id} {result.score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Identify on sqlite')
    parser.add_argument('--template', type=str, required=True, help='file name')
    parser.add_argument('--sqlite_file_name', type=str, required=True, help='sqlite file name')
    args_ = parser.parse_args()
    main(args_)