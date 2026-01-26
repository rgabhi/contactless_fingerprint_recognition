from pynsdk.biometric_client import NBiometricClient
from pynsdk.biometrics import NBiometricOperations, NSubject, NBiometricStatus
from pynsdk.core import NBuffer
import argparse

def main(args):
    if not args.template:
        print ("Template name is not provided")
        return 
    if not args.sqlite_file_name:
        print ("SQLite db name is not provided")
        return 

    client = NBiometricClient()
    client.set_database_connection_to_sqlite(args.sqlite_file_name)
    
    template_buffer = NBuffer.from_file(args.template)
    subject = NSubject()
    subject.template_buffer = template_buffer
    subject.id = 1
    task = client.create_task(subject, NBiometricOperations.enroll)
    client.perform_task(task)
    status = task.get_status()
    if status != NBiometricStatus.ok:
        print (f"Enrollment was unsuccessful. Status: {status}")
    else:
        print (f"Enrollment was successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enroll on sqlite')
    parser.add_argument('--template', type=str, required=True, help='file name')
    parser.add_argument('--sqlite_file_name', type=str, required=True, help='sqlite file name')
    args_ = parser.parse_args()
    main(args_)