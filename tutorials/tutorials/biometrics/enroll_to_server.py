from pynsdk.biometric_client import NBiometricClient, NClusterBiometricConnection
from pynsdk.biometrics import NBiometricOperations, NSubject, NBiometricStatus
from pynsdk.core import NBuffer
import argparse

def main(args):
    if not args.template:
        print ("Template name is not provided")
        return 

    client = NBiometricClient()
    client.local_operations = NBiometricOperations.none

    connection = NClusterBiometricConnection.create_with_host(host=args.server_host, port=args.port, admin_port=args.admin_port)
    client.remote_connections.add(connection)

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
    parser = argparse.ArgumentParser(description='Enroll on server')
    parser.add_argument('--template', type=str, required=True, help='file name')
    parser.add_argument('--server_host', type=str, default="127.0.0.1", help='server name')
    parser.add_argument('--admin_port', type=int, default=24932, help='admin port')
    parser.add_argument('--port', type=int, default=25452, help='port')
    args_ = parser.parse_args()
    main(args_)