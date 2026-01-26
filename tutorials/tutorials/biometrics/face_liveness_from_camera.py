import argparse
from pynsdk.biometrics import NBiometricType, NSubject, NBiometricCaptureOptions, NLivenessMode, NFace
from pynsdk.biometric_client import NBiometricClient
from pynsdk.core import NAsyncStatus
from pynsdk.devices import NCamera, NDeviceType
from pynsdk.licensing import NLicense, NLicenseManager
import cv2

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
    client.set_biometric_types(NBiometricType.face)
    client.use_device_manager(True)
    client.initialize()
    client.set_property("Faces.LivenessMode", NLivenessMode.passive)

    device_manager = client.get_device_manager()
    device_manager.set_device_types(NDeviceType.camera)
    device_manager.init_devices()
    device = device_manager.get_device(args.camera_idx)
    client.set_face_capture_device(device)

    subject = NSubject()
    face = NFace()
    face.set_capture_options(NBiometricCaptureOptions.stream | NBiometricCaptureOptions.manual)
    subject.faces.add(face)
    status = client.capture_async(subject)

    #wait 1s
    if not (NCamera(device).is_capturing()):
        cv2.waitKey(1000)

    while(status.get_status() != NAsyncStatus.completed):
        try:
            nframe = face.image
        except:
            cv2.waitKey(1000)
            continue

        frame = nframe.to_numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face.objects
        if face.objects.count > 0:
            face_attribute = face.objects[0]
            rect = face_attribute.get_rect()
            left_eye = face_attribute.get_left_eye_center_point()
            right_eye = face_attribute.get_right_eye_center_point()
            mout_center = face_attribute.get_mouth_center_point()

            cv2.rectangle(frame, (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height), (0, 255, 0), 1)
            cv2.circle(frame, (int(left_eye.x), int(left_eye.y)), 1, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_eye.x), int(right_eye.y)), 1, (0, 255, 0), -1)
            cv2.circle(frame, (int(mout_center.x), int(mout_center.y)), 1, (0, 255, 0), -1)

            cv2.putText(frame, f'Liveness: {face_attribute.get_liveness_score()}', (rect.x, rect.y), cv2.FONT_HERSHEY_DUPLEX, 0.67, (0, 0, 0), thickness=1)

            feature_points = face_attribute.feature_points
            for i in range(0, feature_points.count):
                cv2.circle(frame, (int(feature_points[i].x), int(feature_points[i].y)), 1, (0, 255, 0), -1)

        cv2.imshow('frame', frame)    
        key = cv2.waitKey(1) 
        if key == ord('q'):
            break

        if key == ord('s'):
            client.force_start()
    
    print(f"Capturing complete with status: {subject.status.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face liveness from camera sample')
    parser.add_argument('--camera_idx', type=int, default=0, help='camera index')
    args_ = parser.parse_args()
    main(args_)
