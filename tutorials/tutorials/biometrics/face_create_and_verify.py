import argparse
from nsdk.media import NImage, NPixelFormat
from nsdk.biometrics import FaceEngine, NBiometricOperations 
from nsdk.licensing import NLicense, NLicenseManager
import cv2

def main(args):
    if not args.file_name:
        print ("File name is not provided")
        return 

    ##=========================================================================
    ## TRIAL MODE
    ##=========================================================================
    ## By default trial is disabled - you have to explicitly set it to enable it.

    is_trial_mode = True
    NLicenseManager.set_trial_mode(is_trial_mode)
    print(f"Trial mode: {is_trial_mode}")

    ##=========================================================================

    licenses = "FaceMatcher,FaceClient,FaceExtractor"
    failed = False
    for license in licenses.split(','):
        if not NLicense.obtain("/local", 5000, license):
            print(f"Failed to obtain license: {license}")
            failed = True
        else:
            print(f"License obtained successfully: {license}")

    if failed:
        return
    
    engine = FaceEngine()
    engine.biometric_engine.faces_minimal_iod = 8
    engine.biometric_engine.faces_confidence_threshold = 1
    engine.biometric_engine.faces_detect_properties = True
    engine.biometric_engine.faces_detect_feature_points = True
    engine.biometric_engine.faces_quality_threshold = 0
    engine.biometric_engine.matching_threshold = 0
    print('faces confidence threshold:', engine.biometric_engine.faces_confidence_threshold)
    print('faces detect properties:', engine.biometric_engine.faces_detect_properties)
    print('faces detect feature points:', engine.biometric_engine.faces_detect_feature_points)
    print('faces quality threshold:', engine.biometric_engine.faces_quality_threshold)
    print('matching threshold:', engine.biometric_engine.matching_threshold)

    print(f'{args.file_name}:')
    image = NImage(args.file_name)
    faces, templates = engine.detect_faces(image, operation=NBiometricOperations.create_template)
    img = image.to_numpy(NPixelFormat.rgb_8u)
    for fi, face in enumerate(faces):
        #print('\t', face)
        rect = face.get_rect()
        conf = face.confidence
        cv2.rectangle(img, (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height), (0, 255, 0), 3)
        cv2.putText(img, f'{fi}: {conf}', (rect.x, rect.y), cv2.FONT_HERSHEY_DUPLEX, 0.67, (0, 0, 0), thickness=3)
        cv2.putText(img, f'{fi}: {conf}', (rect.x, rect.y), cv2.FONT_HERSHEY_DUPLEX, 0.67, (255, 255, 255), thickness=1)
        points = face.feature_points
        for point in points:
            cv2.circle(img, (point.x, point.y), 3, (0, 0, 255))

        if templates is not None:
            scores = []
            for template in templates:
                scores.append(engine.match_templates(templates[fi], template))
            print(scores)
    cv2.imshow('image', img[..., ::-1])
    cv2.waitKey(-1) & 0xFF


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face create and verify sample')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)