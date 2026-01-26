import argparse
from pynsdk.media import NImage
from pynsdk.biometrics import NFace, NSubject, NBiometricEngine, NBiometricOperations, BiometricAttributeId
from pynsdk.licensing import NLicense, NLicenseManager

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

    license = "FaceClient"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    engine = NBiometricEngine()
    engine.faces_quality_threshold = 0
    engine.faces_confidence_threshold = 0
    engine.faces_check_icao_compliance = True
    subject = NSubject()
    face = NFace()
    face.image = NImage(args.file_name)
    subject.faces.add(face)

    status = engine.perform_operation(subject, NBiometricOperations.assess_quality)
    print (status)

    if face.objects.count > 0:
        attributes = face.objects[0]
        
        # get specific attributes by id
        print(f"sharpness: {attributes.get_attribute_value(BiometricAttributeId.sharpness)}")
        print(f"background uniformity: {attributes.get_attribute_value(BiometricAttributeId.background_uniformity)}")
        print(f"grayscale density: {attributes.get_attribute_value(BiometricAttributeId.grayscale_density)}")
        print(f"contrast: {attributes.get_attribute_value(BiometricAttributeId.contrast)}")
        print(f"noise: {attributes.get_attribute_value(BiometricAttributeId.noise)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face icao check sample')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)
