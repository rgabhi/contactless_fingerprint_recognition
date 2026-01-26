import argparse
from pynsdk.media import NImage, NPixelFormat
from pynsdk.biometrics import NBiometricEngine, NBiometricOperations, NBiometricStatus, NIris, NSubject, NEImageType, BiometricAttributeId
from pynsdk.licensing import NLicense, NLicenseManager
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

    license = "IrisClient"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    engine = NBiometricEngine()
    engine.irises_liveness_confidence_threshold = 0
    nimage = NImage(args.file_name)

    iris = NIris()
    iris.image = nimage
    iris.image_type = (NEImageType.cropped_and_masked)
    subject = NSubject()
    subject.irises.add(iris)

    status =  engine.perform_operation(subject, NBiometricOperations.segment)
    # Sample returns no iris objects from provided image.
    if status == NBiometricStatus.ok:
        iris_attributes = iris.objects[0]

        print("quality: ", iris_attributes.quality)
        print("confidence: ", iris_attributes.confidence)
        print("grayscale utilisation: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.grayscale_utilisation)))
        print("interlace: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.interlace)))
        print("iris pupil border concentricity: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.iris_pupil_concentricity)))
        print("iris pupil contrast: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.iris_pupil_contrast)))
        print("iris sclera contrast: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.iris_sclera_contrast)))
        print("margin adequacy: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.margin_adequacy)))
        print("pupil boundary circularity: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.pupil_boundary_circularity)))
        print("pupil iris ratio: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.pupil_to_iris_ratio)))
        print("sharpness: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.sharpness)))
        print("usable_iris_area: ", int(iris_attributes.get_attribute_value(BiometricAttributeId.usable_iris_area)))

        seg_img = subject.irises[1].image
        np_seg_img = seg_img.to_numpy(NPixelFormat.grayscale_8u)

        cv2.imshow("segmented_image", np_seg_img)
        cv2.waitKey(-1)
    else:
        print ("Segmentation failed: " + status.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment iris from image')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)
