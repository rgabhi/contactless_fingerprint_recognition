import argparse
from nsdk.biometrics import FingerEngine
from pynsdk.biometrics import NFinger, NBiometricStatus
from pynsdk.devices import NDeviceManager, NBiometricDevice, NDeviceType
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

    license = "FingerExtractor"
    if not NLicense.obtain("/local", 5000, license):
        print(f"Failed to obtain license: {license}")
        return
    print(f"License obtained successfully: {license}")

    # Choose finger scanner
    device_manager = NDeviceManager()
    device_manager.set_device_types(NDeviceType.finger_scanner|NDeviceType.fscanner) # List only finger scanners
    print(f"Detecting devices...")
    device_manager.init_devices()
    devices_count = device_manager.get_device_count()
    print(f"Found {devices_count} device(s):")
    for i in range(devices_count):
        print(f"\t{i} - '{device_manager.get_device(i).get_display_name()}'")
    print(f"Input device's id you want to select:")
    selected_device_id = input()
    selected_device = device_manager.get_device(int(selected_device_id))
    print(f"Selected '{selected_device.get_display_name()}'")

    # Capture fingerprint image
    biometric_device = NBiometricDevice(selected_device)
    biometric = NFinger()
    print(f"Place finger on the scanner. Capturing...")
    status = biometric_device.capture(biometric, 10000)

    if status is not NBiometricStatus.ok:
        print(f"Failed capturing with '{status.name}' status. Closing...")
        exit()

    print(f"Saving image to: '{args.file_name}'")
    biometric.image.save_to_file(args.file_name)

    engine = FingerEngine()
    nimage = biometric.image
    _, finger_templates = engine.extract_finger(nimage)
    print(f"Saving template to: '{args.template_name}'")
    finger_templates[0].to_buffer().to_file(args.template_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enroll finger from scanner')
    parser.add_argument('--file_name', type=str, default="", help='file name')
    parser.add_argument('--template_name', type=str, default="", help='file name')
    args_ = parser.parse_args()
    main(args_)