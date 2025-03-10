import json
import shutil
from droidbot.device import Device
from droidbot.app import App
from droidbot.device_state import DeviceState
import os

from droidbot.rvandroid_policy import RVAndroidPolicy

HOST = "http://localhost:11434"
MODEL = "llama3.2:1b"


def execute(app_path, output_dir=None):
    device = create_device()
    app = App(app_path, output_dir=output_dir)

    policy = RVAndroidPolicy(device, app, True)

    cont = 1
    try:
        device.set_up()
        device.connect()
        
        # Enable show touches for better visualization
        device.adb.shell("settings put system show_touches 1")
        print("Show touches enabled")
        
        device.install_app(app)
        device.start_app(app)
        while (True):
            input("Press ENTER to continue...")
            state = device.get_current_state()
            print(f"state_str={state.state_str}")
            print(f"structure_str={state.structure_str}")
            
            event = policy.generate_event()
            print(f"Generated event: {event}")
            # event.send(device)
            
            # Send the event using device.send_event which works with all event types
            # device.send_event(event)
            
            # Optional: Take a screenshot after execution
            # img_path = device.take_screenshot()
            # if img_path:
            #     print(f"Screenshot saved at: {img_path}")
            
            # Optional: Save state information for debugging
            # prefix = f"{cont:03}"
            # cont += 1
            # state_file = os.path.join(output_dir, prefix + ".state")
            # with open(state_file, "w") as arquivo:
            #     json.dump(create_message(state), arquivo, indent=3)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping execution.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during execution: {e}")
    finally:
        device.disconnect()
        print("Device disconnected.")


def create_device(device_serial="emulator-5554",
                  is_emulator=True,
                  output_dir="/home/pedro/tmp",
                  cv_mode=False,
                  grant_perm=True,
                  enable_accessibility_hard=False,
                  humanoid=None,
                  ignore_ad=True):
    return Device(
        device_serial=device_serial,
        is_emulator=is_emulator,
        output_dir=output_dir,
        cv_mode=cv_mode,
        grant_perm=grant_perm,
        enable_accessibility_hard=enable_accessibility_hard,
        humanoid=humanoid,
        ignore_ad=ignore_ad)


def create_message(state: DeviceState):
    message = {
        "activity": state.foreground_activity,
        "stack": state.activity_stack,
        "screen_size": {
            "width": state.width,
            "height": state.height
        },
        "view_tree": state.view_tree
    }
    return message


if __name__ == "__main__":
    base_dir = "/home/pedro/desenvolvimento/workspaces/workspaces-doutorado/workspace-rv/rvsec/rv-android/out"
    out_dir = "/home/pedro/tmp/screenshots"
    apk = "{}/cryptoapp.apk".format(base_dir)
    execute(apk, out_dir)