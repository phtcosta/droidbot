import json
import shutil
# from ollama import Client
from droidbot.device import Device
from droidbot.app import App
from droidbot.device_state import DeviceState
import os

from droidbot.rvandroid_policy import RVAndroidPolicy

# https://github.com/ollama/ollama-python

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
        device.install_app(app)
        device.start_app(app)
        while (True):
            input("pressione ENTER para continuar ...")
            state = device.get_current_state()
            print(f"state_str={state.state_str}")
            print(f"structure_str={state.structure_str}")
            event = policy.generate_event()
            print(event)

            # prefix = f"{cont:03}"
            # cont += 1
            #
            # message = create_message(state)
            # state_file = os.path.join(out_dir, prefix + ".state")
            # with open(state_file, "w") as arquivo:
            #     json.dump(message, arquivo, indent=3)
            #
            # img_file = os.path.join(out_dir, prefix + ".png")
            # img_path = device.take_screenshot()
            # print(f"img_path={img_path}")
            # shutil.move(img_path, img_file)
    except KeyboardInterrupt:
        # self.logger.info("Keyboard interrupt.")
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        # self.stop()
        # sys.exit(-1)
        #
        # self.stop()
        # self.logger.info("DroidBot Stopped")

    device.disconnect()


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
