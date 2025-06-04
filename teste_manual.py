import json
import shutil
from droidbot.device import Device
from droidbot.app import App
from droidbot.device_state import DeviceState
import os
import argparse

from droidbot.input_event import CompoundEvent
from droidbot.rvandroid_policy import RVAndroidPolicy
# from droidbot.rvandroid_policy_novo import RVAndroidPolicy

HOST = "http://192.168.0.20:11434"
MODEL = "llama3.2:1b"
DEFAULT_SERVER_URL = "http://localhost:5000"

# Parse command line arguments to enable selecting the strategy
def parse_args():
    parser = argparse.ArgumentParser(description='Test DroidBot with RVAndroid policy')
    parser.add_argument('--batch', action='store_true', help='Use batch action strategy')
    parser.add_argument('--server-url', default=DEFAULT_SERVER_URL, help='RVAndroid server URL')
    return parser.parse_args()


def execute(app_path, output_dir=None, use_batch_strategy=True, server_url=DEFAULT_SERVER_URL):
    device = create_device()
    app = App(app_path, output_dir=output_dir)

    policy = RVAndroidPolicy(device, app, True, server_url=server_url)

    try:
        device.set_up()
        device.connect()
        
        device.install_app(app)
        device.start_app(app)
        while (True):
            input("Press ENTER to continue...")
            state = device.get_current_state()            
            print(f"structure_str={state.structure_str}") # estrutura da tela
            print(f"state_str={state.state_str}") # estrutura da tela com os dados
            
            event = policy.generate_event()
            print(f"Generated event: {event}")                        
            
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
    
    # Parse command line arguments
    args = parse_args()
    
    # Before running DroidBot, make sure the RVAndroid server is running with:
    # - For single action: python teste_run_server.py --strategy single_action
    # - For batch action: python teste_run_server.py --strategy flow_based_batch_action
    
    # Run DroidBot with the appropriate strategy
    execute(apk, out_dir, use_batch_strategy=args.batch, server_url=args.server_url)