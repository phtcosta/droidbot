# droidbot/policy/rvandroid_policy.py

import logging
import json
import traceback
import requests
import time
from typing import Optional, Dict, List, Any

from droidbot.app import App

from .input_policy import UtgBasedInputPolicy, UtgGreedySearchPolicy, POLICY_GREEDY_DFS
from droidbot.input_event import KeyEvent, IntentEvent, TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, CompoundEvent
from droidbot.utg import UTG
from droidbot.device import Device
from droidbot.device_state import DeviceState

class RVAndroidPolicy(UtgBasedInputPolicy):
    """
    A DroidBot input policy that communicates with RV-Android server
    to get next actions based on current app state.
    """

    def __init__(self, device: Device, app: App, random_input, server_url="http://localhost:5000/api/get_actions"):
        super(RVAndroidPolicy, self).__init__(device, app, random_input)
        print("****************************************** RVAndroidPolicy")
        self.logger = logging.getLogger('RVAndroidPolicy')
        self.server_url = server_url
        self.fallback_policy = UtgGreedySearchPolicy(device, app, random_input, POLICY_GREEDY_DFS)
        self.current_state = None
        self.action_history = []
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        # TODO apenas para teste ... remover
        # Enable show touches for better visualization
        self.device.adb.shell("settings put system show_touches 1")

        self.logger.info(f"RVAndroidPolicy initialized with server: {server_url}")

    def generate_event(self):
        """
        Generate the next input event based on suggestions from RV-Android server
        :return: The next input event (can be a CompoundEvent with multiple events)
        """
        # Check if the app is still running
        self.ensure_app_is_active()
        
        # Get current device state
        current_state: DeviceState = self.device.get_current_state()
        
        # Prepare state data for the server
        state_data = self._prepare_state_data(current_state)

        try:
            # Request actions from RV-Android server
            actions = self._get_actions_from_server(state_data)
            if actions and len(actions) > 0:
                self.consecutive_errors = 0
                
                # If there are multiple actions, create a CompoundEvent
                if len(actions) > 1:
                    self.logger.info(f"Creating CompoundEvent with {len(actions)} actions")
                    events = []
                    for action in actions:
                        try:
                            event = self._convert_to_droidbot_event(action)
                            events.append(event)
                        except Exception as e:
                            self.logger.error(f"Error converting action to event: {e}")
                    
                    # Update action history with compound action
                    action_desc = f"Compound action with {len(events)} events"
                    self._update_action_history(action_desc)
                    
                    return CompoundEvent(events=events)
                else:
                    # Handle single action case
                    event = self._convert_to_droidbot_event(actions[0])
                    return event

            # If no valid actions, use fallback policy
            self.logger.warning("No valid actions from server, using fallback policy")
            return self.fallback_policy.generate_event()

        except Exception as e:
            self.logger.error(f"Error getting actions from server: {e}")
            traceback.print_exc()
            self.consecutive_errors += 1

            # If too many consecutive errors, use fallback policy
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.logger.warning(f"Too many consecutive errors ({self.consecutive_errors}), using fallback policy")
                return self.fallback_policy.generate_event()

            # Simple retry with backoff
            time.sleep(1)
            return KeyEvent(name="BACK")

    def _prepare_state_data(self, state):
        """
        Convert DroidBot state to the format expected by RV-Android
        :param state: DroidBot state
        :return: State data dictionary for RV-Android
        """
        # Convert state to dictionary representation
        state_dict = state.to_dict()

        # Add action history
        state_dict["action_history"] = self.action_history[-10:] if self.action_history else []

        return state_dict

    def _get_actions_from_server(self, state_data, timeout=30):
        """
        Send state data to RV-Android server and get suggested actions
        :param state_data: State data for RV-Android
        :return: List of actions from RV-Android
        """
        try:
            self.logger.info(f"Sending state to server: {self.server_url}")
            response = requests.post(
                self.server_url,
                json=state_data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            print(f"<<<<<<<< RESPONSE: {response}")
            if response.status_code != 200:
                self.logger.error(f"Server returned status {response.status_code}: {response.text}")
                return []

            response_data = response.json()
            print(f"<<<<<<<< RESPONSE DATA: {response_data}")
            if "actions" not in response_data:
                self.logger.error(f"Invalid response format from server: {response_data}")
                return []

            return response_data["actions"]

        except requests.RequestException as e:
            self.logger.error(f"Failed to communicate with server: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse server response: {e}")
            return []

    def _convert_to_droidbot_event(self, action):
        """
        Convert RV-Android action to DroidBot event
        :param action: Action from RV-Android
        :return: DroidBot input event
        """
        self.logger.info(f"Converting to droidbot event: {action}")
        
        action_type = action.get("action_type", "").lower()
        target = action.get("target", "")
        params = action.get("params", {})
        coordinates = action.get("coordinates", None)
        
        print(f"*** action={action}")
        print(f"action_type={action_type}")    
        print(f"target={target}")  
        print(f"params={params}")
        print(f"coordinates={coordinates}")

        # Extract coordinates directly from the action if available
        x, y = None, None
        if coordinates and isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
            x, y = coordinates[0], coordinates[1]
        
        # Also check if target is in "x y" format (as a fallback)
        elif isinstance(target, str) and " " in target:
            parts = target.split()
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                x, y = int(parts[0]), int(parts[1])
        
        try:
            # For key events, não precisamos de coordenadas
            if action_type == "key_event":
                key_name = params.get("name", "BACK")
                return KeyEvent(name=key_name)
                
            # Se temos coordenadas, usá-las diretamente
            if x is not None and y is not None:
                if action_type == "click":
                    return TouchEvent(x=x, y=y)
                elif action_type == "long_click":
                    return LongTouchEvent(x=x, y=y)
                elif action_type in ["scroll_up", "scroll_down", "scroll_left", "scroll_right", "scroll"]:
                    direction = action_type.replace("scroll_", "") if "_" in action_type else params.get("direction", "DOWN")
                    return ScrollEvent(x=x, y=y, direction=direction.upper())
                elif action_type == "set_text":
                    text = params.get("text", "")
                    return SetTextEvent(x=x, y=y, text=text)
            
            # Se temos um resource ID (mas sem coordenadas), buscar a view
            elif ":" in target and not target.isdigit():
                # Look up the view in the current state
                current_state = self.device.get_current_state()
                view = self._find_view_by_resource_id(current_state, target)
                
                if view:
                    if action_type == "click":
                        return TouchEvent(view=view)
                    elif action_type == "long_click":
                        return LongTouchEvent(view=view)
                    elif action_type in ["scroll_up", "scroll_down", "scroll_left", "scroll_right", "scroll"]:
                        direction = action_type.replace("scroll_", "") if "_" in action_type else params.get("direction", "DOWN")
                        return ScrollEvent(view=view, direction=direction.upper())
                    elif action_type == "set_text":
                        text = params.get("text", "")
                        return SetTextEvent(view=view, text=text)
                else:
                    # Se a view não foi encontrada, usar centro da tela (último recurso)
                    self.logger.warning(f"View with resource_id {target} not found, using center of screen")
                    screen_width = self.device.get_width()
                    screen_height = self.device.get_height()
                    center_x = screen_width // 2
                    center_y = screen_height // 2
                    
                    if action_type == "click":
                        return TouchEvent(x=center_x, y=center_y)
                    elif action_type == "long_click":
                        return LongTouchEvent(x=center_x, y=center_y)
                    elif action_type in ["scroll_up", "scroll_down", "scroll_left", "scroll_right", "scroll"]:
                        direction = action_type.replace("scroll_", "") if "_" in action_type else params.get("direction", "DOWN")
                        return ScrollEvent(x=center_x, y=center_y, direction=direction.upper())
            
            # Último recurso: usar centro da tela
            else:
                self.logger.warning(f"No coordinates or valid target for {action_type}, using center of screen")
                screen_width = self.device.get_width()
                screen_height = self.device.get_height()
                center_x = screen_width // 2
                center_y = screen_height // 2
                
                if action_type == "click":
                    return TouchEvent(x=center_x, y=center_y)
                elif action_type == "long_click":
                    return LongTouchEvent(x=center_x, y=center_y)
                elif action_type in ["scroll_up", "scroll_down", "scroll_left", "scroll_right", "scroll"]:
                    direction = action_type.replace("scroll_", "") if "_" in action_type else params.get("direction", "DOWN")
                    return ScrollEvent(x=center_x, y=center_y, direction=direction.upper())
                elif action_type == "set_text":
                    text = params.get("text", "")
                    return SetTextEvent(x=center_x, y=center_y, text=text)
                    
            # Se nada funcionou
            self.logger.warning(f"Could not create event for action: {action}")
            return KeyEvent(name="BACK")
        except Exception as e:
            self.logger.error(f"Error converting action to event: {e}")
            traceback.print_exc()
            return KeyEvent(name="BACK")

    def _find_view_by_resource_id(self, state, resource_id):
        """
        Find a view by resource ID in the current state
        :param state: Current device state
        :param resource_id: Resource ID to look for
        :return: View dictionary if found, None otherwise
        """
        if not state or not resource_id:
            return None
            
        for view in state.views:
            if view.get('resource_id') == resource_id:
                return view
                
        # Try with just the ID part (without package)
        if ':id/' in resource_id:
            id_part = resource_id.split(':id/')[-1]
            for view in state.views:
                if view.get('resource_id', '').endswith(id_part):
                    return view
                    
        return None
    
    def _update_action_history(self, action_desc):
        """
        Update the action history with a new action description
        :param action_desc: Description of the action
        """
        if len(self.action_history) >= 20:
            self.action_history.pop(0)
        self.action_history.append(action_desc)

    def handle_utg_event(self, event):
        """Callback for UTG events"""
        self.current_state = None  # Reset current state
        
    def ensure_app_is_active(self):
        # Get the package name of the (target) application under test
        target_package = self.app.package_name
        # Get the currently focused package
        current_state = self.device.get_current_state()
        current_package = current_state.foreground_activity.split('/')[0] if current_state.foreground_activity else None    
        # current_package = self.device.adb.shell("dumpsys window windows | grep -E 'mCurrentFocus' | cut -d'/' -f1 | cut -d' ' -f6").strip()
        # Check if it's the target application
        if current_package != target_package:
            self.logger.warning(f"Target app not in foreground. Current: {current_package}, Expected: {target_package}")
            # Restart the application
            self.device.start_app(self.app)
            time.sleep(2)  # Wait for initialization            
                