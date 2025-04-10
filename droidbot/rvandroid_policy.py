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
                
                # Check if there's a metadata flag indicating this is a single action strategy
                is_single_action = self._check_if_single_action_strategy(actions)
                
                # If single action strategy is detected, only use the first action
                if is_single_action and len(actions) > 1:
                    self.logger.info(f"Single action strategy detected but received {len(actions)} actions. Using only the first action.")
                    actions = [actions[0]]
                
                # If there are multiple actions and not a single action strategy, create a CompoundEvent
                if len(actions) > 1:
                    self.logger.info(f"Creating CompoundEvent with {len(actions)} actions")
                    events = []
                    for action in actions:
                        try:
                            event = self._convert_to_droidbot_event(action)
                            events.append(event)
                            # Update history for each action within compound event
                            self._update_action_history(self._get_action_description(action))
                        except Exception as e:
                            self.logger.error(f"Error converting action to event: {e}")
                    
                    return CompoundEvent(events=events)
                else:
                    # Handle single action case
                    action = actions[0]
                    event = self._convert_to_droidbot_event(action)
                    # Update action history
                    self._update_action_history(self._get_action_description(action))
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

    def _check_if_single_action_strategy(self, actions):
        """
        Check if the actions indicate a single action strategy.
        This could be indicated by metadata in the action or by checking 
        specific properties in the actions.
        
        :param actions: List of actions from RV-Android
        :return: True if single action strategy is detected, False otherwise
        """
        # Check if any action has metadata indicating single action strategy
        for action in actions:
            meta = action.get("meta", {})
            if meta.get("strategy_type") in ["single_action", "dspy_single_action"]:
                return True
                
            # Also check for other indicators that might be present
            if action.get("single_action_mode", False):
                return True
                
            # Check if the explanation mentions single action mode
            explanation = action.get("explanation", "").lower()
            if "single action" in explanation or "single_action" in explanation:
                return True
        
        # Check if there's only one action and it has a strong indicator it's meant to be alone
        if len(actions) == 1 and actions[0].get("solo_action", False):
            return True
            
        return False

    def _prepare_state_data(self, state):
        """
        Convert DroidBot state to the format expected by RV-Android
        :param state: DroidBot state
        :return: State data dictionary for RV-Android
        """
        # Convert state to dictionary representation
        state_dict = state.to_dict()

        state_dict["package_name"] = state.view_tree.get("package", "")

        # Add action history
        state_dict["action_history"] = self.action_history[-20:] if self.action_history else []
        
        # Add dynamic transition tracking if applicable
        if hasattr(self, 'dynamic_transition_tracker'):
            state_dict["dynamic_transitions"] = self.dynamic_transition_tracker.to_dict()

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
            
            if response.status_code != 200:
                self.logger.error(f"Server returned status {response.status_code}: {response.text}")
                return []

            response_data = response.json()
            
            if "actions" not in response_data:
                self.logger.error(f"Invalid response format from server: {response_data}")
                return []

            # Log how many actions were received
            actions = response_data["actions"]
            self.logger.info(f"Received {len(actions)} actions from server")
            
            # Check if this is a single action strategy response
            if response_data.get("strategy_type") in ["single_action", "dspy_single_action"]:
                self.logger.info("Single action strategy detected from server metadata")
                # Mark each action with metadata indicating it's from a single action strategy
                for action in actions:
                    if "meta" not in action:
                        action["meta"] = {}
                    action["meta"]["strategy_type"] = response_data.get("strategy_type")
                
                # If more than one action is returned despite a single action strategy,
                # only return the first action with a warning
                if len(actions) > 1:
                    self.logger.warning("Single action strategy returned multiple actions. Using only the first one.")
                    return [actions[0]]

            return actions

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
        explanation = action.get("explanation", "")
        
        # Update explanation metadata with action ID if present
        action_id = action.get("action_id", None)
        if action_id:
            explanation = f"[Action ID: {action_id}] {explanation}"

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
            # For key events, we don't need coordinates
            if action_type == "key_event":
                key_name = params.get("name", "BACK")
                return KeyEvent(name=key_name)
                
            # If we have coordinates, use them directly
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
            
            # If we have a resource ID (but no coordinates), search for the view
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
                    # If view not found, use center of screen (last resort)
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
            
            # Last resort: use center of screen
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
                    
            # If nothing worked
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
    
    def _get_action_description(self, action):
        """
        Create a descriptive string for an action to store in the history
        :param action: The action to describe
        :return: String description of the action
        """
        action_type = action.get("action_type", "unknown")
        target = action.get("target", "")
        action_id = action.get("action_id", "")
        explanation = action.get("explanation", "")
        
        # Include action_id if available
        action_id_str = f" (ID: {action_id})" if action_id else ""
        
        # Generate a description based on the action type and target
        if action_type == "click":
            return f"CLICK{action_id_str} on {target} - {explanation}"
        elif action_type == "long_click":
            return f"LONG_CLICK{action_id_str} on {target} - {explanation}"
        elif "scroll" in action_type:
            direction = action_type.replace("scroll_", "").upper() if "_" in action_type else "DOWN"
            return f"SCROLL {direction}{action_id_str} on {target} - {explanation}"
        elif action_type == "set_text":
            text = action.get("params", {}).get("text", "")
            return f"SET_TEXT{action_id_str} '{text}' on {target} - {explanation}"
        elif action_type == "key_event":
            key_name = action.get("params", {}).get("name", "BACK")
            return f"KEY {key_name}{action_id_str} - {explanation}"
        else:
            return f"{action_type.upper()}{action_id_str} on {target} - {explanation}"
    
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
        # Check if it's the target application
        if current_package != target_package:
            self.logger.warning(f"Target app not in foreground. Current: {current_package}, Expected: {target_package}")
            # Restart the application
            self.device.start_app(self.app)
            time.sleep(2)  # Wait for initialization            
                
    def track_screen_transition(self, from_state, to_state, action):
        """
        Track screen transitions to build a dynamic transition graph
        :param from_state: Source state
        :param to_state: Target state after action
        :param action: Action that caused the transition
        """
        # Initialize dynamic transition tracker if not exists
        if not hasattr(self, 'dynamic_transition_tracker'):
            # We'll use a simple dictionary to track transitions
            self.dynamic_transition_tracker = {
                "activities": {},
                "transitions": []
            }
            
        # Record screen transition
        if from_state and to_state:
            from_activity = from_state.foreground_activity
            to_activity = to_state.foreground_activity
            
            # Skip if same activity (no transition)
            if from_activity == to_activity:
                return
                
            # Record transition
            transition = {
                "from": from_activity,
                "to": to_activity,
                "action": self._get_action_description(action) if isinstance(action, dict) else str(action),
                "timestamp": time.time()
            }
            
            self.dynamic_transition_tracker["transitions"].append(transition)
            
            # Update activity visit counts
            if to_activity not in self.dynamic_transition_tracker["activities"]:
                self.dynamic_transition_tracker["activities"][to_activity] = {
                    "visit_count": 0,
                    "first_visit": time.time(),
                    "tested_elements": []
                }
            
            self.dynamic_transition_tracker["activities"][to_activity]["visit_count"] += 1
            self.dynamic_transition_tracker["activities"][to_activity]["last_visit"] = time.time()
            
            self.logger.info(f"Tracked transition: {from_activity} -> {to_activity}")
            