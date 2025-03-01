# droidbot/policy/rvandroid_policy.py

import logging
import json
import requests
import time
from typing import Optional, Dict, List, Any

from .input_policy import UtgBasedInputPolicy, UtgGreedySearchPolicy, POLICY_GREEDY_DFS
from droidbot.input_event import KeyEvent, IntentEvent, TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent
from droidbot.utg import UTG


class RVAndroidPolicy(UtgBasedInputPolicy):
    """
    A DroidBot input policy that communicates with RV-Android server
    to get next actions based on current app state.
    """

    # def generate_event_based_on_utg(self):
    #     pass

    def __init__(self, device, app, random_input, server_url="http://localhost:5000/api/get_actions"):
        super(RVAndroidPolicy, self).__init__(device, app, random_input)
        print("****************************************** RVAndroidPolicy")
        self.logger = logging.getLogger('RVAndroidPolicy')
        self.server_url = server_url
        self.fallback_policy = UtgGreedySearchPolicy(device, app, random_input, POLICY_GREEDY_DFS)
        self.current_state = None
        self.action_history = []
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3

        self.logger.info(f"RVAndroidPolicy initialized with server: {server_url}")

    def generate_event(self):
        """
        Generate the next input event based on suggestions from RV-Android server
        :return: The next input event
        """
        # Get current device state
        current_state = self.current_state
        if current_state is None:
            current_state = self.device.get_current_state()
            self.current_state = current_state

        # Check if the app is still running
        if not self.device.is_foreground(self.app):
            return KeyEvent(name="BACK")

        # Prepare state data for the server
        state_data = self._prepare_state_data(current_state)

        try:
            # Request actions from RV-Android server
            actions = self._get_actions_from_server(state_data)
            if actions and len(actions) > 0:
                self.consecutive_errors = 0
                return self._convert_to_droidbot_event(actions[0])

            # If no valid actions, use fallback policy
            self.logger.warning("No valid actions from server, using fallback policy")
            return self.fallback_policy.generate_event()

        except Exception as e:
            self.logger.error(f"Error getting actions from server: {e}")
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

    def _get_actions_from_server(self, state_data):
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
                timeout=30
            )
            print(f"<<<<<<<< RESPONSE: {response}")
            if response.status_code != 200:
                self.logger.error(f"Server returned status {response.status_code}: {response.text}")
                return []

            response_data = response.json()
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
        action_type = action.get("action_type", "").lower()
        target = action.get("target", "")
        params = action.get("params", {})

        # Track action for history
        action_desc = f"{action_type} on {target}"
        if len(self.action_history) >= 20:
            self.action_history.pop(0)
        self.action_history.append(action_desc)

        try:
            # Handle different action types
            if action_type == "click":
                return TouchEvent(view=target)
            elif action_type == "long_click":
                return LongTouchEvent(view=target)
            elif action_type in ["scroll_up", "scroll_down", "scroll_left", "scroll_right", "scroll"]:
                direction = action_type.replace("scroll_", "") if "_" in action_type else params.get("direction",
                                                                                                     "DOWN")
                return ScrollEvent(view=target, direction=direction.upper())
            elif action_type == "set_text":
                text = params.get("text", "")
                return SetTextEvent(view=target, text=text)
            elif action_type == "key_event":
                key_name = params.get("name", "BACK")
                return KeyEvent(name=key_name)
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return KeyEvent(name="BACK")
        except Exception as e:
            self.logger.error(f"Error converting action to event: {e}")
            return KeyEvent(name="BACK")

    def handle_utg_event(self, event):
        """Callback for UTG events"""
        self.current_state = None  # Reset current state
