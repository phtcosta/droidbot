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
    
    Features:
    - Supports both single action and batch action strategies
    - Validates and optimizes batch action sequences for UI patterns
    - Reports batch execution results back to the server
    - Tracks batch execution statistics for performance monitoring
    - Handles UI pattern-specific optimizations (forms, lists, etc.)
    - Provides error recovery and reporting for batch actions
    - Maintains compatibility with existing single-action workflows
    """

    def __init__(self, device: Device, app: App, random_input, server_url="http://localhost:5000"):
        super(RVAndroidPolicy, self).__init__(device, app, random_input)
        print("****************************************** RVAndroidPolicy (with Batch Support)")
        self.logger = logging.getLogger('RVAndroidPolicy')
        self.server_url = server_url
        self.fallback_policy = UtgGreedySearchPolicy(device, app, random_input, POLICY_GREEDY_DFS)
        self.current_state = None
        self.action_history = []
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        # Initialize batch action tracking
        self.pending_batch_reports = []
        self.batch_execution_stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_batch_actions": 0,
            "successful_batch_actions": 0,
            "patterns": {}
        }
        
        # Enable show touches for better visualization
        self.device.adb.shell("settings put system show_touches 1")

        self.logger.info(f"RVAndroidPolicy initialized with server: {server_url}")
        self.logger.info("Batch action support enabled")

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
                
                # Determine if this is a batch operation or single action
                is_batch_operation = not self._check_if_single_action_strategy(actions)
                
                # For batch operations with multiple actions, create a CompoundEvent
                if is_batch_operation and len(actions) > 1:
                    self.logger.info(f"Batch operation detected with {len(actions)} actions")
                    print(f"*** EXECUTING BATCH OPERATION: {len(actions)} ACTIONS ***")
                
                # If single action strategy is detected but multiple actions were received
                elif not is_batch_operation and len(actions) > 1:
                    self.logger.info(f"Single action strategy detected but received {len(actions)} actions. Using only the first action.")
                    actions = [actions[0]]
                
                # Process multiple actions as batch if not marked as single action
                if len(actions) > 1:
                    self.logger.info(f"Creating CompoundEvent with {len(actions)} actions")
                    events = []
                    
                    # Store batch metadata for reporting results back to the server
                    batch_metadata = self._extract_batch_metadata(actions)
                    self.logger.info(f"Batch metadata: {batch_metadata}")
                    
                    # Validate batch action sequence for coherence
                    validated_actions = self._validate_batch_actions(actions)
                    if not validated_actions:
                        self.logger.warning("Batch action validation failed, using fallback policy")
                        return self.fallback_policy.generate_event()
                        
                    # Process validated actions
                    for action in validated_actions:
                        try:
                            # Add batch execution context to each action
                            if "meta" not in action:
                                action["meta"] = {}
                            action["meta"]["batch_execution"] = True
                            action["meta"]["batch_id"] = batch_metadata.get("batch_id", "unknown")
                            
                            event = self._convert_to_droidbot_event(action)
                            events.append(event)
                            # Update history for each action within compound event
                            self._update_action_history(self._get_action_description(action))
                        except Exception as e:
                            self.logger.error(f"Error converting action to event: {e}")
                            
                            # For batch actions, report the error but continue with the rest
                            if not self._report_batch_action_error(action, e, batch_metadata):
                                self.logger.warning("Failed to report batch action error to server")
                    
                    # Check if we have any events after filtering
                    if not events:
                        self.logger.warning("No valid events created from batch actions, using fallback policy")
                        return self.fallback_policy.generate_event()
                    
                    # Create compound event with all the validated events
                    compound_event = CompoundEvent(events=events)
                    
                    # Schedule batch result reporting after execution
                    self._schedule_batch_result_reporting(batch_metadata, len(events))
                    
                    return compound_event
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
        # IMPROVED DETECTION LOGIC:
        
        # FAST PATH: If only one action is provided, it's treated as a single action
        if len(actions) == 1:
            return True
            
        # FAST PATH: If multiple actions, treat as batch by default unless explicitly marked as single
        if len(actions) > 1:
            # Only consider it a single action strategy if explicitly marked as such
            single_action_count = 0
            batch_action_count = 0
            
            for action in actions:
                meta = action.get("meta", {})
                # Check for explicit batch indicators
                if (meta.get("strategy_type") in ["batch_action", "flow_based_batch", "flow_based_batch_action"] or
                    meta.get("is_batch_part", False) or
                    "batch_id" in meta):
                    batch_action_count += 1
                    print("*** BATCH ACTION INDICATOR FOUND IN ACTION METADATA ***")
                
                # Check for explicit single action indicators
                elif (meta.get("strategy_type") in ["single_action", "dspy_single_action"] or
                      action.get("single_action_mode", False)):
                    single_action_count += 1
            
            # If any actions have batch indicators, treat as batch
            if batch_action_count > 0:
                return False
                
            # If ALL actions have single action indicators, treat as single
            if single_action_count == len(actions):
                return True
                
            # Otherwise, treat multiple actions as a batch by default
            print(f"*** TREATING {len(actions)} ACTIONS AS BATCH BY DEFAULT ***")
            return False
            
        # Detailed check for single actions (usually won't reach here with improved logic)
        for action in actions:
            meta = action.get("meta", {})
            if meta.get("strategy_type") in ["single_action", "dspy_single_action"]:
                return True
                
            # Check if the explanation mentions single action mode
            explanation = action.get("explanation", "").lower()
            if "single action" in explanation or "single_action" in explanation:
                return True
        
        # Default to single action for unspecified cases
        return True

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

    def _extract_batch_metadata(self, actions):
        """
        Extract metadata from a batch of actions to track their execution
        
        :param actions: List of actions from the batch
        :return: Dictionary with batch metadata
        """
        # Create a unique batch ID if not present
        batch_id = None
        batch_type = None
        batch_pattern = None
        batch_confidence = None
        
        # Try to find batch metadata in any of the actions
        for action in actions:
            meta = action.get("meta", {})
            
            # Use the first batch_id found
            if not batch_id and meta.get("batch_id"):
                batch_id = meta.get("batch_id")
                
            # Use the first strategy_type found
            if not batch_type and meta.get("strategy_type"):
                batch_type = meta.get("strategy_type")
                
            # Use the first pattern_type found
            if not batch_pattern and meta.get("pattern_type"):
                batch_pattern = meta.get("pattern_type")
                
            # Use the first pattern_confidence found
            if batch_confidence is None and "pattern_confidence" in meta:
                batch_confidence = meta.get("pattern_confidence")
            
            # If we found all metadata, we can stop searching
            if batch_id and batch_type and batch_pattern and batch_confidence is not None:
                break
                
        # Generate a batch_id if none was found
        if not batch_id:
            import uuid
            batch_id = f"batch_{uuid.uuid4().hex[:8]}"
            
        # Determine batch type if none was found
        if not batch_type:
            batch_type = "flow_based_batch" if len(actions) > 1 else "single_action"
            
        result = {
            "batch_id": batch_id,
            "strategy_type": batch_type,
            "action_count": len(actions),
            "timestamp": time.time(),
        }
        
        # Add pattern information if available
        if batch_pattern:
            result["pattern_type"] = batch_pattern
            
        if batch_confidence is not None:
            result["pattern_confidence"] = batch_confidence
            
        return result

    def _validate_batch_actions(self, actions):
        """
        Validate a sequence of batch actions for coherence and feasibility
        
        :param actions: List of actions to validate
        :return: List of validated actions (possibly filtered or reordered)
        """
        if not actions:
            return []
            
        # Filter out actions with obvious issues
        validated_actions = []
        for i, action in enumerate(actions):
            action_type = action.get("action_type", "").lower()
            
            # Basic validation - must have an action_type
            if not action_type:
                self.logger.warning(f"Action at index {i} has no action_type, skipping")
                continue
                
            # Action type must be one of the supported types
            if action_type not in ["click", "long_click", "scroll_up", "scroll_down", 
                                 "scroll_left", "scroll_right", "scroll", "set_text", "key_event"]:
                self.logger.warning(f"Action at index {i} has unsupported action_type: {action_type}, skipping")
                continue
                
            # Additional validation for text input actions
            if action_type == "set_text":
                params = action.get("params", {})
                if "text" not in params:
                    self.logger.warning(f"Set text action at index {i} has no text parameter, skipping")
                    continue
                    
            # Add action index metadata for tracking
            if "meta" not in action:
                action["meta"] = {}
            action["meta"]["batch_index"] = i
            
            validated_actions.append(action)
            
        # Ensure the actions are in a sensible order (e.g., click before set_text)
        # This is a simple heuristic and might need to be expanded based on UI patterns
        ordered_actions = self._order_batch_actions(validated_actions)
        
        return ordered_actions
        
    def _order_batch_actions(self, actions):
        """
        Order batch actions to ensure they execute in a logical sequence
        
        :param actions: List of actions to order
        :return: Ordered list of actions
        """
        # For form patterns, typically we want:
        # 1. Set text fields first
        # 2. Then handle click actions like checkboxes
        # 3. Finally click submit/next buttons
        
        # Extract pattern type if available
        pattern_type = None
        for action in actions:
            meta = action.get("meta", {})
            if meta.get("pattern_type"):
                pattern_type = meta.get("pattern_type")
                break
                
        # Order based on pattern type
        if pattern_type == "form":
            # For forms, order: set_text, clicks, then submit button
            clicks = []
            set_texts = []
            submit_actions = []
            other_actions = []
            
            for action in actions:
                action_type = action.get("action_type", "").lower()
                explanation = action.get("explanation", "").lower()
                
                # Check if this is likely a submit button
                is_submit = any(keyword in explanation for keyword in 
                             ["submit", "login", "register", "next", "continue", "send", "save"])
                
                if action_type == "set_text":
                    set_texts.append(action)
                elif action_type == "click" and is_submit:
                    submit_actions.append(action)
                elif action_type == "click":
                    clicks.append(action)
                else:
                    other_actions.append(action)
                    
            return set_texts + clicks + other_actions + submit_actions
        
        elif pattern_type == "list":
            # For lists, keep the original order but ensure scrolls come after clicks
            clicks = []
            scrolls = []
            other_actions = []
            
            for action in actions:
                action_type = action.get("action_type", "").lower()
                
                if action_type == "click" or action_type == "long_click":
                    clicks.append(action)
                elif "scroll" in action_type:
                    scrolls.append(action)
                else:
                    other_actions.append(action)
                    
            return clicks + other_actions + scrolls
            
        # Default: preserve the original order
        return actions
    
    def _report_batch_action_error(self, action, error, batch_metadata):
        """
        Report an error with a batch action to the server
        
        :param action: The action that caused an error
        :param error: The error information
        :param batch_metadata: Batch metadata for context
        :return: True if successfully reported, False otherwise
        """
        try:
            error_url = self.server_url.replace("get_actions", "report_batch_error")
            
            # Prepare error report
            error_report = {
                "batch_id": batch_metadata.get("batch_id", "unknown"),
                "action_index": action.get("meta", {}).get("batch_index", -1),
                "action": action,
                "error_message": str(error),
                "error_type": error.__class__.__name__,
                "timestamp": time.time()
            }
            
            # Send error report to server
            response = requests.post(
                error_url,
                json=error_report,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Error reporting failed with status {response.status_code}: {response.text}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to report batch action error: {e}")
            return False
    
    def _schedule_batch_result_reporting(self, batch_metadata, executed_count):
        """
        Schedule reporting of batch execution results back to the server
        
        :param batch_metadata: Metadata about the batch
        :param executed_count: How many actions were successfully executed
        """
        batch_id = batch_metadata.get("batch_id")
        total_actions = batch_metadata.get("action_count", 0)
        strategy_type = batch_metadata.get("strategy_type", "unknown")
        pattern_type = batch_metadata.get("pattern_type", "unknown")
        
        # Store the batch information for reporting after execution
        if not hasattr(self, 'pending_batch_reports'):
            self.pending_batch_reports = []
            
        report = {
            "batch_id": batch_id,
            "total_actions": total_actions,
            "executed_actions": executed_count,
            "strategy_type": strategy_type,
            "pattern_type": pattern_type,
            "pending_report": True,
            "scheduled_time": time.time() + 1,  # Report after 1 second to allow execution
            "success_rate": executed_count / total_actions if total_actions > 0 else 0
        }
        
        self.pending_batch_reports.append(report)
        
        # Update batch execution statistics
        self.batch_execution_stats["total_batches"] += 1
        self.batch_execution_stats["total_batch_actions"] += total_actions
        self.batch_execution_stats["successful_batch_actions"] += executed_count
        
        if executed_count == total_actions:
            self.batch_execution_stats["successful_batches"] += 1
        else:
            self.batch_execution_stats["failed_batches"] += 1
            
        # Track pattern-specific statistics
        if pattern_type != "unknown":
            if pattern_type not in self.batch_execution_stats["patterns"]:
                self.batch_execution_stats["patterns"][pattern_type] = {
                    "total_batches": 0,
                    "successful_batches": 0, 
                    "total_actions": 0,
                    "successful_actions": 0
                }
                
            self.batch_execution_stats["patterns"][pattern_type]["total_batches"] += 1
            self.batch_execution_stats["patterns"][pattern_type]["total_actions"] += total_actions
            self.batch_execution_stats["patterns"][pattern_type]["successful_actions"] += executed_count
            
            if executed_count == total_actions:
                self.batch_execution_stats["patterns"][pattern_type]["successful_batches"] += 1
        
        # Log batch statistics
        self.logger.info(f"Batch execution scheduled for reporting: {batch_id}, {executed_count}/{total_actions} actions executed")
        
        # Add statistics to the report
        report["batch_stats"] = {
            "total_batches_executed": self.batch_execution_stats["total_batches"],
            "batch_success_rate": self.batch_execution_stats["successful_batches"] / self.batch_execution_stats["total_batches"] 
                                 if self.batch_execution_stats["total_batches"] > 0 else 0,
            "action_success_rate": self.batch_execution_stats["successful_batch_actions"] / self.batch_execution_stats["total_batch_actions"]
                                  if self.batch_execution_stats["total_batch_actions"] > 0 else 0
        }
        
    def _process_pending_batch_reports(self):
        """
        Process any pending batch result reports
        """
        if not hasattr(self, 'pending_batch_reports'):
            return
            
        current_time = time.time()
        reports_to_send = []
        remaining_reports = []
        
        # Find reports that are ready to send
        for report in self.pending_batch_reports:
            if current_time >= report.get("scheduled_time", 0):
                reports_to_send.append(report)
            else:
                remaining_reports.append(report)
                
        # Update the pending reports list
        self.pending_batch_reports = remaining_reports
        
        # Send reports to the server
        for report in reports_to_send:
            try:
                report_url = self.server_url.replace("get_actions", "report_batch_result")
                
                # Add current state information
                current_state = self.device.get_current_state()
                report["current_state"] = {
                    "activity": current_state.foreground_activity,
                    "view_count": len(current_state.views) if current_state.views else 0,
                    "timestamp": time.time()
                }
                
                # Send report to server
                response = requests.post(
                    report_url,
                    json=report,
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                if response.status_code != 200:
                    self.logger.warning(f"Batch result reporting failed with status {response.status_code}: {response.text}")
            except Exception as e:
                self.logger.error(f"Failed to report batch result: {e}")

    def _get_actions_from_server(self, state_data, timeout=30):
        """
        Send state data to RV-Android server and get suggested actions
        :param state_data: State data for RV-Android
        :return: List of actions from RV-Android
        """
        # First process any pending batch reports
        self._process_pending_batch_reports()
        
        try:
            self.logger.info(f"Sending state to server: {self.server_url}")
            response = requests.post(
                self.server_url+"/api/get_actions",
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
            
            # Extract and add strategy metadata to all actions
            strategy_type = response_data.get("strategy_type")
            pattern_type = response_data.get("pattern_type")
            batch_id = response_data.get("batch_id")
            
            # Print metadata for debugging
            print(f"Response metadata: strategy_type={strategy_type}, pattern_type={pattern_type}, batch_id={batch_id}")
            if len(actions) > 1:
                print(f"Multiple actions received ({len(actions)}) - potential batch operation")
            
            if strategy_type:
                for action in actions:
                    if "meta" not in action:
                        action["meta"] = {}
                    action["meta"]["strategy_type"] = strategy_type
                    
                    # Also add pattern info if available
                    if pattern_type:
                        action["meta"]["pattern_type"] = pattern_type
                        
                    # Add batch ID if available
                    if batch_id:
                        action["meta"]["batch_id"] = batch_id
            
            # IMPROVED BATCH DETECTION:
            # We will determine if this is a batch operation based on:
            # 1. The number of actions received
            # 2. Available metadata
            
            # Consider it a batch operation if ANY of these conditions are true:
            # - 2 or more actions are returned AND strategy_type doesn't explicitly indicate single action
            # - batch_id is present in the response
            # - pattern_type is present in the response
            
            is_batch_operation = (
                (len(actions) > 1 and strategy_type not in ["single_action", "dspy_single_action"]) or
                batch_id is not None or
                pattern_type is not None
            )
            
            # If we detect a batch operation, make sure all actions are properly marked
            if is_batch_operation:
                self.logger.info(f"Batch operation detected: {len(actions)} actions")
                print(f"*** BATCH OPERATION DETECTED: {len(actions)} actions ***")
                
                # Mark all actions as part of a batch
                for action in actions:
                    if "meta" not in action:
                        action["meta"] = {}
                    # Force the strategy_type to indicate batch
                    action["meta"]["strategy_type"] = "flow_based_batch_action"
                    action["meta"]["is_batch_part"] = True
                    # Add batch_id if not already present
                    if not action["meta"].get("batch_id") and batch_id:
                        action["meta"]["batch_id"] = batch_id
                
                return actions
                
            # For single action cases
            elif strategy_type in ["single_action", "dspy_single_action"] or len(actions) == 1:
                self.logger.info("Single action case detected")
                
                # If multiple actions were returned with a single action strategy,
                # only return the first action with a warning
                if len(actions) > 1:
                    self.logger.warning("Single action strategy returned multiple actions. Using only the first one.")
                    print("WARNING: Single action strategy returned multiple actions, ignoring all but the first.")
                    return [actions[0]]
                    
                return actions
            
            # Default case (if we can't determine definitively)
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
            