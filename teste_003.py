# test_rvandroid_enhanced.py

"""
Enhanced RVAndroid Testing Script for IDE Execution

This script provides a comprehensive testing environment for the RVAndroid system,
designed for direct execution from IDEs like PyCharm and VSCode. All configuration
parameters are hardcoded for easy modification and immediate execution.

Features:
    - Direct IDE execution without command-line arguments
    - Comprehensive server health monitoring and validation
    - Advanced DroidBot integration with the enhanced RVAndroid policy
    - Real-time performance metrics and statistics collection
    - Robust error handling and recovery mechanisms
    - Support for both single action and batch action strategies
    - Detailed logging and debugging capabilities

Architecture:
    The script implements a modular architecture with clear separation between
    configuration management, server interaction, DroidBot execution, and
    monitoring functionality. Each component includes comprehensive error
    handling and performance optimization.

Usage:
    Simply modify the configuration section below and run the script directly
    from your IDE. No command-line arguments required.

Created: 2025-06-02
Authors: RV-Android Team
Version: 2.0.0
"""

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from droidbot.device import Device
from droidbot.app import App
from droidbot.device_state import DeviceState
from droidbot.input_event import CompoundEvent

# Import the enhanced RVAndroid policy
from droidbot.rvandroid_policy_novo import RVAndroidPolicy


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS AS NEEDED
# ============================================================================

class TestConfiguration:
    """
    Centralized configuration for RVAndroid testing.
    Modify these parameters according to your testing requirements.
    """
    
    # Application and testing configuration
    APK_PATH = "/home/pedro/desenvolvimento/workspaces/workspaces-doutorado/workspace-rv/rvsec/rv-android/out/cryptoapp.apk"
    OUTPUT_DIR = "/home/pedro/tmp/rvandroid_test"
    MAX_CYCLES = 50
    CYCLE_TIMEOUT = 30
    
    # Server configuration
    SERVER_URL = "http://localhost:5000"
    SERVER_TIMEOUT = 30
    SERVER_RETRIES = 3
    
    # Policy configuration
    BATCH_ACTIONS = True  # Enable batch action support
    ACTION_DELAY = 0.3    # Delay between actions in seconds
    FALLBACK_ENABLED = True
    INCLUDE_SCREENSHOTS = True
    
    # Device configuration
    DEVICE_SERIAL = "emulator-5554"
    EMULATOR_NAME = "rvandroid_test"
    GRANT_PERMISSIONS = True
    
    # Monitoring and debugging
    DEBUG = True  # Enable debug logging
    COLLECT_METRICS = True
    SAVE_STATES = True  # Save state information for analysis
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Performance tuning
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    HEALTH_CHECK_INTERVAL = 300  # 5 minutes
    MAX_CONSECUTIVE_FAILURES = 3
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        return {
            "apk_path": cls.APK_PATH,
            "output_dir": cls.OUTPUT_DIR,
            "max_cycles": cls.MAX_CYCLES,
            "server_url": cls.SERVER_URL,
            "batch_actions": cls.BATCH_ACTIONS,
            "debug": cls.DEBUG,
            "collect_metrics": cls.COLLECT_METRICS
        }
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters."""
        if not os.path.exists(cls.APK_PATH):
            raise ValueError(f"APK file not found: {cls.APK_PATH}")
        
        if cls.MAX_CYCLES <= 0:
            raise ValueError("Max cycles must be positive")
        
        if cls.CYCLE_TIMEOUT <= 0:
            raise ValueError("Cycle timeout must be positive")
        
        if not cls.SERVER_URL.startswith(('http://', 'https://')):
            raise ValueError("Server URL must start with http:// or https://")


class ServerHealthMonitor:
    """
    Comprehensive server health monitoring and validation.
    
    This component provides robust server health checking, connectivity validation,
    and performance monitoring for the RVAndroid server connection.
    """
    
    def __init__(self, server_url: str, timeout: int = 10):
        """
        Initialize server health monitor.
        
        Args:
            server_url: RVAndroid server URL
            timeout: Health check timeout in seconds
        """
        self.server_url = server_url
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.ServerHealthMonitor")
        
        # Health tracking
        self.last_health_check = 0
        self.consecutive_failures = 0
        self.health_history = []
    
    def check_server_health(self) -> bool:
        """
        Perform comprehensive server health check.
        
        Returns:
            True if server is healthy and responsive
        """
        try:
            import requests
            
            health_url = f"{self.server_url}/health"
            self.logger.debug(f"Checking server health: {health_url}")
            
            start_time = time.time()
            response = requests.get(health_url, timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                self.logger.info(
                    f"Server health check passed (response time: {response_time*1000:.1f}ms)"
                )
                
                # Log detailed health information
                if self.logger.isEnabledFor(logging.DEBUG):
                    self._log_health_details(health_data)
                
                self.consecutive_failures = 0
                self.health_history.append({
                    'timestamp': time.time(),
                    'healthy': True,
                    'response_time': response_time
                })
                
                return True
            else:
                self.logger.warning(f"Server health check failed: HTTP {response.status_code}")
                self._record_health_failure()
                return False
                
        except Exception as e:
            self.logger.error(f"Server health check failed: {e}")
            self._record_health_failure()
            return False
    
    def wait_for_server(self, max_wait_time: int = 60) -> bool:
        """
        Wait for server to become available with retry logic.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if server becomes available
        """
        self.logger.info(f"Waiting for server to become available (max {max_wait_time}s)")
        
        start_time = time.time()
        retry_interval = 2.0
        
        while time.time() - start_time < max_wait_time:
            if self.check_server_health():
                return True
            
            remaining_time = max_wait_time - (time.time() - start_time)
            if remaining_time > 0:
                sleep_time = min(retry_interval, remaining_time)
                self.logger.info(f"Server not ready, retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                retry_interval = min(retry_interval * 1.2, 10.0)  # Exponential backoff
        
        self.logger.error(f"Server did not become available within {max_wait_time}s")
        return False
    
    def _log_health_details(self, health_data: Dict[str, Any]):
        """Log detailed server health information."""
        stats = health_data.get('statistics', {})
        components = health_data.get('components', {})
        
        self.logger.debug(f"Server uptime: {health_data.get('uptime_seconds', 0):.1f}s")
        self.logger.debug(f"Requests processed: {stats.get('requests_processed', 0)}")
        self.logger.debug(f"Success rate: {stats.get('success_rate', 0):.1%}")
        self.logger.debug(f"Components: {', '.join(k for k, v in components.items() if v == 'available')}")
    
    def _record_health_failure(self):
        """Record health check failure for tracking."""
        self.consecutive_failures += 1
        self.health_history.append({
            'timestamp': time.time(),
            'healthy': False,
            'response_time': None
        })
        
        # Keep only recent history
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-50:]


class TestingSessionManager:
    """
    Comprehensive testing session management with monitoring and control.
    
    This component manages the complete testing session lifecycle, including
    device setup, policy execution, monitoring, and cleanup operations.
    """
    
    def __init__(self):
        """Initialize testing session manager with configuration."""
        self.config = TestConfiguration
        self.logger = logging.getLogger(f"{__name__}.TestingSessionManager")
        
        # Session state
        self.device = None
        self.app = None
        self.policy = None
        self.session_start_time = None
        self.cycle_count = 0
        self.successful_cycles = 0
        
        # Monitoring data
        self.cycle_times = []
        self.error_count = 0
        self.session_metrics = {}
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self._shutdown_requested = False
    
    def run_testing_session(self) -> bool:
        """
        Execute complete testing session with comprehensive monitoring.
        
        Returns:
            True if session completed successfully
        """
        try:
            self.session_start_time = time.time()
            self.logger.info("Starting RVAndroid testing session")
            
            # Initialize testing environment
            if not self._setup_testing_environment():
                return False
            
            # Execute testing cycles
            self._execute_testing_cycles()
            
            # Generate session report
            self._generate_session_report()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in testing session: {e}", exc_info=True)
            return False
            
        finally:
            self._cleanup_testing_environment()
    
    def _setup_testing_environment(self) -> bool:
        """
        Set up complete testing environment with validation.
        
        Returns:
            True if setup completed successfully
        """
        try:
            self.logger.info("Setting up testing environment")
            
            # Create output directory
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
            # Create device instance
            self.device = self._create_device()
            if not self.device:
                return False
            
            # Create app instance
            self.app = App(self.config.APK_PATH, output_dir=self.config.OUTPUT_DIR)
            
            # Setup device connection
            self.device.set_up()
            self.device.connect()
            
            # Install and start application
            self.device.install_app(self.app)
            self.device.start_app(self.app)
            
            # Initialize enhanced RVAndroid policy
            self.policy = self._create_rvandroid_policy()
            if not self.policy:
                return False
            
            self.logger.info("Testing environment setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up testing environment: {e}", exc_info=True)
            return False
    
    def _create_device(self) -> Optional[Device]:
        """
        Create and configure DroidBot device instance.
        
        Returns:
            Configured Device instance or None if creation fails
        """
        try:
            device = Device(
                device_serial=self.config.DEVICE_SERIAL,
                is_emulator=True,
                output_dir=self.config.OUTPUT_DIR,
                cv_mode=False,
                grant_perm=self.config.GRANT_PERMISSIONS,
                enable_accessibility_hard=False,
                humanoid=None,
                ignore_ad=True
            )
            
            self.logger.debug(f"Created device instance: {self.config.DEVICE_SERIAL}")
            return device
            
        except Exception as e:
            self.logger.error(f"Error creating device: {e}")
            return None
    
    def _create_rvandroid_policy(self) -> Optional[RVAndroidPolicy]:
        """
        Create and configure enhanced RVAndroid policy.
        
        Returns:
            Configured RVAndroidPolicy instance or None if creation fails
        """
        try:
            policy_config = {
                'server_url': self.config.SERVER_URL,
                'request_timeout': self.config.SERVER_TIMEOUT,
                'max_retries': self.config.SERVER_RETRIES,
                'action_delay': self.config.ACTION_DELAY,
                'fallback_enabled': self.config.FALLBACK_ENABLED,
                'include_screenshots': self.config.INCLUDE_SCREENSHOTS,
                'batch_action_support': self.config.BATCH_ACTIONS,
                'log_level': self.config.LOG_LEVEL,
                'collect_metrics': self.config.COLLECT_METRICS,
                'health_check_interval': self.config.HEALTH_CHECK_INTERVAL,
                'max_consecutive_failures': self.config.MAX_CONSECUTIVE_FAILURES
            }
            print(f"policy_config={policy_config}")
            
            policy = RVAndroidPolicy(
                device=self.device,
                app=self.app,
                random_input=True,
                **policy_config
            )
            
            self.logger.info("Enhanced RVAndroid policy created successfully")
            return policy
            
        except Exception as e:
            self.logger.error(f"Error creating RVAndroid policy: {e}")
            return None
    
    def _execute_testing_cycles(self):
        """
        Execute testing cycles with comprehensive monitoring and control.
        
        This method implements the main testing loop with detailed monitoring,
        error handling, and performance tracking for each testing cycle.
        """
        self.logger.info(f"Starting testing cycles (max: {self.config.MAX_CYCLES})")
        
        while (self.cycle_count < self.config.MAX_CYCLES and 
               not self._shutdown_requested):
            
            cycle_start_time = time.time()
            
            try:
                self.cycle_count += 1
                self.logger.info(f"Executing cycle {self.cycle_count}/{self.config.MAX_CYCLES}")
                
                # Get current state
                current_state = self.device.get_current_state()
                if not current_state:
                    self.logger.warning("No current state available")
                    continue
                
                # Log state information
                self._log_state_information(current_state)
                
                # Generate event using RVAndroid policy
                event = self.policy.generate_event()
                if not event:
                    self.logger.warning("No event generated by policy")
                    continue
                
                # Execute event with monitoring
                self._execute_event_with_monitoring(event)
                
                # Record successful cycle
                cycle_time = time.time() - cycle_start_time
                self.cycle_times.append(cycle_time)
                self.successful_cycles += 1
                
                self.logger.debug(f"Cycle completed in {cycle_time*1000:.1f}ms")
                
                # Optional: Save state information
                if self.config.SAVE_STATES:
                    self._save_state_information(current_state, self.cycle_count)
                
                # Apply inter-cycle delay
                if self.config.ACTION_DELAY > 0:
                    time.sleep(self.config.ACTION_DELAY)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error in cycle {self.cycle_count}: {e}", exc_info=True)
                
                # Continue with next cycle after brief delay
                time.sleep(1.0)
        
        self.logger.info(f"Testing cycles completed: {self.successful_cycles}/{self.cycle_count} successful")
    
    def _execute_event_with_monitoring(self, event):
        """
        Execute DroidBot event with comprehensive monitoring and logging.
        
        Args:
            event: DroidBot event to execute
        """
        event_start_time = time.time()
        
        try:
            self.logger.debug(f"Executing event: {type(event).__name__}")
            
            # Handle compound events (batch actions)
            if isinstance(event, CompoundEvent):
                self.logger.info(f"Executing batch operation with {len(event.events)} actions")
                
                for i, sub_event in enumerate(event.events):
                    self.logger.debug(f"  Batch action {i+1}/{len(event.events)}: {type(sub_event).__name__}")
                    self.device.send_event(sub_event)
                    
                    # Brief pause between batch actions
                    if i < len(event.events) - 1:
                        time.sleep(0.5)
            else:
                # Execute single event
                self.device.send_event(event)
            
            execution_time = time.time() - event_start_time
            self.logger.debug(f"Event executed in {execution_time*1000:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"Error executing event: {e}", exc_info=True)
            raise
    
    def _log_state_information(self, state: DeviceState):
        """
        Log detailed state information for debugging and monitoring.
        
        Args:
            state: Current device state
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Current activity: {state.foreground_activity}")
            self.logger.debug(f"State ID: {getattr(state, 'state_str', 'unknown')}")
            self.logger.debug(f"View count: {len(state.views) if state.views else 0}")
        
        # Always log activity changes
        if hasattr(self, '_last_activity'):
            if self._last_activity != state.foreground_activity:
                self.logger.info(f"Activity changed: {self._last_activity} -> {state.foreground_activity}")
        
        self._last_activity = state.foreground_activity
    
    def _save_state_information(self, state: DeviceState, cycle_number: int):
        """
        Save state information to file for analysis.
        
        Args:
            state: Current device state
            cycle_number: Current cycle number
        """
        try:
            state_dir = os.path.join(self.config.OUTPUT_DIR, "states")
            os.makedirs(state_dir, exist_ok=True)
            
            state_file = os.path.join(state_dir, f"state_{cycle_number:04d}.json")
            
            state_data = {
                "cycle_number": cycle_number,
                "timestamp": time.time(),
                "activity": state.foreground_activity,
                "state_str": getattr(state, 'state_str', None),
                "view_count": len(state.views) if state.views else 0
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error saving state information: {e}")
    
    def _generate_session_report(self):
        """
        Generate comprehensive session report with performance metrics.
        
        This method creates a detailed report of the testing session including
        performance metrics, success rates, and component statistics.
        """
        try:
            session_duration = time.time() - self.session_start_time
            
            # Calculate performance metrics
            avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
            success_rate = self.successful_cycles / max(self.cycle_count, 1)
            
            # Get policy statistics if available
            policy_stats = {}
            if self.policy and hasattr(self.policy, 'get_policy_statistics'):
                try:
                    policy_stats = self.policy.get_policy_statistics()
                except Exception as e:
                    self.logger.warning(f"Error getting policy statistics: {e}")
            
            # Create comprehensive report
            report = {
                "session_summary": {
                    "duration_seconds": round(session_duration, 2),
                    "total_cycles": self.cycle_count,
                    "successful_cycles": self.successful_cycles,
                    "success_rate": round(success_rate, 3),
                    "error_count": self.error_count,
                    "avg_cycle_time_ms": round(avg_cycle_time * 1000, 2)
                },
                "configuration": self.config.to_dict(),
                "policy_statistics": policy_stats,
                "performance_metrics": {
                    "cycle_times_ms": [round(ct * 1000, 2) for ct in self.cycle_times[-10:]],
                    "min_cycle_time_ms": round(min(self.cycle_times) * 1000, 2) if self.cycle_times else 0,
                    "max_cycle_time_ms": round(max(self.cycle_times) * 1000, 2) if self.cycle_times else 0,
                    "cycles_per_minute": round(self.cycle_count / (session_duration / 60), 2) if session_duration > 0 else 0
                },
                "timestamp": time.time()
            }
            
            # Save report to file
            report_file = os.path.join(self.config.OUTPUT_DIR, "session_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Log summary
            self.logger.info("=== SESSION REPORT ===")
            self.logger.info(f"Duration: {session_duration:.1f}s")
            self.logger.info(f"Cycles: {self.successful_cycles}/{self.cycle_count} successful ({success_rate:.1%})")
            self.logger.info(f"Average cycle time: {avg_cycle_time*1000:.1f}ms")
            self.logger.info(f"Errors: {self.error_count}")
            self.logger.info(f"Report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating session report: {e}", exc_info=True)
    
    def _cleanup_testing_environment(self):
        """
        Clean up testing environment with comprehensive resource management.
        
        This method performs thorough cleanup of all testing resources including
        device disconnection, policy cleanup, and temporary file removal.
        """
        try:
            self.logger.info("Cleaning up testing environment")
            
            # Disconnect device
            if self.device:
                try:
                    self.device.disconnect()
                    self.logger.debug("Device disconnected successfully")
                except Exception as e:
                    self.logger.warning(f"Error disconnecting device: {e}")
            
            # Clean up policy resources
            if self.policy:
                try:
                    # Get final statistics before cleanup
                    if hasattr(self.policy, 'get_policy_statistics'):
                        final_stats = self.policy.get_policy_statistics()
                        self.logger.debug(f"Final policy statistics: {final_stats}")
                except Exception as e:
                    self.logger.warning(f"Error getting final policy statistics: {e}")
            
            self.logger.info("Testing environment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals for graceful termination."""
        self.logger.info(f"Received signal {signum}, requesting graceful shutdown")
        self._shutdown_requested = True


def setup_logging():
    """Configure comprehensive logging for the testing session."""
    config = TestConfiguration
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, config.LOG_LEVEL.upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File handler
    log_file = os.path.join(config.OUTPUT_DIR, 'rvandroid_test.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def main():
    """
    Main entry point for RVAndroid testing script.
    
    This function orchestrates the complete testing process including configuration
    validation, server health checking, and testing session execution with
    comprehensive error handling.
    """
    try:
        # Validate configuration
        TestConfiguration.validate()
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        logger.info("=== RVAndroid Enhanced Testing Script ===")
        logger.info(f"Configuration: {TestConfiguration.to_dict()}")
        
        # Check server health
        health_monitor = ServerHealthMonitor(TestConfiguration.SERVER_URL, TestConfiguration.SERVER_TIMEOUT)
        if not health_monitor.wait_for_server(max_wait_time=60):
            logger.error("Server is not available. Please start the RVAndroid server first.")
            return False
        
        # Run testing session
        session_manager = TestingSessionManager()
        success = session_manager.run_testing_session()
        
        if success:
            logger.info("Testing session completed successfully")
            return True
        else:
            logger.error("Testing session failed")
            return False
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return False
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    Entry point for IDE execution.
    
    To use this script:
    1. Modify the configuration parameters in the TestConfiguration class above
    2. Ensure the RVAndroid server is running
    3. Run this script directly from your IDE (PyCharm, VSCode, etc.)
    
    The script will automatically:
    - Validate configuration
    - Check server health
    - Set up the testing environment
    - Execute testing cycles
    - Generate comprehensive reports
    - Clean up resources
    """
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n=== Test execution completed successfully ===")
    print(f"Results saved to: {TestConfiguration.OUTPUT_DIR}")
    print("Check the session_report.json for detailed metrics.")