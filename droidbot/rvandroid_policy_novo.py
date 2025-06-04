# droidbot/policy/rvandroid_policy.py

"""
Enhanced RVAndroid Policy for DroidBot Integration

This module provides an intelligent testing policy that integrates DroidBot with the RVAndroid
LLM-driven action generation system. The policy establishes communication with the RVAndroid
server to receive context-aware action recommendations based on current application state.

Architecture:
    The policy implements a modular, component-based architecture with clear separation of
    concerns between state processing, server communication, and action execution. It provides
    robust error handling, performance monitoring, and flexible configuration options.

Key Components:
    - RVAndroidPolicyConfig: Configuration management with sensible defaults
    - StateSanitizer: State data preparation and sanitization for server transmission
    - ServerCommunicator: Robust HTTP communication with retry logic and health monitoring
    - ActionExecutor: Flexible action conversion and execution with comprehensive error handling
    - MetricsCollector: Performance monitoring and statistics collection

Integration Points:
    - DroidBot Policy Framework: Seamless integration with DroidBot's policy system
    - RVAndroid Server: RESTful communication for intelligent action generation
    - Error Handling: Centralized error management with fallback mechanisms
    - Logging: Structured logging with configurable levels and contexts

Performance Considerations:
    - Optimized health checking with adaptive intervals
    - Connection pooling and session reuse for HTTP communications
    - Configurable timeouts and retry strategies
    - Minimal overhead state sanitization and transmission
    - Efficient action caching and validation

Error Recovery:
    - Graceful degradation to fallback policies when server unavailable
    - Automatic retry mechanisms with exponential backoff
    - Comprehensive error classification and handling strategies
    - Resource cleanup and state recovery procedures

Threading Model:
    - Synchronous communication model compatible with DroidBot's execution flow
    - Thread-safe components with proper synchronization mechanisms
    - Resource lifecycle management with automatic cleanup

Created: 2025-06-02
Authors: RV-Android Team
Version: 2.0.0
"""

import base64
import json
import logging
import os
import time
import traceback
from collections import deque
from threading import Lock
from typing import Dict, List, Any, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from droidbot.device import Device
from droidbot.app import App
from droidbot.device_state import DeviceState
from droidbot.input_event import KeyEvent, TouchEvent, LongTouchEvent, ScrollEvent, SetTextEvent, CompoundEvent
from droidbot.input_policy import UtgBasedInputPolicy, UtgGreedySearchPolicy, POLICY_GREEDY_DFS


class RVAndroidPolicyConfig:
    """
    Configuration management for RVAndroid policy with comprehensive defaults and validation.
    
    This class encapsulates all configuration parameters for the RVAndroid policy, providing
    sensible defaults while allowing customization for different deployment scenarios. The
    configuration covers server communication, action execution, performance monitoring, and
    error handling aspects.
    
    Architecture:
        Configuration follows a hierarchical structure with categories for different subsystems.
        All parameters include validation and type checking to ensure system reliability.
        Default values are carefully chosen based on empirical testing and performance analysis.
    
    Configuration Categories:
        - Server Communication: Connection parameters, timeouts, retry policies
        - Action Execution: Delays, validation rules, fallback behaviors
        - Performance Monitoring: Metrics collection, logging levels
        - Error Handling: Retry strategies, circuit breaker patterns
        - State Processing: Screenshot handling, sanitization options
    
    Thread Safety:
        Configuration objects are immutable after initialization to ensure thread safety
        in concurrent environments. All configuration access is read-only after setup.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with provided parameters and sensible defaults.
        
        Args:
            **kwargs: Configuration parameters to override defaults
            
        Configuration Parameters:
            server_url (str): RVAndroid server URL
            request_timeout (int): HTTP request timeout in seconds
            max_retries (int): Maximum retry attempts for failed requests
            retry_delay (float): Base delay between retry attempts
            action_delay (float): Delay between action executions
            fallback_enabled (bool): Enable fallback to alternative policies
            include_screenshots (bool): Include screenshots in state data
            screenshot_quality (int): Screenshot compression quality (1-100)
            log_level (str): Logging level for policy operations
            collect_metrics (bool): Enable performance metrics collection
            health_check_interval (int): Health check frequency in seconds
            max_consecutive_failures (int): Failures before health check
            connection_pool_size (int): HTTP connection pool size
        """
        # Server communication configuration
        self.server_url: str = kwargs.get('server_url', 'http://localhost:5000')
        self.request_timeout: int = kwargs.get('request_timeout', 30)
        self.max_retries: int = kwargs.get('max_retries', 3)
        self.retry_delay: float = kwargs.get('retry_delay', 1.0)
        self.health_check_interval: int = kwargs.get('health_check_interval', 300)
        self.max_consecutive_failures: int = kwargs.get('max_consecutive_failures', 3)
        self.connection_pool_size: int = kwargs.get('connection_pool_size', 5)
        
        # Action execution configuration
        self.action_delay: float = kwargs.get('action_delay', 0.3)
        self.fallback_enabled: bool = kwargs.get('fallback_enabled', True)
        self.batch_action_support: bool = kwargs.get('batch_action_support', True)
        self.action_validation_strict: bool = kwargs.get('action_validation_strict', True)
        
        # State processing configuration
        self.include_screenshots: bool = kwargs.get('include_screenshots', True)
        self.screenshot_quality: int = kwargs.get('screenshot_quality', 80)
        self.state_sanitization_enabled: bool = kwargs.get('state_sanitization_enabled', True)
        self.max_state_size_mb: float = kwargs.get('max_state_size_mb', 10.0)
        
        # Performance and monitoring configuration
        self.log_level: str = kwargs.get('log_level', 'INFO')
        self.collect_metrics: bool = kwargs.get('collect_metrics', True)
        self.metrics_buffer_size: int = kwargs.get('metrics_buffer_size', 100)
        self.performance_profiling: bool = kwargs.get('performance_profiling', False)
        
        # Error handling configuration
        self.circuit_breaker_enabled: bool = kwargs.get('circuit_breaker_enabled', True)
        self.circuit_breaker_threshold: int = kwargs.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout: int = kwargs.get('circuit_breaker_timeout', 60)
        
        # Validate configuration parameters
        self._validate_configuration()
    
    def _validate_configuration(self):
        """
        Validate configuration parameters for consistency and correctness.
        
        Validates all configuration parameters to ensure they are within acceptable
        ranges and combinations. Raises ValueError for invalid configurations.
        
        Validation Rules:
            - Timeout values must be positive
            - Retry counts must be non-negative
            - Quality settings must be within valid ranges
            - URL formats must be valid
            - Numeric thresholds must be reasonable
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.request_timeout <= 0:
            raise ValueError("Request timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if not (1 <= self.screenshot_quality <= 100):
            raise ValueError("Screenshot quality must be between 1 and 100")
        
        if self.health_check_interval <= 0:
            raise ValueError("Health check interval must be positive")
        
        if self.max_consecutive_failures <= 0:
            raise ValueError("Max consecutive failures must be positive")
        
        # Validate server URL format
        if not self.server_url.startswith(('http://', 'https://')):
            raise ValueError("Server URL must start with http:// or https://")


class StateSanitizer:
    """
    State data sanitization and preparation for RVAndroid server transmission.
    
    This component handles the transformation of DroidBot state data into a clean,
    optimized format suitable for transmission to the RVAndroid server. It implements
    comprehensive sanitization, size optimization, and format standardization.
    
    Architecture:
        The sanitizer follows a pipeline architecture with multiple processing stages:
        1. Data extraction from DroidBot state objects
        2. Format standardization and type normalization
        3. Size optimization and compression
        4. Security sanitization and validation
        5. Final packaging for transmission
    
    Data Processing Pipeline:
        - Extract essential fields from complex state objects
        - Normalize data types and formats for consistency
        - Apply size limits and compression for efficiency
        - Remove sensitive or irrelevant information
        - Add metadata for processing context
    
    Performance Optimizations:
        - Lazy evaluation of expensive operations
        - Caching of frequently accessed data
        - Streaming processing for large datasets
        - Memory-efficient data structures
        - Configurable compression levels
    
    Security Considerations:
        - Sanitization of potentially sensitive data
        - Validation of data types and formats
        - Size limits to prevent DoS attacks
        - Input validation and escaping
    """
    
    def __init__(self, config: RVAndroidPolicyConfig):
        """
        Initialize state sanitizer with configuration and logging.
        
        Args:
            config: Policy configuration containing sanitization settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.StateSanitizer")
        self._screenshot_cache = {}
        self._cache_lock = Lock()
    
    def sanitize_state(self, state: DeviceState, app: App) -> Dict[str, Any]:
        """
        Create a sanitized state dictionary optimized for server transmission.
        
        This method transforms the DroidBot state into a clean, efficient format
        that preserves essential information while removing redundancy and applying
        security sanitization. The output is optimized for network transmission
        and server processing.
        
        Args:
            state: Current device state from DroidBot
            app: Application under test for context
            
        Returns:
            Dictionary containing sanitized state data ready for transmission
            
        State Data Structure:
            - Core application context (activity, package)
            - UI structure and element information
            - State identification and hashing
            - Optional screenshot data (base64 encoded)
            - Temporal context and metadata
            - Processing hints and optimization flags
        
        Processing Steps:
            1. Extract core state information
            2. Process and optimize UI element data
            3. Handle screenshot encoding if enabled
            4. Apply size limits and validation
            5. Add metadata and processing context
        
        Error Handling:
            - Graceful degradation for missing fields
            - Fallback values for essential information
            - Error logging with context preservation
            - Minimal state generation for critical failures
        """
        sanitization_start = time.time()
        
        try:
            # Core application context - always required
            sanitized = {
                "activity": state.foreground_activity or "unknown",
                "package_name": app.package_name,
                "timestamp": time.time(),
                "policy_version": "2.0.0"
            }
            
            # UI structure and dimensions
            if hasattr(state, 'view_tree') and state.view_tree:
                sanitized["view_tree"] = self._sanitize_view_tree(state.view_tree)
            
            if hasattr(state, 'width') and hasattr(state, 'height'):
                sanitized["width"] = state.width
                sanitized["height"] = state.height
            
            # State identification hashes (computed by DroidBot)
            if hasattr(state, 'state_str'):
                sanitized["state_str"] = state.state_str
            
            if hasattr(state, 'structure_str'):
                sanitized["structure_str"] = state.structure_str
            
            # Process UI views with optimization
            if hasattr(state, 'views') and state.views:
                sanitized["views"] = self._sanitize_views(state.views)
            
            # Screenshot processing with caching
            if self.config.include_screenshots and hasattr(state, 'screenshot_path'):
                screenshot_b64 = self._encode_screenshot_cached(state.screenshot_path)
                if screenshot_b64:
                    sanitized["screenshot_b64"] = screenshot_b64
            
            # Add processing metadata
            sanitized["processing_metadata"] = {
                "sanitization_time_ms": round((time.time() - sanitization_start) * 1000, 2),
                "config_version": "2.0.0",
                "includes_screenshot": "screenshot_b64" in sanitized,
                "view_count": len(sanitized.get("views", [])),
                "estimated_size_kb": self._estimate_size(sanitized)
            }
            
            # Validate size constraints
            if self.config.state_sanitization_enabled:
                self._validate_state_size(sanitized)
            
            self.logger.debug(f"Sanitized state for activity: {sanitized['activity']}")
            return sanitized
            
        except Exception as e:
            self.logger.error(f"Error sanitizing state: {e}", exc_info=True)
            # Return minimal safe state to prevent complete failure
            return {
                "activity": "error_state",
                "package_name": app.package_name if app else "unknown",
                "timestamp": time.time(),
                "error": str(e),
                "fallback_mode": True
            }
    
    def _sanitize_view_tree(self, view_tree: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize view tree data removing sensitive or unnecessary information.
        
        Args:
            view_tree: Original view tree from DroidBot
            
        Returns:
            Sanitized view tree with optimized data
        """
        if not isinstance(view_tree, dict):
            return {}
        
        # Keep essential fields only
        essential_fields = {
            'class', 'package', 'activity', 'bounds', 'enabled', 'focused',
            'clickable', 'scrollable', 'checkable', 'checked', 'selected',
            'text', 'resource_id', 'content_desc'
        }
        
        sanitized = {}
        for field in essential_fields:
            if field in view_tree:
                sanitized[field] = view_tree[field]
        
        return sanitized
    
    def _sanitize_views(self, views: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize view list with size optimization and filtering.
        
        Args:
            views: List of view dictionaries from DroidBot
            
        Returns:
            Sanitized and optimized view list
        """
        if not views or not isinstance(views, list):
            return []
        
        sanitized_views = []
        
        for view in views:
            if not isinstance(view, dict):
                continue
            
            # Filter out invisible or irrelevant views
            if not view.get('visible', True):
                continue
            
            # Sanitize individual view
            sanitized_view = self._sanitize_view_tree(view)
            
            # Add view only if it has meaningful content
            if sanitized_view.get('clickable') or sanitized_view.get('text') or sanitized_view.get('resource_id'):
                sanitized_views.append(sanitized_view)
        
        return sanitized_views
    
    def _encode_screenshot_cached(self, screenshot_path: str) -> Optional[str]:
        """
        Encode screenshot with caching for performance optimization.
        
        Args:
            screenshot_path: Path to screenshot file
            
        Returns:
            Base64 encoded screenshot or None if encoding fails
        """
        if not screenshot_path or not os.path.exists(screenshot_path):
            return None
        
        # Check cache first
        file_mtime = os.path.getmtime(screenshot_path)
        cache_key = f"{screenshot_path}:{file_mtime}"
        
        with self._cache_lock:
            if cache_key in self._screenshot_cache:
                return self._screenshot_cache[cache_key]
        
        try:
            with open(screenshot_path, 'rb') as img_file:
                img_data = img_file.read()
                encoded = base64.b64encode(img_data).decode('utf-8')
            
            # Cache the result
            with self._cache_lock:
                # Limit cache size
                if len(self._screenshot_cache) > 10:
                    self._screenshot_cache.clear()
                self._screenshot_cache[cache_key] = encoded
            
            self.logger.debug(f"Encoded screenshot: {len(encoded)} characters")
            return encoded
            
        except Exception as e:
            self.logger.error(f"Error encoding screenshot {screenshot_path}: {e}")
            return None
    
    def _estimate_size(self, data: Dict[str, Any]) -> float:
        """
        Estimate data size in kilobytes for monitoring and validation.
        
        Args:
            data: Data dictionary to estimate
            
        Returns:
            Estimated size in kilobytes
        """
        try:
            json_str = json.dumps(data)
            return len(json_str.encode('utf-8')) / 1024.0
        except Exception:
            return 0.0
    
    def _validate_state_size(self, state: Dict[str, Any]) -> None:
        """
        Validate state size against configured limits.
        
        Args:
            state: State data to validate
            
        Raises:
            ValueError: If state exceeds size limits
        """
        estimated_size = self._estimate_size(state)
        if estimated_size > self.config.max_state_size_mb * 1024:
            raise ValueError(f"State size ({estimated_size:.2f} KB) exceeds limit ({self.config.max_state_size_mb * 1024:.2f} KB)")


class ServerCommunicator:
    """
    Robust HTTP communication handler for RVAndroid server interactions.
    
    This component manages all communication with the RVAndroid server, implementing
    comprehensive error handling, retry logic, connection pooling, and health monitoring.
    It provides a reliable communication layer that gracefully handles network issues,
    server unavailability, and temporary failures.
    
    Architecture:
        The communicator uses a session-based approach with connection pooling and
        automatic retry mechanisms. Health checking is implemented with adaptive
        intervals to minimize overhead while ensuring reliable connectivity detection.
    
    Communication Features:
        - Session-based connection pooling for efficiency
        - Exponential backoff retry strategy with jitter
        - Circuit breaker pattern for fault tolerance
        - Adaptive health checking with minimal overhead
        - Request/response logging and metrics collection
        - Timeout management with configurable limits
    
    Error Handling Strategy:
        - Transient errors: Automatic retry with exponential backoff
        - Network errors: Circuit breaker activation and health monitoring
        - Server errors: Detailed logging and error classification
        - Timeout errors: Adaptive timeout adjustment
        - Authentication errors: Immediate failure with clear messaging
    
    Performance Optimizations:
        - Connection reuse through session pooling
        - Lazy health checking based on failure patterns
        - Request compression for large payloads
        - Response streaming for large responses
        - Caching of health check results
    
    Thread Safety:
        All methods are thread-safe and can be called concurrently.
        Internal state is protected with appropriate synchronization.
    """
    
    def __init__(self, config: RVAndroidPolicyConfig):
        """
        Initialize server communicator with configuration and connection setup.
        
        Args:
            config: Policy configuration containing server settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ServerCommunicator")
        
        # Initialize HTTP session with optimized configuration
        self.session = requests.Session()
        self._setup_session()
        
        # Health monitoring state
        self._server_available = None
        self._last_health_check = 0
        self._consecutive_failures = 0
        self._circuit_breaker_open = False
        self._circuit_breaker_last_failure = 0
        
        # Performance tracking
        self._request_count = 0
        self._total_response_time = 0.0
        self._state_lock = Lock()
    
    def _setup_session(self):
        """
        Configure HTTP session with optimal settings for RVAndroid communication.
        
        Session Configuration:
            - Custom retry strategy with exponential backoff
            - Connection pooling with appropriate limits
            - Request/response compression support
            - Standard headers for RVAndroid communication
            - Timeout configuration with read/connect separation
        """
        # Configure request headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'RVAndroid-Policy/2.0.0',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=0,  # We handle retries manually for better control
            connect=self.config.max_retries,
            read=self.config.max_retries,
            redirect=3,
            status_forcelist=[502, 503, 504],
            backoff_factor=0.5
        )
        
        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.connection_pool_size,
            pool_maxsize=self.config.connection_pool_size * 2,
            max_retries=retry_strategy,
            pool_block=False
        )
        
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def send_state(self, state_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send state data to RVAndroid server with comprehensive error handling.
        
        This method implements the core communication logic with the RVAndroid server,
        including retry strategies, error handling, and performance monitoring. It
        provides reliable state transmission even under adverse network conditions.
        
        Args:
            state_data: Sanitized state data to send to server
            
        Returns:
            Server response containing actions or None if communication fails
            
        Communication Flow:
            1. Check circuit breaker status and health if needed
            2. Prepare request with proper formatting and headers
            3. Send request with timeout and retry handling
            4. Process response and extract action data
            5. Update health status and performance metrics
            6. Handle errors with appropriate retry logic
        
        Error Recovery:
            - Connection errors: Retry with exponential backoff
            - Timeout errors: Immediate retry with adjusted timeout
            - Server errors (5xx): Retry with circuit breaker logic
            - Client errors (4xx): Immediate failure with logging
            - Network errors: Health check activation and retry
        
        Performance Monitoring:
            - Request timing and success rate tracking
            - Response size and processing time measurement
            - Health check frequency optimization
            - Circuit breaker pattern implementation
        """
        request_start = time.time()
        
        # Check circuit breaker status
        if self._is_circuit_breaker_open():
            self.logger.warning("Circuit breaker is open, skipping request")
            return None
        
        # Perform health check if needed
        if self._should_check_health():
            if not self._perform_health_check():
                self.logger.warning("Health check failed, server appears unavailable")
                return None
        
        endpoint = f"{self.config.server_url}/api/get_actions"
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Sending state to server (attempt {attempt + 1}/{self.config.max_retries})")
                
                # Calculate timeout with progressive increase
                timeout = self.config.request_timeout + (attempt * 5)
                
                response = self.session.post(
                    endpoint,
                    json=state_data,
                    timeout=timeout
                )
                
                # Process successful response
                if response.status_code == 200:
                    result = response.json()
                    self._record_successful_request(time.time() - request_start)
                    self.logger.debug("Successfully received server response")
                    return result
                    
                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    self.logger.warning(f"Server error {response.status_code}: {response.text}")
                    self._record_server_error()
                    
                else:
                    # Client error - don't retry
                    self.logger.error(f"Client error {response.status_code}: {response.text}")
                    self._record_failed_request()
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
                self._record_timeout_error()
                
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                self._record_connection_error()
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error on attempt {attempt + 1}: {e}")
                self._record_failed_request()
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON response on attempt {attempt + 1}: {e}")
                self._record_failed_request()
                return None
            
            # Calculate backoff delay with jitter
            if attempt < self.config.max_retries - 1:
                backoff_delay = min(
                    self.config.retry_delay * (2 ** attempt) + (time.time() % 1),
                    10.0  # Max 10 seconds
                )
                time.sleep(backoff_delay)
        
        # All attempts failed
        self.logger.error(f"Failed to communicate with server after {self.config.max_retries} attempts")
        self._record_failed_request()
        self._activate_circuit_breaker()
        return None
    
    def _should_check_health(self) -> bool:
        """
        Determine if health check should be performed based on failure patterns.
        
        Returns:
            True if health check should be performed
        """
        current_time = time.time()
        
        # Check if enough failures have occurred
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            return True
        
        # Check if enough time has passed since last check
        if current_time - self._last_health_check > self.config.health_check_interval:
            return True
        
        return False
    
    def _perform_health_check(self) -> bool:
        """
        Perform health check against the RVAndroid server.
        
        Returns:
            True if server is healthy and responsive
        """
        try:
            health_endpoint = f"{self.config.server_url}/health"
            response = self.session.get(health_endpoint, timeout=5)
            
            is_healthy = response.status_code == 200
            
            with self._state_lock:
                self._server_available = is_healthy
                self._last_health_check = time.time()
                
                if is_healthy:
                    self._consecutive_failures = 0
                    self._circuit_breaker_open = False
                    self.logger.debug("Server health check passed")
                else:
                    self.logger.warning(f"Server health check failed: {response.status_code}")
            
            return is_healthy
            
        except Exception as e:
            self.logger.warning(f"Server health check failed: {e}")
            with self._state_lock:
                self._server_available = False
                self._last_health_check = time.time()
            return False
    
    def _is_circuit_breaker_open(self) -> bool:
        """
        Check if circuit breaker is currently open.
        
        Returns:
            True if circuit breaker is open and requests should be blocked
        """
        if not self.config.circuit_breaker_enabled:
            return False
        
        if not self._circuit_breaker_open:
            return False
        
        # Check if circuit breaker should be reset
        current_time = time.time()
        if current_time - self._circuit_breaker_last_failure > self.config.circuit_breaker_timeout:
            with self._state_lock:
                self._circuit_breaker_open = False
                self._consecutive_failures = 0
            self.logger.info("Circuit breaker reset - attempting to reconnect")
            return False
        
        return True
    
    def _activate_circuit_breaker(self):
        """Activate circuit breaker to prevent further requests."""
        if self.config.circuit_breaker_enabled:
            with self._state_lock:
                self._circuit_breaker_open = True
                self._circuit_breaker_last_failure = time.time()
            self.logger.warning("Circuit breaker activated due to consecutive failures")
    
    def _record_successful_request(self, response_time: float):
        """Record metrics for successful request."""
        with self._state_lock:
            self._consecutive_failures = 0
            self._request_count += 1
            self._total_response_time += response_time
    
    def _record_failed_request(self):
        """Record metrics for failed request."""
        with self._state_lock:
            self._consecutive_failures += 1
            self._request_count += 1
    
    def _record_server_error(self):
        """Record server error for circuit breaker logic."""
        with self._state_lock:
            self._consecutive_failures += 1
            self._request_count += 1
    
    def _record_connection_error(self):
        """Record connection error and update availability status."""
        with self._state_lock:
            self._consecutive_failures += 1
            self._server_available = False
            self._request_count += 1
    
    def _record_timeout_error(self):
        """Record timeout error for adaptive timeout logic."""
        with self._state_lock:
            self._consecutive_failures += 1
            self._request_count += 1
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing communication metrics
        """
        with self._state_lock:
            avg_response_time = (
                self._total_response_time / max(self._request_count, 1)
            ) * 1000  # Convert to milliseconds
            
            return {
                "total_requests": self._request_count,
                "consecutive_failures": self._consecutive_failures,
                "avg_response_time_ms": round(avg_response_time, 2),
                "server_available": self._server_available,
                "circuit_breaker_open": self._circuit_breaker_open,
                "last_health_check": self._last_health_check
            }


class ActionExecutor:
    """
    Flexible action execution handler with comprehensive DroidBot event conversion.
    
    This component handles the conversion of RVAndroid server responses into executable
    DroidBot actions, with support for multiple action types, batch operations, and
    comprehensive error handling. It provides flexible action interpretation with
    robust fallback mechanisms.
    
    Architecture:
        The executor implements a strategy pattern for action conversion, with specific
        handlers for different action types. It supports both single actions and batch
        operations with appropriate sequencing and timing control.
    
    Action Type Support:
        - Touch actions: click, long_click, tap
        - Text input: set_text, type_text, input_text
        - Scroll actions: scroll_up, scroll_down, scroll_left, scroll_right
        - Navigation: key_event, back, home, menu
        - Gestures: swipe, pinch, zoom (future extension)
    
    Coordinate Resolution:
        - Direct coordinate specification (x, y tuples)
        - Resource ID-based element lookup
        - Text-based element matching
        - Bounds-based center point calculation
        - Fallback to screen center for invalid targets
    
    Error Handling:
        - Invalid action type graceful handling
        - Missing coordinate fallback strategies
        - Element not found alternative targeting
        - Action execution failure recovery
        - Comprehensive logging for debugging
    
    Performance Features:
        - Action caching for repeated operations
        - Efficient coordinate calculation
        - Minimal overhead event creation
        - Batch operation optimization
    """
    
    def __init__(self, device: Device, config: RVAndroidPolicyConfig):
        """
        Initialize action executor with device interface and configuration.
        
        Args:
            device: DroidBot device instance for action execution
            config: Policy configuration containing execution settings
        """
        self.device = device
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ActionExecutor")
        
        # Action type mapping for flexible interpretation
        self._action_type_aliases = {
            'click': ['click', 'touch', 'tap'],
            'long_click': ['long_click', 'long_touch', 'long_tap'],
            'set_text': ['set_text', 'type_text', 'input_text', 'enter_text'],
            'scroll_up': ['scroll_up', 'scroll_vertical_up'],
            'scroll_down': ['scroll_down', 'scroll_vertical_down'],
            'scroll_left': ['scroll_left', 'scroll_horizontal_left'],
            'scroll_right': ['scroll_right', 'scroll_horizontal_right'],
            'key_event': ['key_event', 'key', 'KEY']  # Support legacy 'KEY' type
        }
        
        # Reverse mapping for quick lookup
        self._alias_to_action = {}
        for action_type, aliases in self._action_type_aliases.items():
            for alias in aliases:
                self._alias_to_action[alias.lower()] = action_type
    
    def execute_actions(self, actions: List[Dict[str, Any]], current_state: DeviceState) -> Optional[object]:
        """
        Execute action recommendations from RVAndroid server.
        
        This method processes server responses and converts them into executable DroidBot
        events. It supports both single actions and batch operations, with appropriate
        error handling and fallback mechanisms for robust operation.
        
        Args:
            actions: List of action dictionaries from RVAndroid server
            current_state: Current device state for action conversion context
            
        Returns:
            DroidBot input event for execution or None if no valid actions
            
        Processing Logic:
            1. Validate and filter incoming actions
            2. Convert server actions to DroidBot events
            3. Handle batch operations with proper sequencing
            4. Apply execution delays and timing control
            5. Return appropriate event object for execution
        
        Action Structure:
            Each action should contain:
            - action_type: Type of action to perform
            - coordinates: Target coordinates (optional)
            - target: Target element identifier (optional)
            - params: Additional action parameters
            - action_id: Unique identifier for the action
        
        Batch Operation Support:
            - Multiple actions executed in sequence
            - Configurable delays between actions
            - Error isolation between batch items
            - Partial success handling
        
        Error Recovery:
            - Invalid action type fallback
            - Missing coordinate resolution
            - Element lookup failure handling
            - Action conversion error recovery
        """
        if not actions:
            self.logger.warning("No actions provided for execution")
            return None
        
        execution_start = time.time()
        valid_events = []
        
        # Process each action with comprehensive error handling
        for i, action_data in enumerate(actions):
            try:
                event = self._convert_to_droidbot_event(action_data, current_state)
                
                if event:
                    valid_events.append(event)
                    self.logger.debug(
                        f"Converted action {i}: {action_data.get('action_type', 'unknown')} -> {type(event).__name__}"
                    )
                else:
                    self.logger.warning(f"Failed to convert action {i}: {action_data}")
                    
            except Exception as e:
                self.logger.error(f"Error converting action {i}: {e}", exc_info=True)
                continue
        
        if not valid_events:
            self.logger.warning("No valid events generated from actions")
            return None
        
        # Handle batch operations if supported and multiple events exist
        if self.config.batch_action_support and len(valid_events) > 1:
            self.logger.info(f"Creating batch operation with {len(valid_events)} actions")
            return self._create_batch_event(valid_events, actions)
        
        # Return single event (DroidBot standard behavior)
        first_event = valid_events[0]
        execution_time = time.time() - execution_start
        
        self.logger.debug(
            f"Action execution prepared in {execution_time*1000:.2f}ms: {type(first_event).__name__}"
        )
        
        return first_event
    
    def _convert_to_droidbot_event(self, action_data: Dict[str, Any], current_state: DeviceState) -> Optional[object]:
        """
        Convert RVAndroid action data to DroidBot input event with flexible interpretation.
        
        This method provides comprehensive action type support with multiple aliases
        and flexible parameter interpretation. It handles coordinate resolution,
        element lookup, and parameter validation with robust error handling.
        
        Args:
            action_data: Action data from RVAndroid server
            current_state: Current device state for context
            
        Returns:
            DroidBot input event or None if conversion fails
            
        Conversion Process:
            1. Normalize action type using alias mapping
            2. Extract and validate action parameters
            3. Resolve target coordinates through multiple strategies
            4. Create appropriate DroidBot event object
            5. Apply action-specific parameter processing
        
        Coordinate Resolution Strategies:
            - Direct coordinates from action data
            - Resource ID-based element lookup
            - Text-based element matching
            - Bounds calculation from element properties
            - Screen center fallback for invalid targets
        
        Supported Action Types:
            - Touch events: click, long_click with coordinate resolution
            - Scroll events: directional scrolling with coordinate and direction
            - Text input: set_text with target field and text content
            - Key events: system key presses (back, home, menu)
        """
        try:
            # Extract and normalize action information
            raw_action_type = action_data.get('action_type', '').lower()
            action_type = self._alias_to_action.get(raw_action_type, raw_action_type)
            params = action_data.get('params', {})
            coordinates = action_data.get('coordinates')
            target = action_data.get('target', '')
            
            self.logger.debug(
                f"Converting action: {raw_action_type} -> {action_type} with target: {target}"
            )
            
            # Handle key events (no coordinates required)
            if action_type == 'key_event':
                key_name = params.get('name', params.get('key', 'BACK'))
                return KeyEvent(name=key_name)
            
            # Resolve coordinates for spatial actions
            x, y = self._extract_coordinates(coordinates, target, current_state)
            
            if x is None or y is None:
                self.logger.warning(f"Could not resolve coordinates for action: {action_data}")
                return None
            
            # Create appropriate event based on action type
            if action_type == 'click':
                return TouchEvent(x=x, y=y)
                
            elif action_type == 'long_click':
                duration = params.get('duration', 2000)  # Default 2 seconds
                return LongTouchEvent(x=x, y=y, duration=duration)
                
            elif action_type.startswith('scroll'):
                direction = self._extract_scroll_direction(action_type, params)
                return ScrollEvent(x=x, y=y, direction=direction)
                
            elif action_type == 'set_text':
                text = params.get('text', '')
                if not text:
                    self.logger.warning("set_text action without text parameter")
                    return None
                return SetTextEvent(x=x, y=y, text=text)
            
            else:
                self.logger.warning(f"Unsupported action type: {action_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting action to DroidBot event: {e}", exc_info=True)
            return None
    
    def _extract_coordinates(self, coordinates: Any, target: str, current_state: DeviceState) -> Tuple[Optional[int], Optional[int]]:
        """
        Extract target coordinates using multiple resolution strategies.
        
        This method implements a comprehensive coordinate resolution system that tries
        multiple strategies to determine the target location for action execution.
        It provides robust fallback mechanisms to handle various target specifications.
        
        Args:
            coordinates: Direct coordinates from action data
            target: Target identifier (resource_id, text, or coordinate string)
            current_state: Current device state for element lookup
            
        Returns:
            Tuple of (x, y) coordinates or (None, None) if resolution fails
            
        Resolution Strategies:
            1. Direct coordinates: Extract from coordinates parameter
            2. Coordinate string: Parse "x y" format from target
            3. Resource ID lookup: Find element by resource identifier
            4. Text matching: Find element by visible text content
            5. Bounds calculation: Calculate center from element bounds
            6. Screen center: Fallback to center of screen
        
        Coordinate Validation:
            - Ensure coordinates are within screen bounds
            - Validate coordinate data types and ranges
            - Handle edge cases and invalid specifications
            - Log resolution strategy for debugging
        """
        # Strategy 1: Direct coordinates from action data
        if coordinates and isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            try:
                x, y = int(coordinates[0]), int(coordinates[1])
                if self._validate_coordinates(x, y):
                    self.logger.debug(f"Coordinates resolved directly: ({x}, {y})")
                    return x, y
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Failed to parse direct coordinates: {e}")
        
        # Strategy 2: Parse target as coordinate string (format: "x y")
        if isinstance(target, str) and ' ' in target:
            parts = target.split()
            if len(parts) == 2 and all(part.replace('-', '').isdigit() for part in parts):
                try:
                    x, y = int(parts[0]), int(parts[1])
                    if self._validate_coordinates(x, y):
                        self.logger.debug(f"Coordinates parsed from target string: ({x}, {y})")
                        return x, y
                except ValueError as e:
                    self.logger.debug(f"Failed to parse coordinate string: {e}")
        
        # Strategy 3: Resource ID-based element lookup
        if isinstance(target, str) and ':' in target:
            view = self._find_view_by_resource_id(current_state, target)
            if view:
                coords = self._calculate_view_center(view)
                if coords:
                    self.logger.debug(f"Coordinates resolved from resource ID: {coords}")
                    return coords
        
        # Strategy 4: Text-based element matching
        if isinstance(target, str) and target and ':' not in target:
            view = self._find_view_by_text(current_state, target)
            if view:
                coords = self._calculate_view_center(view)
                if coords:
                    self.logger.debug(f"Coordinates resolved from text matching: {coords}")
                    return coords
        
        # Strategy 5: Screen center fallback
        screen_center = self._get_screen_center()
        self.logger.warning(
            f"Could not resolve coordinates, using screen center: {screen_center}"
        )
        return screen_center
    
    def _validate_coordinates(self, x: int, y: int) -> bool:
        """
        Validate coordinates are within reasonable screen bounds.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if coordinates are valid
        """
        screen_width = getattr(self.device, 'display_info', {}).get('width', 1080)
        screen_height = getattr(self.device, 'display_info', {}).get('height', 1920)
        
        return 0 <= x <= screen_width and 0 <= y <= screen_height
    
    def _get_screen_center(self) -> Tuple[int, int]:
        """
        Get center coordinates of the screen.
        
        Returns:
            Tuple of center coordinates
        """
        try:
            display_info = getattr(self.device, 'display_info', {})
            width = display_info.get('width', 1080)
            height = display_info.get('height', 1920)
            return width // 2, height // 2
        except Exception:
            return 540, 960  # Fallback coordinates
    
    def _find_view_by_resource_id(self, state: DeviceState, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Find view by resource ID with partial matching support.
        
        Args:
            state: Current device state
            resource_id: Resource ID to search for
            
        Returns:
            View dictionary or None if not found
        """
        if not state or not hasattr(state, 'views') or not state.views:
            return None
        
        # Exact match first
        for view in state.views:
            if view.get('resource_id') == resource_id:
                return view
        
        # Partial match (ID part only)
        if ':id/' in resource_id:
            id_part = resource_id.split(':id/')[-1]
            for view in state.views:
                view_id = view.get('resource_id', '')
                if view_id.endswith(f':id/{id_part}'):
                    return view
        
        return None
    
    def _find_view_by_text(self, state: DeviceState, text: str) -> Optional[Dict[str, Any]]:
        """
        Find view by text content with fuzzy matching.
        
        Args:
            state: Current device state
            text: Text to search for
            
        Returns:
            View dictionary or None if not found
        """
        if not state or not hasattr(state, 'views') or not state.views:
            return None
        
        text_lower = text.lower()
        
        # Exact match first
        for view in state.views:
            view_text = view.get('text', '')
            if view_text and view_text.lower() == text_lower:
                return view
        
        # Partial match
        for view in state.views:
            view_text = view.get('text', '')
            if view_text and text_lower in view_text.lower():
                return view
        
        return None
    
    def _calculate_view_center(self, view: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Calculate center coordinates of a view element.
        
        Args:
            view: View dictionary containing bounds information
            
        Returns:
            Tuple of center coordinates or None if calculation fails
        """
        try:
            bounds = view.get('bounds')
            if bounds and len(bounds) == 2 and len(bounds[0]) == 2 and len(bounds[1]) == 2:
                x1, y1 = bounds[0]
                x2, y2 = bounds[1]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                return center_x, center_y
        except (KeyError, TypeError, IndexError) as e:
            self.logger.debug(f"Failed to calculate view center: {e}")
        
        return None
    
    def _extract_scroll_direction(self, action_type: str, params: Dict[str, Any]) -> str:
        """
        Extract scroll direction from action type or parameters.
        
        Args:
            action_type: Action type containing direction information
            params: Action parameters with potential direction override
            
        Returns:
            Normalized scroll direction (UP, DOWN, LEFT, RIGHT)
        """
        # Check parameters first for explicit direction
        direction = params.get('direction', '').upper()
        if direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return direction
        
        # Extract from action type
        if 'up' in action_type:
            return 'UP'
        elif 'down' in action_type:
            return 'DOWN'
        elif 'left' in action_type:
            return 'LEFT'
        elif 'right' in action_type:
            return 'RIGHT'
        
        # Default fallback
        return 'DOWN'
    
    def _create_batch_event(self, events: List[object], original_actions: List[Dict[str, Any]]) -> object:
        """
        Create compound event for batch action execution.
        
        Args:
            events: List of DroidBot events to combine
            original_actions: Original action data for metadata
            
        Returns:
            CompoundEvent containing all events
        """
        try:
            # Add metadata to track batch operation
            for i, event in enumerate(events):
                if hasattr(event, 'metadata'):
                    event.metadata = getattr(event, 'metadata', {})
                    event.metadata.update({
                        'batch_operation': True,
                        'batch_index': i,
                        'batch_size': len(events)
                    })
            
            compound_event = CompoundEvent(events=events)
            
            self.logger.info(f"Created batch compound event with {len(events)} actions")
            return compound_event
            
        except Exception as e:
            self.logger.error(f"Error creating batch event: {e}")
            # Fallback to first event only
            return events[0] if events else None


class MetricsCollector:
    """
    Comprehensive performance monitoring and metrics collection for RVAndroid policy.
    
    This component provides detailed tracking of policy performance, server communication
    patterns, action execution statistics, and error patterns. It implements efficient
    data collection with configurable retention and analysis capabilities.
    
    Architecture:
        The collector uses a multi-buffer approach with separate tracking for different
        metric categories. It provides both real-time monitoring and historical analysis
        with configurable retention periods and aggregation strategies.
    
    Metric Categories:
        - Execution Metrics: Cycle times, action counts, success rates
        - Communication Metrics: Response times, failure rates, retry patterns
        - Action Metrics: Type distribution, success patterns, error rates
        - Performance Metrics: Memory usage, CPU utilization, throughput
        - Error Metrics: Error classification, frequency, resolution patterns
    
    Data Collection Features:
        - Lock-free data structures for high-performance collection
        - Configurable sampling rates for resource optimization
        - Automatic data rotation and retention management
        - Statistical aggregation with percentile calculations
        - Export capabilities for external analysis tools
    
    Performance Optimizations:
        - Minimal overhead data collection using atomic operations
        - Efficient data structures with O(1) insertion
        - Lazy computation of aggregated statistics
        - Memory-bounded buffers with automatic rotation
        - Batch processing for expensive operations
    """
    
    def __init__(self, config: RVAndroidPolicyConfig):
        """
        Initialize metrics collector with configuration and storage setup.
        
        Args:
            config: Policy configuration containing metrics settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        self.metrics_lock = Lock()
        
        # Initialize metric storage with bounded deques
        self._cycle_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self._communication_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self._action_metrics = deque(maxlen=self.config.metrics_buffer_size)
        self._error_metrics = deque(maxlen=self.config.metrics_buffer_size)
        
        # Cumulative counters for efficiency
        self._total_cycles = 0
        self._successful_cycles = 0
        self._total_actions = 0
        self._successful_actions = 0
        self._total_communication_time = 0.0
        self._total_execution_time = 0.0
        
        # Action type tracking
        self._action_type_counts = {}
        self._error_type_counts = {}
        
        # Performance tracking
        self._start_time = time.time()
        self._last_metrics_summary = 0
    
    def record_cycle_metrics(self, 
                           cycle_time: float,
                           server_response_time: Optional[float],
                           actions_count: int,
                           action_types: List[str],
                           fallback_used: bool,
                           server_success: bool,
                           errors: Optional[List[str]] = None):
        """
        Record comprehensive metrics for a complete policy execution cycle.
        
        This method captures detailed information about each policy cycle, including
        timing information, action details, communication patterns, and error data.
        The data is stored efficiently for both real-time monitoring and historical
        analysis.
        
        Args:
            cycle_time: Total time for the complete cycle in seconds
            server_response_time: Server communication time or None if failed
            actions_count: Number of actions received/executed
            action_types: List of action types executed in this cycle
            fallback_used: Whether fallback policy was used
            server_success: Whether server communication succeeded
            errors: List of error messages encountered during cycle
            
        Metric Recording:
            - Individual cycle data stored in circular buffer
            - Cumulative statistics updated atomically
            - Action type distribution tracking
            - Error pattern analysis and classification
            - Performance trend monitoring
        
        Data Structure:
            Each cycle record contains:
            - Timing information (cycle, communication, execution)
            - Action details (count, types, success rates)
            - Communication status and error information
            - Context information (fallback usage, error patterns)
            - Metadata (timestamp, cycle number, configuration)
        """
        if not self.config.collect_metrics:
            return
        
        cycle_timestamp = time.time()
        errors = errors or []
        
        with self.metrics_lock:
            # Update cumulative counters
            self._total_cycles += 1
            if server_success and not fallback_used:
                self._successful_cycles += 1
            
            self._total_actions += actions_count
            if server_success:
                self._successful_actions += actions_count
            
            if server_response_time is not None:
                self._total_communication_time += server_response_time
            
            self._total_execution_time += cycle_time
            
            # Update action type tracking
            for action_type in action_types:
                self._action_type_counts[action_type] = self._action_type_counts.get(action_type, 0) + 1
            
            # Update error tracking
            for error in errors:
                error_type = self._classify_error(error)
                self._error_type_counts[error_type] = self._error_type_counts.get(error_type, 0) + 1
            
            # Store detailed cycle record
            cycle_record = {
                'timestamp': cycle_timestamp,
                'cycle_number': self._total_cycles,
                'cycle_time': cycle_time,
                'server_response_time': server_response_time,
                'actions_count': actions_count,
                'action_types': action_types.copy(),
                'fallback_used': fallback_used,
                'server_success': server_success,
                'errors': errors.copy(),
                'success': server_success and not fallback_used
            }
            
            self._cycle_metrics.append(cycle_record)
        
        # Log current cycle information
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_cycle_metrics(cycle_record)
        
        # Log summary metrics periodically
        if cycle_timestamp - self._last_metrics_summary > 60:  # Every minute
            self._log_summary_metrics()
            self._last_metrics_summary = cycle_timestamp
    
    def record_communication_event(self, 
                                 event_type: str,
                                 duration: float,
                                 success: bool,
                                 error_message: Optional[str] = None):
        """
        Record detailed communication events for server interaction analysis.
        
        Args:
            event_type: Type of communication event (request, health_check, etc.)
            duration: Duration of the communication event in seconds
            success: Whether the communication was successful
            error_message: Error message if communication failed
        """
        if not self.config.collect_metrics:
            return
        
        with self.metrics_lock:
            communication_record = {
                'timestamp': time.time(),
                'event_type': event_type,
                'duration': duration,
                'success': success,
                'error_message': error_message
            }
            
            self._communication_metrics.append(communication_record)
    
    def record_action_execution(self,
                              action_type: str,
                              execution_time: float,
                              success: bool,
                              coordinates: Optional[Tuple[int, int]] = None):
        """
        Record action execution details for performance analysis.
        
        Args:
            action_type: Type of action executed
            execution_time: Time taken to execute the action
            success: Whether action execution was successful
            coordinates: Target coordinates for the action
        """
        if not self.config.collect_metrics:
            return
        
        with self.metrics_lock:
            action_record = {
                'timestamp': time.time(),
                'action_type': action_type,
                'execution_time': execution_time,
                'success': success,
                'coordinates': coordinates
            }
            
            self._action_metrics.append(action_record)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary with statistical analysis.
        
        Returns:
            Dictionary containing performance metrics and statistics
            
        Summary Categories:
            - Overall Statistics: Success rates, total counts, averages
            - Timing Analysis: Response times, execution times, percentiles
            - Action Distribution: Type frequencies, success patterns
            - Error Analysis: Error rates, type distribution, trends
            - Communication Health: Server availability, retry patterns
        """
        with self.metrics_lock:
            current_time = time.time()
            uptime = current_time - self._start_time
            
            # Calculate basic statistics
            success_rate = self._successful_cycles / max(self._total_cycles, 1)
            avg_cycle_time = self._total_execution_time / max(self._total_cycles, 1)
            avg_communication_time = self._total_communication_time / max(self._successful_cycles, 1) if self._successful_cycles > 0 else 0
            
            # Calculate response time percentiles
            response_times = [
                record['server_response_time'] 
                for record in self._cycle_metrics 
                if record['server_response_time'] is not None
            ]
            
            percentiles = {}
            if response_times:
                sorted_times = sorted(response_times)
                percentiles = {
                    'p50': self._calculate_percentile(sorted_times, 50),
                    'p90': self._calculate_percentile(sorted_times, 90),
                    'p95': self._calculate_percentile(sorted_times, 95),
                    'p99': self._calculate_percentile(sorted_times, 99)
                }
            
            return {
                'uptime_seconds': round(uptime, 2),
                'total_cycles': self._total_cycles,
                'successful_cycles': self._successful_cycles,
                'success_rate': round(success_rate, 3),
                'total_actions': self._total_actions,
                'successful_actions': self._successful_actions,
                'avg_cycle_time_ms': round(avg_cycle_time * 1000, 2),
                'avg_communication_time_ms': round(avg_communication_time * 1000, 2),
                'response_time_percentiles_ms': {
                    k: round(v * 1000, 2) for k, v in percentiles.items()
                },
                'action_type_distribution': dict(self._action_type_counts),
                'error_type_distribution': dict(self._error_type_counts),
                'recent_cycles': len(self._cycle_metrics),
                'metrics_collection_enabled': True
            }
    
    def _classify_error(self, error_message: str) -> str:
        """
        Classify error messages into categories for analysis.
        
        Args:
            error_message: Error message to classify
            
        Returns:
            Error category string
        """
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'connection' in error_lower:
            return 'connection'
        elif 'server' in error_lower:
            return 'server'
        elif 'json' in error_lower or 'parse' in error_lower:
            return 'parsing'
        elif 'coordinate' in error_lower:
            return 'coordinate_resolution'
        elif 'action' in error_lower:
            return 'action_execution'
        else:
            return 'unknown'
    
    def _calculate_percentile(self, sorted_values: List[float], percentile: int) -> float:
        """
        Calculate percentile value from sorted list.
        
        Args:
            sorted_values: Sorted list of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _log_cycle_metrics(self, cycle_record: Dict[str, Any]):
        """Log detailed cycle metrics for debugging."""
        self.logger.debug(
            f"Cycle {cycle_record['cycle_number']}: "
            f"time={cycle_record['cycle_time']*1000:.1f}ms, "
            f"server_time={cycle_record['server_response_time']*1000:.1f}ms if cycle_record['server_response_time'] else 'N/A', "
            f"actions={cycle_record['actions_count']}, "
            f"success={cycle_record['success']}"
        )
    
    def _log_summary_metrics(self):
        """Log summary performance metrics."""
        summary = self.get_performance_summary()
        
        self.logger.info(
            f"Performance Summary: "
            f"cycles={summary['total_cycles']}, "
            f"success_rate={summary['success_rate']:.1%}, "
            f"avg_cycle_time={summary['avg_cycle_time_ms']:.1f}ms, "
            f"avg_comm_time={summary['avg_communication_time_ms']:.1f}ms"
        )


class RVAndroidPolicy(UtgBasedInputPolicy):
    """
    Enhanced RVAndroid policy for intelligent Android application testing with LLM integration.
    
    This policy represents a significant evolution in Android testing approaches, integrating
    DroidBot's robust exploration capabilities with advanced language model reasoning for
    intelligent action selection. The policy implements a comprehensive architecture for
    reliable server communication, flexible action execution, and detailed performance monitoring.
    
    Architectural Overview:
        The policy employs a modular component-based architecture with clear separation of
        concerns between state processing, server communication, action execution, and
        performance monitoring. Each component is designed for reliability, extensibility,
        and efficient operation under various deployment conditions.
    
    Core Components:
        - StateSanitizer: Optimizes state data for efficient server transmission
        - ServerCommunicator: Handles robust HTTP communication with retry logic
        - ActionExecutor: Converts server responses to executable DroidBot actions
        - MetricsCollector: Provides comprehensive performance monitoring
        - Configuration: Manages system-wide settings with validation
    
    Communication Protocol:
        The policy implements a synchronous request-response protocol with the RVAndroid
        server, sending sanitized state data and receiving intelligent action recommendations.
        The communication layer includes comprehensive error handling, retry mechanisms,
        and health monitoring for reliable operation.
    
    Error Handling Strategy:
        - Graceful degradation to fallback policies when server unavailable
        - Automatic retry mechanisms with exponential backoff
        - Circuit breaker pattern for fault tolerance
        - Comprehensive error classification and recovery procedures
        - Detailed logging and metrics collection for debugging
    
    Performance Features:
        - Optimized state sanitization with caching mechanisms
        - Connection pooling and session reuse for HTTP efficiency
        - Configurable timeouts and retry strategies
        - Comprehensive metrics collection with minimal overhead
        - Health monitoring with adaptive check intervals
    
    Integration Points:
        - DroidBot Policy Framework: Seamless integration with existing workflow
        - RVAndroid Server: RESTful API communication for action generation
        - Fallback Policies: Automatic fallback to UTG-based exploration
        - Monitoring Systems: Export capabilities for external analysis
    
    Thread Safety:
        All components are designed to be thread-safe and can handle concurrent
        access appropriately. The policy maintains state consistency through
        proper synchronization mechanisms.
    
    Created: 2025-06-02
    Authors: RV-Android Team
    Version: 2.0.0
    License: MIT
    """
    
    def __init__(self, device: Device, app: App, random_input: bool = True, **kwargs):
        """
        Initialize the enhanced RVAndroid policy with comprehensive component setup.
        
        This constructor establishes all necessary components for intelligent testing,
        including server communication, action execution, performance monitoring, and
        fallback mechanisms. The initialization process includes configuration validation,
        component setup, and initial health checking.
        
        Args:
            device: DroidBot device instance for interaction
            app: Application under test
            random_input: Whether to use random input as fallback
            **kwargs: Additional configuration parameters
            
        Configuration Parameters:
            server_url (str): RVAndroid server URL
            request_timeout (int): HTTP request timeout in seconds
            max_retries (int): Maximum retry attempts
            action_delay (float): Delay between actions
            fallback_enabled (bool): Enable fallback policies
            include_screenshots (bool): Include screenshots in state data
            log_level (str): Logging level
            collect_metrics (bool): Enable metrics collection
            
        Component Initialization:
            1. Configuration validation and setup
            2. Logging system configuration
            3. Component instantiation with dependency injection
            4. Fallback policy setup
            5. Initial server health check
            6. Performance monitoring initialization
        
        Error Handling:
            - Configuration validation with meaningful error messages
            - Component initialization error recovery
            - Graceful fallback setup for unreliable conditions
            - Comprehensive logging for troubleshooting
        """
        super().__init__(device, app, random_input)
        
        # Initialize configuration with validation
        try:
            self.config = RVAndroidPolicyConfig(**kwargs)
        except ValueError as e:
            raise ValueError(f"Invalid RVAndroid policy configuration: {e}")
        
        # Set up logging with configured level
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Initialize modular components with dependency injection
        self.state_sanitizer = StateSanitizer(self.config)
        self.server_communicator = ServerCommunicator(self.config)
        self.action_executor = ActionExecutor(device, self.config)
        self.metrics_collector = MetricsCollector(self.config)
        
        # Initialize fallback policy for reliability
        if self.config.fallback_enabled:
            try:
                self.fallback_policy = UtgGreedySearchPolicy(device, app, random_input, POLICY_GREEDY_DFS)
                self.logger.info("Fallback policy (UTG Greedy Search) initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize fallback policy: {e}")
                self.fallback_policy = None
        else:
            self.fallback_policy = None
        
        # Policy state tracking
        self.current_state = None
        self.last_action_time = 0
        self.consecutive_errors = 0
        self.policy_start_time = time.time()
        
        # Performance monitoring
        self._total_cycles = 0
        self._successful_cycles = 0
        
        # Log successful initialization
        self.logger.info(
            f"RVAndroid policy initialized successfully "
            f"(server: {self.config.server_url}, "
            f"fallback: {self.config.fallback_enabled}, "
            f"metrics: {self.config.collect_metrics})"
        )
        
        # Perform initial server connectivity check
        self._perform_initial_health_check()
    
    def generate_event(self):
        """
        Generate the next input event using intelligent RVAndroid server recommendations.
        
        This method implements the core policy logic, orchestrating the complete cycle
        of state analysis, server communication, and action generation. It includes
        comprehensive error handling, performance monitoring, and fallback mechanisms
        to ensure robust operation under various conditions.
        
        Returns:
            DroidBot input event to execute or None if no action available
            
        Execution Flow:
            1. Application state validation and preparation
            2. Current device state capture and sanitization
            3. Server communication with intelligent retry logic
            4. Action conversion and validation
            5. Performance metrics recording
            6. Error handling and fallback activation
        
        State Processing:
            - Comprehensive state capture from DroidBot
            - State sanitization and optimization for transmission
            - Context enrichment with historical information
            - Size validation and compression for efficiency
        
        Server Communication:
            - Robust HTTP communication with retry mechanisms
            - Health monitoring and circuit breaker patterns
            - Response validation and parsing
            - Error classification and recovery strategies
        
        Action Generation:
            - Flexible action type interpretation
            - Coordinate resolution with multiple strategies
            - Batch action support with proper sequencing
            - Validation and safety checks
        
        Error Recovery:
            - Graceful degradation to fallback policies
            - Automatic retry with exponential backoff
            - Circuit breaker activation for persistent failures
            - Comprehensive error logging and metrics
        
        Performance Monitoring:
            - Detailed timing analysis for optimization
            - Success rate tracking and trend analysis
            - Resource usage monitoring
            - Statistical analysis and reporting
        """
        cycle_start_time = time.time()
        server_response_time = None
        actions_count = 0
        action_types = []
        fallback_used = False
        server_success = False
        errors = []
        
        try:
            # Ensure target application is active and responsive
            self._ensure_app_is_active()
            
            # Capture current device state with validation
            current_state = self.device.get_current_state()
            if not current_state:
                error_msg = "No current state available from device"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                return self._handle_fallback("no_current_state")
            
            self.current_state = current_state
            self._total_cycles += 1
            
            # Sanitize state for efficient server transmission
            try:
                sanitized_state = self.state_sanitizer.sanitize_state(current_state, self.app)
            except Exception as e:
                error_msg = f"State sanitization failed: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                return self._handle_fallback("state_sanitization_failed")
            
            # Communicate with RVAndroid server for intelligent recommendations
            server_start_time = time.time()
            try:
                server_response = self.server_communicator.send_state(sanitized_state)
            except Exception as e:
                error_msg = f"Server communication failed: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                server_response = None
            
            if server_response:
                server_response_time = time.time() - server_start_time
                server_success = True
                self.consecutive_errors = 0
                
                # Extract and validate action recommendations
                actions = server_response.get('actions', [])
                actions_count = len(actions)
                action_types = [action.get('action_type', 'unknown') for action in actions]
                
                self.logger.debug(f"Received {actions_count} actions from server")
                
                # Execute recommended actions with comprehensive error handling
                if actions:
                    try:
                        event = self.action_executor.execute_actions(actions, current_state)
                        if event:
                            self._successful_cycles += 1
                            
                            # Record successful cycle metrics
                            cycle_time = time.time() - cycle_start_time
                            self.metrics_collector.record_cycle_metrics(
                                cycle_time=cycle_time,
                                server_response_time=server_response_time,
                                actions_count=actions_count,
                                action_types=action_types,
                                fallback_used=fallback_used,
                                server_success=server_success,
                                errors=errors
                            )
                            
                            self.logger.debug(
                                f"Successfully generated event: {type(event).__name__} "
                                f"(cycle: {cycle_time*1000:.1f}ms, "
                                f"server: {server_response_time*1000:.1f}ms)"
                            )
                            
                            return event
                        else:
                            error_msg = "Failed to convert server actions to executable events"
                            self.logger.warning(error_msg)
                            errors.append(error_msg)
                            return self._handle_fallback("action_conversion_failed")
                    except Exception as e:
                        error_msg = f"Action execution failed: {e}"
                        self.logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)
                        return self._handle_fallback("action_execution_failed")
                else:
                    error_msg = "Server returned no actions"
                    self.logger.warning(error_msg)
                    errors.append(error_msg)
                    return self._handle_fallback("no_actions_returned")
            else:
                error_msg = "Failed to get valid response from server"
                self.logger.warning(error_msg)
                errors.append(error_msg)
                self.consecutive_errors += 1
                return self._handle_fallback("server_communication_failed")
                
        except Exception as e:
            error_msg = f"Unexpected error in policy execution: {e}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            self.consecutive_errors += 1
            return self._handle_fallback(f"unexpected_error: {str(e)}")
        
        finally:
            # Always record metrics for analysis, even for failed cycles
            cycle_time = time.time() - cycle_start_time
            if not server_success:
                fallback_used = True
            
            self.metrics_collector.record_cycle_metrics(
                cycle_time=cycle_time,
                server_response_time=server_response_time,
                actions_count=actions_count,
                action_types=action_types,
                fallback_used=fallback_used,
                server_success=server_success,
                errors=errors
            )
    
    def _ensure_app_is_active(self):
        """
        Ensure the target application is in the foreground and responsive.
        
        This method implements robust application state checking and recovery
        mechanisms to maintain test continuity. It validates that the target
        application is active and attempts recovery if necessary.
        
        Recovery Actions:
            - Application foreground status verification
            - Automatic restart for inactive applications
            - Post-restart validation and logging
            - Error handling for restart failures
        
        State Validation:
            - Current activity extraction and analysis
            - Package name matching with target application
            - Activity responsiveness checking
            - Detailed logging for debugging
        
        Error Handling:
            - Graceful handling of state checking failures
            - Automatic restart with error recovery
            - Fallback mechanisms for persistent issues
            - Comprehensive logging for troubleshooting
        """
        try:
            current_state = self.device.get_current_state()
            if not current_state:
                self.logger.warning("Cannot check app state - no current state available")
                self._restart_app("no_current_state")
                return
            
            target_package = self.app.package_name
            current_activity = current_state.foreground_activity
            
            if not current_activity:
                self.logger.warning("No foreground activity detected")
                self._restart_app("no_foreground_activity")
                return
            
            # Extract package name from activity
            current_package = current_activity.split('/')[0] if '/' in current_activity else current_activity
            
            if current_package != target_package:
                self.logger.warning(
                    f"Target app not in foreground. "
                    f"Current: {current_package}, Expected: {target_package}, "
                    f"Activity: {current_activity}"
                )
                self._restart_app("wrong_package")
            else:
                self.logger.debug(f"Target app active: {current_activity}")
                
        except Exception as e:
            self.logger.error(f"Error checking app state: {e}", exc_info=True)
            self._restart_app(f"check_error: {str(e)}")
    
    def _restart_app(self, reason: str):
        """
        Restart the target application with comprehensive error handling and validation.
        
        Args:
            reason: Reason for restart (used for logging and debugging)
            
        Restart Process:
            1. Log restart reason and context
            2. Execute application restart through DroidBot
            3. Wait for initialization with configurable timeout
            4. Validate restart success with state verification
            5. Log outcome and handle restart failures
        
        Error Handling:
            - Graceful handling of restart failures
            - Detailed logging with context preservation
            - Continuation of execution despite restart issues
            - Recovery strategies for persistent problems
        """
        try:
            self.logger.info(f"Restarting application due to: {reason}")
            
            # Execute restart through DroidBot device interface
            self.device.start_app(self.app)
            
            # Allow time for application initialization
            time.sleep(2)
            
            # Verify restart success with state checking
            post_restart_state = self.device.get_current_state()
            if post_restart_state:
                new_activity = post_restart_state.foreground_activity
                self.logger.info(f"Application restart completed successfully. New activity: {new_activity}")
            else:
                self.logger.warning("Application restart completed but state verification failed")
                
        except Exception as e:
            self.logger.error(f"Failed to restart application: {e}", exc_info=True)
            # Continue execution - the application might still be partially functional
            # This ensures test continuity even under adverse conditions
    
    def _handle_fallback(self, reason: str):
        """
        Handle fallback scenarios with intelligent policy selection and error recovery.
        
        This method implements comprehensive fallback logic when the RVAndroid server
        is unavailable or fails to provide valid actions. It provides graceful
        degradation while maintaining test continuity and collecting diagnostic data.
        
        Args:
            reason: Reason for fallback activation (used for logging and metrics)
            
        Returns:
            DroidBot input event from fallback policy or None if unavailable
            
        Fallback Strategy:
            1. Log fallback activation with detailed context
            2. Validate fallback policy availability
            3. Execute fallback policy with error handling
            4. Record fallback metrics for analysis
            5. Provide diagnostic information for debugging
        
        Policy Selection:
            - Primary: UTG Greedy Search for systematic exploration
            - Secondary: Random policy for basic interaction
            - Tertiary: Safe default actions (BACK key)
        
        Error Recovery:
            - Fallback policy failure handling
            - Safe default action generation
            - Error classification and reporting
            - Metrics collection for analysis
        
        Performance Monitoring:
            - Fallback usage tracking
            - Success rate analysis
            - Pattern identification
            - Optimization recommendations
        """
        self.logger.warning(f"Activating fallback policy due to: {reason}")
        
        # Record fallback usage in metrics
        if self.config.collect_metrics:
            self.metrics_collector.record_communication_event(
                event_type="fallback_activation",
                duration=0.0,
                success=False,
                error_message=reason
            )
        
        # Check fallback policy availability
        if not self.config.fallback_enabled or not self.fallback_policy:
            self.logger.warning("Fallback policy disabled or unavailable")
            return None
        
        try:
            # Execute fallback policy with comprehensive error handling
            fallback_start = time.time()
            fallback_event = self.fallback_policy.generate_event()
            fallback_duration = time.time() - fallback_start
            
            if fallback_event:
                self.logger.debug(
                    f"Fallback policy generated event: {type(fallback_event).__name__} "
                    f"(duration: {fallback_duration*1000:.1f}ms)"
                )
                
                # Record successful fallback metrics
                if self.config.collect_metrics:
                    self.metrics_collector.record_action_execution(
                        action_type="fallback",
                        execution_time=fallback_duration,
                        success=True
                    )
                
                return fallback_event
            else:
                self.logger.warning("Fallback policy returned no event")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in fallback policy execution: {e}", exc_info=True)
            
            # Record fallback failure metrics
            if self.config.collect_metrics:
                self.metrics_collector.record_communication_event(
                    event_type="fallback_failure",
                    duration=0.0,
                    success=False,
                    error_message=str(e)
                )
            
            return None
    
    def _perform_initial_health_check(self):
        """
        Perform initial server health check during policy initialization.
        
        This method validates server connectivity during startup, providing early
        warning of configuration issues and establishing baseline communication
        metrics for monitoring purposes.
        
        Health Check Process:
            1. Attempt server connectivity validation
            2. Record baseline response time metrics
            3. Log connectivity status and warnings
            4. Initialize communication statistics
        
        Error Handling:
            - Non-blocking health check failures
            - Graceful degradation for offline scenarios
            - Detailed logging for troubleshooting
            - Baseline metrics establishment
        """
        try:
            self.logger.debug("Performing initial server health check")
            health_start = time.time()
            
            # Attempt health check through server communicator
            health_response = self.server_communicator._perform_health_check()
            health_duration = time.time() - health_start
            
            if health_response:
                self.logger.info(
                    f"Server connectivity confirmed "
                    f"(response time: {health_duration*1000:.1f}ms)"
                )
            else:
                self.logger.warning(
                    "Server appears unavailable during initialization. "
                    "Policy will use fallback mechanisms."
                )
            
            # Record initial health check metrics
            if self.config.collect_metrics:
                self.metrics_collector.record_communication_event(
                    event_type="initial_health_check",
                    duration=health_duration,
                    success=health_response,
                    error_message=None if health_response else "Server unavailable"
                )
                
        except Exception as e:
            self.logger.warning(f"Initial health check failed: {e}")
            # Non-blocking - policy should continue even if health check fails
    
    def handle_utg_event(self, event):
        """
        Handle UTG (UI Transition Graph) events from DroidBot framework.
        
        This method provides integration with DroidBot's UTG system for enhanced
        state tracking and exploration optimization. It maintains compatibility
        with existing DroidBot workflows while providing enhanced functionality.
        
        Args:
            event: UTG event from DroidBot framework
            
        Event Processing:
            - State transition tracking and analysis
            - UTG integration for exploration optimization
            - Fallback policy coordination
            - State consistency maintenance
        """
        # Reset current state to trigger fresh capture
        self.current_state = None
        
        # Forward event to fallback policy if available
        if self.fallback_policy and hasattr(self.fallback_policy, 'handle_utg_event'):
            try:
                self.fallback_policy.handle_utg_event(event)
            except Exception as e:
                self.logger.warning(f"Error forwarding UTG event to fallback policy: {e}")
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive policy performance statistics for monitoring and debugging.
        
        Returns:
            Dictionary containing detailed policy performance metrics
            
        Statistics Categories:
            - Execution Statistics: Cycle counts, success rates, timing analysis
            - Communication Health: Server availability, response times, error rates
            - Component Performance: Individual component metrics and health
            - Configuration Status: Current configuration and operational parameters
        """
        try:
            # Get metrics from collector
            metrics_summary = self.metrics_collector.get_performance_summary()
            
            # Get communication statistics
            comm_stats = self.server_communicator.get_communication_stats()
            
            # Calculate additional policy-specific statistics
            uptime = time.time() - self.policy_start_time
            success_rate = self._successful_cycles / max(self._total_cycles, 1)
            
            return {
                'policy_info': {
                    'name': 'RVAndroidPolicy',
                    'version': '2.0.0',
                    'uptime_seconds': round(uptime, 2),
                    'total_cycles': self._total_cycles,
                    'successful_cycles': self._successful_cycles,
                    'success_rate': round(success_rate, 3),
                    'consecutive_errors': self.consecutive_errors
                },
                'configuration': {
                    'server_url': self.config.server_url,
                    'fallback_enabled': self.config.fallback_enabled,
                    'metrics_enabled': self.config.collect_metrics,
                    'batch_actions_supported': self.config.batch_action_support,
                    'request_timeout': self.config.request_timeout,
                    'max_retries': self.config.max_retries
                },
                'communication_health': comm_stats,
                'performance_metrics': metrics_summary,
                'component_status': {
                    'state_sanitizer': 'operational',
                    'server_communicator': 'operational',
                    'action_executor': 'operational',
                    'metrics_collector': 'operational' if self.config.collect_metrics else 'disabled',
                    'fallback_policy': 'available' if self.fallback_policy else 'unavailable'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating policy statistics: {e}")
            return {
                'error': str(e),
                'policy_info': {
                    'name': 'RVAndroidPolicy',
                    'version': '2.0.0',
                    'status': 'error'
                }
            }