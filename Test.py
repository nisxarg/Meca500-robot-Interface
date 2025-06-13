"""
Manual Robot Control GUI
-------------------------

"""

import re
from PyQt6.QtWidgets import QMainWindow
import threading
from PyQt6.QtCore import QMetaObject, Q_ARG, Qt

import cv2
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel
from PyQt6.QtWidgets import QStackedLayout
import time
import sys
from functools import partial
from typing import List, Callable, Dict
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout,
    QSlider, QTextEdit, QGridLayout, QPushButton, QTabWidget,
    QLineEdit, QGroupBox, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from mecademicpy.robot import Robot
import pygame


# Constants
JOINT_LIMITS = [
    (-175, 175),  # Joint 1
    (-70, 90),    # Joint 2
    (-135, 70),   # Joint 3
    (-170, 170),  # Joint 4
    (-115, 115),  # Joint 5
    (-360, 360)   # Joint 6
]

SLIDER_RANGE = 100
DEFAULT_ROBOT_IP = "192.168.0.100"
DEFAULT_VELOCITY = 20
DEFAULT_STEP_SIZE = 1.0
GRIPPER_MAX_OPENING = 5.8  # mm
TIMER_INTERVAL = 50  # ms
CONNECTION_RETRY_INTERVAL = 3000  # ms
JOYSTICK_CHECK_INTERVAL = 2000  # ms
ERROR_DEBOUNCE_INTERVAL = 3.0  # seconds

class ConsoleInterceptor:
    """
    Redirects stdout to the GUI, adding timestamps and intelligent debouncing to reduce spam.
    """
    def __init__(self, callback: Callable[[str], None]):
        """Initializes the interceptor."""
        self.callback = callback
        self._stdout = sys.__stdout__
        self._stderr = sys.__stderr__

        # Stores the last time a message *key* was logged.
        self._last_log_time: Dict[str, float] = {}
        # Stores the content of the last message to prevent immediate exact duplicates.
        self._last_message_content: str = ""
        # Flag to track if we've already notified the user about the error state.
        self._error_state_notified: bool = False

    def _get_message_key(self, msg: str) -> str:
        """Creates a key for a message to group similar spammy messages."""
        msg_lower = msg.lower()
        if "robot is in error" in msg_lower or "please reset" in msg_lower or "already in err" in msg_lower:
            return "error_state"
        if "reset complete" in msg_lower or "resetting robot" in msg_lower:
            # When a reset happens, clear the error notification flag.
            self._error_state_notified = False
            return "reset_action"
        if "would exceed limits" in msg_lower:
            return "limit_warning"
        # For other messages, use the message itself as the key.
        return msg

    def write(self, msg: str) -> None:
        """
        Intelligently writes a message to the GUI, filtering out spam.
        """
        msg = msg.strip()
        if not msg:
            return

        now = time.time()

        # 1. Prevent exact same message from repeating back-to-back immediately.
        if msg == self._last_message_content:
            return
        self._last_message_content = msg

        # 2. Use a key to group and throttle similar messages.
        key = self._get_message_key(msg)

        # If we are in an error state and another error message comes in, suppress it.
        if key == "error_state":
            if self._error_state_notified:
                return  # We already told the user the robot is in an error state.
            # Otherwise, log it and set the flag to prevent more error messages.
            self._error_state_notified = True

        # 3. Throttle all messages by their key to prevent rapid-fire repeats.
        last_time = self._last_log_time.get(key, 0)
        # Allow messages of the same type (key) every 2 seconds.
        if now - last_time < 2.0:
            return

        self._last_log_time[key] = now
        timestamp = time.strftime("[%H:%M:%S]", time.localtime(now))
        final_msg = f"{timestamp} {msg}"

        # Write to the terminal and the GUI console.
        self._stdout.write(final_msg + '\n')
        try:
            self.callback(final_msg)
        except Exception as e:
            self._stdout.write(f"--- ERROR in GUI Console Callback: {e}\n")

    def flush(self) -> None:
        """Flushes the underlying terminal output."""
        self._stdout.flush()
        self._stderr.flush()

class CameraFeed(QLabel):
    """Individual camera feed widget that can be clicked to maximize"""

    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.parent_window = parent
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.first_frame_received = False

        # Setup label properties
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.setText(f"Camera {camera_id + 1}\nLoading...")

        # Enable mouse events
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        """Handle mouse click to maximize/restore camera"""
        if event.button() == Qt.MouseButton.LeftButton and self.parent_window:
            self.parent_window.toggle_camera_maximize(self.camera_id)
        super().mousePressEvent(event)

    def start_camera(self, width=640, height=480):
        """Start camera capture in a separate thread"""
        def init_camera():
            print(f"[CameraFeed] Starting camera {self.camera_id} init...")

            # Set loading message from background thread
            QMetaObject.invokeMethod(
                self,
                "setText",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, f"Camera {self.camera_id + 1}\nLoading...")
            )

            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            if not self.cap.isOpened():
                print(f"[CameraFeed] Failed to open camera {self.camera_id}.")
                QMetaObject.invokeMethod(
                    self,
                    "setText",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, f"Camera {self.camera_id + 1}\n❌ Failed to open")
                )
                return

            print(f"[CameraFeed] Camera {self.camera_id} opened successfully.")
            self.first_frame_received = False

            QMetaObject.invokeMethod(
                self.timer,
                "start",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(int, 30)
            )

        threading.Thread(target=init_camera, daemon=True).start()

    def update_frame(self):
        """Update camera frame"""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        if not self.first_frame_received:
            self.clear()
            self.first_frame_received = True

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)

    def stop_camera(self):
        """Stop camera capture and release resources"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

class CameraWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Camera Feed")
        self.setMinimumSize(1000, 800)
        self.setWindowFlags(Qt.WindowType.Window)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create stacked layout for grid and maximized views
        self.stacked_layout = QStackedLayout()

        # Create grid view widget
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)

        # Create camera feeds for 4 cameras (IDs 0, 1, 2, 3)
        self.camera_feeds = []
        for i in range(4):
            camera_feed = CameraFeed(i, self)
            self.camera_feeds.append(camera_feed)

            # Arrange in 2x2 grid
            row = i // 2
            col = i % 2
            self.grid_layout.addWidget(camera_feed, row, col)

        # Create maximized view widget
        self.maximized_widget = QWidget()
        self.maximized_layout = QVBoxLayout(self.maximized_widget)

        # Back button for returning to grid view
        self.back_button = QPushButton("← Back to Grid View")
        self.back_button.clicked.connect(self.show_grid_view)
        self.back_button.setMaximumHeight(40)
        self.maximized_layout.addWidget(self.back_button)

        # Placeholder for maximized camera
        self.maximized_camera_label = QLabel("Click on a camera to maximize")
        self.maximized_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.maximized_camera_label.setStyleSheet("border: 2px solid gray; background-color: black; color: white;")
        self.maximized_layout.addWidget(self.maximized_camera_label)

        # Add widgets to stacked layout
        self.stacked_layout.addWidget(self.grid_widget)      # index 0 = grid view
        self.stacked_layout.addWidget(self.maximized_widget) # index 1 = maximized view

        # Add stacked layout to main layout
        stacked_container = QWidget()
        stacked_container.setLayout(self.stacked_layout)
        self.main_layout.addWidget(stacked_container)

        # Track current state
        self.current_view = "grid"  # "grid" or "maximized"
        self.maximized_camera_id = None

        # Timer for maximized view updates
        self.maximized_timer = QTimer()
        self.maximized_timer.timeout.connect(self.update_maximized_frame)

    def start_cameras(self, width=640, height=480):
        """Start all camera feeds"""
        print("[CameraWindow] Starting all cameras...")
        for camera_feed in self.camera_feeds:
            camera_feed.start_camera(width, height)

    def toggle_camera_maximize(self, camera_id):
        """Toggle between grid view and maximized view for a specific camera"""
        if self.current_view == "grid":
            self.show_maximized_view(camera_id)
        else:
            self.show_grid_view()

    def show_maximized_view(self, camera_id):
        """Show maximized view of a specific camera"""
        self.current_view = "maximized"
        self.maximized_camera_id = camera_id
        self.stacked_layout.setCurrentIndex(1)

        # Update back button text
        self.back_button.setText(f"← Back to Grid View (Camera {camera_id + 1} Maximized)")

        # Start timer for maximized view
        self.maximized_timer.start(30)

        print(f"[CameraWindow] Maximized camera {camera_id + 1}")

    def show_grid_view(self):
        """Show grid view with all cameras"""
        self.current_view = "grid"
        self.maximized_camera_id = None
        self.stacked_layout.setCurrentIndex(0)

        # Stop maximized timer
        self.maximized_timer.stop()

        print("[CameraWindow] Returned to grid view")

    def update_maximized_frame(self):
        """Update the maximized camera frame"""
        if self.maximized_camera_id is None:
            return

        camera_feed = self.camera_feeds[self.maximized_camera_id]
        if not camera_feed.cap:
            return

        ret, frame = camera_feed.cap.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.maximized_camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.maximized_camera_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Clean up when window is closed"""
        self.maximized_timer.stop()
        for camera_feed in self.camera_feeds:
            camera_feed.stop_camera()
        event.accept()

class MecaPendant(QWidget):
    """
    Main GUI class for controlling the Meca500 robot.
    """

    def __init__(self):
        super().__init__()


        # Setup window properties
        self.setWindowTitle("Versacell Robotic System")


        #  dark theme to the entire application
        self._apply_dark_theme()

        # Initialize robot connection
        self.robot = Robot()

        # Create console and camera widgets
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-family: Consolas; font-size: 11px; color: #00ff00; background-color: #1e1e1e; border: 1px solid #555555;")

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setScaledContents(True)
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Forbidden zones: list of ((x1,y1,z1), (x2,y2,z2))
        self.forbidden_zones = [
            ((200, 0, 308), (300, -130, 200)),  # Example box 1
            ((61, -298, 34), (-126, -114, 121))  # Example box 2
        ]
        self._current_forbidden_zone = None
        self.first_frame_received = False
        self.camera_label.setText("Loading camera feed...")
        self.camera_label.setStyleSheet("font-size: 20px; color: #cccccc; background-color: #1e1e1e; border: 1px solid #555555;")

        # Stack console and camera in the same space
        self.console_camera_stack = QStackedLayout()
        self.console_camera_stack.addWidget(self.console)  # index 0 = console
        self.console_camera_stack.addWidget(self.camera_label)  # index 1 = camera

        # Wrap in a container
        self.console_container = QWidget()
        self.console_container.setLayout(self.console_camera_stack)

        # Redirect stdout to console
        def append_to_console(msg: str):
            self.console.append(msg)

        sys.stdout = sys.stderr = ConsoleInterceptor(append_to_console)

        # Track end effector/tool type
        self.is_vacuum_tool = False  # Will be set by detection logic

        # Initialize timers, joystick, and state
        self.camera_enabled = False
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_capture = None

        self._init_state_variables()
        self._init_timers()
        self._init_joystick()

        # Build the rest of the UI (now console_container is ready)
        self._build_ui()
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setMinimumSize(1024, 768)

        # Set control mode defaults
        self.update_control_buttons()
        self.set_control_mode("mouse")
        self.highlight_joint_group()

    def _apply_dark_theme(self):
        """Apply a refined dark theme to the entire application"""
        dark_style = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: Arial, sans-serif;
        }
        
        QMainWindow {
            background-color: #1e1e1e;
        }
        
        QGroupBox {
            background-color: #3c3c3c;
            border: 1px solid #555555;
            border-radius: 6px;
            margin-top: 1ex;
            font-weight: bold;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #404040;
            border: 1px solid #606060;
            border-radius: 4px;
            padding: 6px 12px;
            color: #ffffff;
            font-weight: normal;
        }
        
        QPushButton:hover {
            background-color: #505050;
            border-color: #707070;
        }
        
        QPushButton:pressed {
            background-color: #353535;
            border-color: #505050;
        }
        
        QPushButton:checked {
            background-color: #0078d4;
            border-color: #106ebe;
            color: white;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 8px;
            background: #2b2b2b;
            margin: 2px 0;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #0078d4;
            border: 1px solid #106ebe;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #106ebe;
        }
        
        QLineEdit {
            background-color: #404040;
            border: 1px solid #606060;
            border-radius: 4px;
            padding: 4px 8px;
            color: #ffffff;
        }
        
        QLineEdit:focus {
            border-color: #0078d4;
        }
        
        QLabel {
            color: #ffffff;
            background-color: transparent;
        }
        
        QTabWidget::pane {
            border: 1px solid #555555;
            background-color: #3c3c3c;
        }
        
        QTabBar::tab {
            background-color: #404040;
            border: 1px solid #606060;
            padding: 8px 16px;
            margin-right: 2px;
            color: #ffffff;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d4;
            border-color: #106ebe;
            color: white;
        }
        
        QTabBar::tab:hover {
            background-color: #505050;
        }
        
        QTextEdit {
            background-color: #2b2b2b;
            border: 1px solid #555555;
            color: #ffffff;
        }
        
        QScrollBar:vertical {
            background-color: #3c3c3c;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #606060;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #707070;
        }
        """

        self.setStyleSheet(dark_style)

    def is_pose_in_forbidden_zone(self, pose: List[float]) -> bool:
        """Call this ONCE per actual new robot pose!"""
        if not pose or len(pose) < 3:
            return False

        x, y, z = pose[:3]
        zone_now = None

        # Determine which forbidden zone (if any) the pose is in
        for idx, (c1, c2) in enumerate(self.forbidden_zones):
            x_min, x_max = sorted([c1[0], c2[0]])
            y_min, y_max = sorted([c1[1], c2[1]])
            z_min, z_max = sorted([c1[2], c2[2]])
            if x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max:
                zone_now = chr(65 + idx)
                break  # Only one zone at a time

        # State machine: only log on transitions
        if zone_now != self._current_forbidden_zone:
            if self._current_forbidden_zone is not None:
                self.log(f"✅ Exited forbidden zone '{self._current_forbidden_zone}'")
            if zone_now is not None:
                self.log(f"❌ Entered forbidden zone '{zone_now}': ({x:.1f}, {y:.1f}, {z:.1f})")
            self._current_forbidden_zone = zone_now

        return zone_now is not None
    def simulate_joint_movement(self, joint_deltas: List[float]) -> List[float]:
        """Simulate what the robot pose would be after applying joint deltas"""
        try:
            current_joints = self.robot.GetJoints()
            if not current_joints:
                return None

            # Apply deltas to current joints
            new_joints = [current_joints[i] + joint_deltas[i] for i in range(6)]

            # Check joint limits
            for i, (new_joint, (min_lim, max_lim)) in enumerate(zip(new_joints, JOINT_LIMITS)):
                if not (min_lim <= new_joint <= max_lim):
                    self.log(f"⚠️ Joint {i + 1} would exceed limits: {new_joint:.2f} (range: {min_lim} to {max_lim})")
                    return None

            # For joint movements, we'll use a simplified approach:
            # Since we don't have access to forward kinematics, we'll assume
            # that small joint movements result in proportional cartesian movements
            # This is a rough approximation but better than nothing

            current_pose = self.robot.GetPose()
            if not current_pose:
                return None

            # For small movements, approximate the cartesian change
            # This is a very simplified approach - in production you'd use proper FK
            estimated_pose = current_pose.copy()

            # Small joint movements typically result in small cartesian movements
            # We'll use a conservative estimate
            max_joint_change = max(abs(d) for d in joint_deltas)
            if max_joint_change > 0:
                # Estimate maximum possible cartesian displacement
                # This is very conservative - actual displacement could be much smaller
                max_cart_displacement = max_joint_change * 10  # rough approximation in mm

                # For safety, we'll check if any axis could potentially move this much
                for i in range(3):  # X, Y, Z axes
                    estimated_pose[i] += max_cart_displacement
                    if self.is_pose_in_forbidden_zone(estimated_pose):
                        return None
                    estimated_pose[i] -= 2 * max_cart_displacement
                    if self.is_pose_in_forbidden_zone(estimated_pose):
                        return None
                    estimated_pose[i] += max_cart_displacement  # restore

            return estimated_pose

        except Exception as e:
            self.log(f"[ERROR] simulate_joint_movement: {e}")
            return None

    def simulate_cartesian_movement(self, cart_deltas: List[float]) -> List[float]:
        """Simulate what the robot pose would be after applying cartesian deltas"""
        try:
            current_pose = self.robot.GetPose()
            if not current_pose:
                return None

            # Apply deltas to current pose
            new_pose = [current_pose[i] + cart_deltas[i] for i in range(6)]
            return new_pose

        except Exception as e:
            self.log(f"[ERROR] simulate_cartesian_movement: {e}")
            return None

    def is_movement_safe(self, movement_type: str, deltas: List[float]) -> bool:
        """Check if a movement would be safe (not enter forbidden zone)"""
        try:
            # Get current pose
            current_pose = self.robot.GetPose()
            if not current_pose:
                self.log("⚠️ Cannot get current robot pose")
                return False

            # Check if already in forbidden zone
            if self.is_pose_in_forbidden_zone(current_pose):
                self.log("❌ Robot is already in forbidden zone - movement blocked")
                return False

            # Simulate the movement
            if movement_type == "joint":
                new_pose = self.simulate_joint_movement(deltas)
            elif movement_type == "cartesian":
                new_pose = self.simulate_cartesian_movement(deltas)
            else:
                self.log(f"⚠️ Unknown movement type: {movement_type}")
                return False

            if new_pose is None:
                return False

            # Check if new pose would be in forbidden zone
            if self.is_pose_in_forbidden_zone(new_pose):
                self.log("❌ Movement blocked: Would enter forbidden zone")
                return False

            return True

        except Exception as e:
            self.log(f"[ERROR] is_movement_safe: {e}")
            return False

    def get_max_safe_step(self, axis: int, direction: int, step_size: float, movement_type: str = "cartesian") -> float:
        """Get the maximum safe step size before hitting a forbidden zone"""
        try:
            current_pose = self.robot.GetPose()
            if not current_pose:
                return 0.0

            # If already in forbidden zone, don't allow any movement
            if self.is_pose_in_forbidden_zone(current_pose):
                return 0.0

            max_safe = step_size
            test_step = step_size * 0.1  # Start with 10% of requested step

            # Binary search for maximum safe step
            for _ in range(10):  # Limit iterations
                deltas = [0.0] * 6
                deltas[axis] = test_step * direction

                if movement_type == "cartesian":
                    test_pose = self.simulate_cartesian_movement(deltas)
                else:  # joint
                    test_pose = self.simulate_joint_movement(deltas)

                if test_pose and not self.is_pose_in_forbidden_zone(test_pose):
                    max_safe = test_step
                    test_step = min(test_step * 1.5, step_size)  # Increase step
                else:
                    test_step = test_step * 0.8  # Decrease step

                if test_step < 0.001:  # Minimum meaningful step
                    break

            return max_safe

        except Exception as e:
            self.log(f"[ERROR] get_max_safe_step: {e}")
            return 0.0

    def _init_state_variables(self) -> None:
        """Initialize all state variables with default values"""
        # Control parameters
        self.velocity_percent = DEFAULT_VELOCITY
        self.joint_step = DEFAULT_STEP_SIZE
        self.cart_step_mm = DEFAULT_STEP_SIZE
        self.cart_step_deg = DEFAULT_STEP_SIZE

        # UI state tracking
        self.joint_active = [False] * 6
        self.cart_active = [False] * 6
        self.joint_sliders = []
        self.cart_sliders = []
        self.joint_inputs = []
        self.cart_inputs = []
        self.joint_boxes = []
        self.cart_boxes = []

        # Error handling state
        self._error_popup_shown = False

        # Joystick state
        self._joystick_was_connected = False
        self.gripper_open = False
        self.control_mode = "mouse"
        self.joystick_submode = "joint"
        self.joystick_joint_group = 0
        self.last_button_states = []

        # Connection state tracking
        self._last_conn_status = None
        self._last_activation_status = False
        self._reconnect_message_shown = False

    def _build_ui(self) -> None:
        """Build the complete user interface"""
        self.tabs = QTabWidget()
        self.init_joint_tab()
        self.init_cartesian_tab()

        left_panel = self._create_left_panel()

        right_panel = self._create_right_panel()
        status_bar = self._create_status_bar()

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addLayout(left_panel, stretch=3)
        main_layout.addLayout(right_panel, stretch=2)

        final_layout = QVBoxLayout()
        final_layout.addLayout(main_layout)
        final_layout.addLayout(status_bar)
        self.setLayout(final_layout)

    def _create_left_panel(self) -> QVBoxLayout:
        """Create the left panel with all controls"""
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # Add tabs at the top
        left_panel.addWidget(self.tabs)

        # Add control mode buttons
        left_panel.addWidget(self._create_control_mode_group())

        # Add robot control buttons
        left_panel.addWidget(self._create_robot_controls_group())

        # Add velocity control
        left_panel.addWidget(self._create_velocity_control_group())

        # Add increment control
        left_panel.addWidget(self._create_increment_control_group())

        # Add gripper control
        left_panel.addWidget(self._create_gripper_control_group())

        return left_panel

    def _create_control_mode_group(self) -> QGroupBox:
        """Create the control mode selection group"""
        mode_box = QGroupBox("Control Mode")
        mode_layout = QHBoxLayout()

        self.mouse_btn = QPushButton("Mouse Mode")
        self.joystick_btn = QPushButton("Joystick Mode")

        self.mouse_btn.clicked.connect(lambda: self.set_control_mode("mouse"))
        self.joystick_btn.clicked.connect(self.toggle_joystick_mode)

        mode_layout.addWidget(self.mouse_btn)
        mode_layout.addWidget(self.joystick_btn)

        mode_box.setLayout(mode_layout)
        return mode_box


    def _create_robot_controls_group(self) -> QGroupBox:
        """Create the robot control buttons group"""
        control_box = QGroupBox("Robot Controls")
        ctrl_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset Error")
        self.reset_button.clicked.connect(self.reset_error)
        ctrl_layout.addWidget(self.reset_button)

        self.home_button = QPushButton("Home")
        self.home_button.clicked.connect(self.go_home)
        ctrl_layout.addWidget(self.home_button)

        self.end_effector_btn1 = QPushButton()
        self.end_effector_btn2 = QPushButton()
        self.end_effector_btn1.clicked.connect(self.handle_end_effector_btn1)
        self.end_effector_btn2.clicked.connect(self.handle_end_effector_btn2)
        ctrl_layout.addWidget(self.end_effector_btn1)
        ctrl_layout.addWidget(self.end_effector_btn2)

        control_box.setLayout(ctrl_layout)
        return control_box
    def _create_velocity_control_group(self) -> QGroupBox:
        """Create the velocity control group"""
        vel_box = QGroupBox("Maximum Jogging Velocity")
        vel_layout = QHBoxLayout()

        self.vel_input = QLineEdit(str(self.velocity_percent))
        self.vel_input.setFixedWidth(50)
        self.vel_input.returnPressed.connect(self.manual_velocity_input)

        vel_dec = QPushButton("<")
        vel_dec.clicked.connect(partial(self.adjust_velocity, -10))

        self.vel_slider = QSlider(Qt.Orientation.Horizontal)
        self.vel_slider.setMinimum(10)
        self.vel_slider.setMaximum(100)
        self.vel_slider.setValue(self.velocity_percent)
        self.vel_slider.setTickInterval(10)
        self.vel_slider.valueChanged.connect(self.set_velocity)

        vel_inc = QPushButton(">")
        vel_inc.clicked.connect(partial(self.adjust_velocity, 10))

        for w in [self.vel_input, vel_dec, self.vel_slider, vel_inc]:
            vel_layout.addWidget(w)

        vel_box.setLayout(vel_layout)
        return vel_box

    def _create_increment_control_group(self) -> QGroupBox:
        """Create the increment control group"""
        inc_box = QGroupBox("Jog Increment (° / mm)")
        inc_layout = QHBoxLayout()

        self.inc_input = QLineEdit(f"{self.joint_step:.1f}")
        self.inc_input.setFixedWidth(50)
        self.inc_input.returnPressed.connect(self.manual_increment_input)

        inc_dec = QPushButton("<")
        inc_dec.clicked.connect(partial(self.adjust_increment, -1))

        self.inc_slider = QSlider(Qt.Orientation.Horizontal)
        self.inc_slider.setMinimum(1)
        self.inc_slider.setMaximum(50)
        self.inc_slider.setValue(int(self.joint_step * 10))
        self.inc_slider.setTickInterval(1)
        self.inc_slider.valueChanged.connect(self.update_increment_from_slider)

        inc_inc = QPushButton(">")
        inc_inc.clicked.connect(partial(self.adjust_increment, 1))

        for w in [self.inc_input, inc_dec, self.inc_slider, inc_inc]:
            inc_layout.addWidget(w)

        inc_box.setLayout(inc_layout)
        return inc_box

    def _create_gripper_control_group(self) -> QGroupBox:
        """Create the gripper control group"""
        gripper_box = QGroupBox("Gripper Control")
        gripper_layout = QVBoxLayout()

        # First row: Gripper percentage slider
        slider_row = QHBoxLayout()
        gripper_label = QLabel("Gripper %")
        gripper_label.setFixedWidth(70)

        self.gripper_slider = QSlider(Qt.Orientation.Horizontal)
        self.gripper_slider.setMinimum(0)
        self.gripper_slider.setMaximum(100)
        self.gripper_slider.setTickInterval(10)
        self.gripper_slider.setSingleStep(1)
        self.gripper_slider.setValue(50)
        self.gripper_slider.valueChanged.connect(self.set_gripper_percent)

        self.gripper_value_label = QLabel("50%")
        self.gripper_value_label.setFixedWidth(40)
        self.gripper_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        slider_row.addWidget(gripper_label)
        slider_row.addWidget(self.gripper_slider, 1)
        slider_row.addWidget(self.gripper_value_label)

        # Second row: Switch button
        self.detect_tool_btn = QPushButton("Switch to Vacuum/Gripper")
        self.detect_tool_btn.clicked.connect(self.toggle_tool_type)
        self.detect_tool_btn.setMaximumHeight(30)

        gripper_layout.addLayout(slider_row)
        gripper_layout.addWidget(self.detect_tool_btn)
        gripper_box.setLayout(gripper_layout)
        return gripper_box

    def _create_right_panel(self) -> QVBoxLayout:
        """Create the right panel with console, emergency stop, and programming controls"""
        right_panel = QVBoxLayout()

        # ✅ Emergency Stop button at the top
        emergency_btn = QPushButton("EMERGENCY STOP")
        emergency_btn.setStyleSheet(
            "background-color: red; color: white; font-weight: bold; font-size: 16px; padding: 1px;"
        )
        emergency_btn.clicked.connect(self.emergency_stop)
        right_panel.addWidget(emergency_btn)

        # ✅ Console section
        console_label = QLabel("Console")
        console_label.setStyleSheet("font-weight: bold;")
        right_panel.addWidget(console_label)

        right_panel.addWidget(self.console_container, stretch=1)

        # ✅ Compact button row for Clear Console and Toggle Camera View
        button_row_layout = QHBoxLayout()

        clear_console_button = QPushButton("Clear")
        clear_console_button.setMaximumWidth(80)
        clear_console_button.clicked.connect(self.console.clear)
        button_row_layout.addWidget(clear_console_button)

        toggle_cam_btn = QPushButton("Camera")
        toggle_cam_btn.setMaximumWidth(80)
        toggle_cam_btn.clicked.connect(self.toggle_camera_view)
        button_row_layout.addWidget(toggle_cam_btn)

        button_row_layout.addStretch()  # Push buttons to the left

        button_row_widget = QWidget()
        button_row_widget.setLayout(button_row_layout)
        right_panel.addWidget(button_row_widget)

        # ✅ Add robot programming interface
        from meca500_programming_interface import add_programming_interface_to_gui
        self.programming_interface = add_programming_interface_to_gui(self)
        right_panel.addWidget(self.programming_interface)

        return right_panel

    def toggle_camera(self):
        self.log("📷 Toggle Camera clicked (function not implemented).")

    def toggle_camera_view(self):
        if hasattr(self, 'camera_window') and self.camera_window.isVisible():
            self.camera_window.close()
            self.log("Multi-camera window closed.")
        else:
            self.camera_window = CameraWindow(self)
            self.camera_window.show()
            QTimer.singleShot(100, self.camera_window.start_cameras)  # Slight delay to avoid UI lag
            self.log("Multi-camera window opened with 4 camera feeds.")

    def update_camera_frame(self):
        if not self.camera_capture:
            return
        ret, frame = self.camera_capture.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(qt_img))

        # In Test.py, add this method inside the MecaPendant class

    def emergency_stop(self):
            """Emergency stop: Abort program, pause motion, and deactivate the robot."""
            try:
                # First, stop any running program sequence to prevent new commands.
                if hasattr(self, 'programming_interface') and self.programming_interface.running:
                    self.programming_interface.stop_program()
                    self.log("▶️ Program execution aborted by Emergency Stop.")

                # Halt the robot's physical motion.
                self.robot.PauseMotion()
                self.log("🛑 Motion Paused.")

                # Deactivate the robot to cut power to motors.
                self.robot.DeactivateRobot()
                self.log("🚨 EMERGENCY STOP: Robot has been deactivated.")

                # Display a critical warning to the user.
                QMessageBox.critical(
                    self,
                    "Emergency Stop Activated",
                    "Motion has been paused and the robot is now deactivated.\n"
                    "The running program was aborted.\n\n"
                    "To recover:\n"
                    "1. Power-cycle the robot.\n"
                    "2. Then re-activate via the GUI."
                )
            except Exception as e:
                self.log(f"[ERROR] An error occurred during emergency stop: {e}")

    def toggle_tool_type(self):
        """Toggle between vacuum and gripper mode manually."""
        self.is_vacuum_tool = not self.is_vacuum_tool
        self.update_end_effector_buttons()
        tool = "Vacuum" if self.is_vacuum_tool else "Gripper"
        self.log(f"Manually set tool: {tool}")

    def update_end_effector_buttons(self):
        """Update the labels and callbacks of the two end-effector buttons depending on tool type."""
        btn1 = getattr(self, "end_effector_btn1", None)
        btn2 = getattr(self, "end_effector_btn2", None)
        if btn1 is None or btn2 is None:
            return

        if self.is_vacuum_tool:
            btn1.setText("Vacuum On")
            btn2.setText("Vacuum Off")
            btn1.setStyleSheet("background-color: #2196F3; color: white;")
            btn2.setStyleSheet("background-color: #b71c1c; color: white;")
            btn1.setToolTip("Activate pneumatic vacuum (pick up part)")
            btn2.setToolTip("Release pneumatic vacuum (drop part)")
        else:
            btn1.setText("Open Gripper")
            btn2.setText("Close Gripper")
            btn1.setStyleSheet("")
            btn2.setStyleSheet("")
            btn1.setToolTip("Open the mechanical gripper")
            btn2.setToolTip("Close the mechanical gripper")

        # Update the status label to reflect current tool type
        self.update_gripper_label(self.gripper_open)

    def handle_end_effector_btn1(self):
        if self.is_vacuum_tool:
            self.vacuum_on()
        else:
            self.open_gripper()

    def handle_end_effector_btn2(self):
        if self.is_vacuum_tool:
            self.vacuum_off()
        else:
            self.close_gripper()

    def vacuum_on(self):
        """Turn vacuum ON (activate suction via pneumatic valve)"""
        try:
            # Only use SetValveState (do not use VacuumGrip!)
            self.robot.SetValveState(0,1)
            self.log("Vacuum ON (suction activated).")
            self.update_gripper_label(True)
        except Exception as e:
            self.log(f"[ERROR] Vacuum on (SetValveState()): {e}")

    def vacuum_off(self):
        """Turn vacuum OFF (release suction via pneumatic valve)"""
        try:
            self.robot.SetValveState(0,0)
            self.log("Vacuum OFF (released).")
            self.update_gripper_label(False)
        except Exception as e:
            self.log(f"[ERROR] Vacuum off (SetValveState(0)): {e}")
    def _create_status_bar(self) -> QHBoxLayout:
        """Create the status bar at the bottom of the window"""
        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(5, 5, 5, 5)
        status_bar.setSpacing(20)

        self.conn_label = QLabel("❌ Disconnected")
        self.conn_label.setStyleSheet("color: red;")

        self.mode_label = QLabel("🖱️ Mouse Mode")
        self.mode_label.setStyleSheet("color: white;")

        self.gripper_label = QLabel("🧲 Gripper: Closed")
        self.gripper_label.setStyleSheet("color: white;")

        self.activate_label = QLabel("⛔ Inactive")
        self.activate_label.setStyleSheet("color: orange; text-decoration: underline; cursor: pointer;")
        self.activate_label.mousePressEvent = self.force_activate_robot

        self.update_gripper_label(self.gripper_open)

        status_bar.addWidget(self.conn_label)
        status_bar.addStretch()
        status_bar.addWidget(self.mode_label)
        status_bar.addStretch()
        status_bar.addWidget(self.gripper_label)
        status_bar.addWidget(self.activate_label)

        return status_bar

    def _init_timers(self) -> None:
        """Initialize all timers used by the application"""
        # Timer for joint jogging
        self.joint_timer = QTimer()
        self.joint_timer.timeout.connect(self.joint_jog_loop)
        self.joint_timer.start(TIMER_INTERVAL)

        # Timer for cartesian jogging
        self.cart_timer = QTimer()
        self.cart_timer.timeout.connect(self.cartesian_jog_loop)
        self.cart_timer.start(TIMER_INTERVAL)

        # Timer for updating position displays
        self.pose_timer = QTimer()
        self.pose_timer.timeout.connect(self.update_joint_and_pose_inputs)
        self.pose_timer.start(1000)  # Update once per second

        # Timer for auto-starting the robot
        self.auto_start_timer = QTimer()
        self.auto_start_timer.setSingleShot(True)
        self.auto_start_timer.timeout.connect(self.auto_start_robot)
        self.auto_start_timer.start(500)

        # Timer for joystick input
        self.joystick_timer = QTimer()
        self.joystick_timer.timeout.connect(self.joystick_loop)
        self.joystick_timer.start(TIMER_INTERVAL)

        # Timer for checking joystick connection
        self.joystick_reconnect_timer = QTimer()
        self.joystick_reconnect_timer.timeout.connect(self.check_joystick_connection)
        self.joystick_reconnect_timer.start(JOYSTICK_CHECK_INTERVAL)

        # Timer for auto-connecting to the robot
        self.auto_connect_timer = QTimer()
        self.auto_connect_timer.timeout.connect(self.auto_start_robot)
        self.auto_connect_timer.start(CONNECTION_RETRY_INTERVAL)

    def _init_joystick(self) -> None:
        """Initialize joystick support"""
        try:
            pygame.init()
            pygame.joystick.init()
            self.joystick = None
        except Exception as e:
            self.log(f"[ERROR] Failed to initialize joystick support: {e}")
            self.joystick = None



    def init_joint_tab(self) -> None:
        """Initialize the joint control tab"""
        tab = QWidget()
        layout = QGridLayout()
        layout.setSpacing(8)

        self.joint_boxes = []
        for i in range(6):
            box = QWidget()
            box_layout = QHBoxLayout()
            box_layout.setSpacing(5)
            box_layout.setContentsMargins(2, 2, 2, 2)

            label = QLabel(f"J{i + 1}")
            label.setFixedWidth(20)
            label.setStyleSheet("color: white;")

            input_field = QLineEdit("0.000")
            input_field.setFixedWidth(70)
            input_field.returnPressed.connect(lambda idx=i: self.set_joint_from_input(idx))
            self.joint_inputs.append(input_field)

            left = QPushButton("◀")
            left.setFixedWidth(35)
            right = QPushButton("▶")
            right.setFixedWidth(35)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.setFixedHeight(20)
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: none;
                    height: 10px;
                    background: #1c1c1c;
                    border-radius: 5px;
                }
                QSlider::handle:horizontal {
                    background: #3c3c3c;
                    border: none;
                    width: 30px;
                    height: 30px;
                    margin: -10px 0;
                    border-radius: 15px;
                }
                QSlider::sub-page:horizontal,
                QSlider::add-page:horizontal {
                    background: transparent;
                }
            """)

            left.pressed.connect(partial(self.nudge_joint, i, -1))
            right.pressed.connect(partial(self.nudge_joint, i, 1))
            slider.sliderPressed.connect(partial(self.set_slider_active, self.joint_active, i, True))
            slider.sliderReleased.connect(partial(self.release_slider, self.joint_sliders, self.joint_active, i))

            self.joint_sliders.append(slider)

            box_layout.addWidget(label)
            box_layout.addWidget(input_field)
            box_layout.addWidget(left)
            box_layout.addWidget(slider)
            box_layout.addWidget(right)

            box.setLayout(box_layout)
            self.joint_boxes.append(box)
            layout.addWidget(box, i, 0)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Joint Jog")

    def init_cartesian_tab(self) -> None:
        """Initialize the cartesian control tab"""
        tab = QWidget()
        layout = QGridLayout()
        layout.setSpacing(8)

        axes = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        self.cart_boxes = []

        for i, axis in enumerate(axes):
            box = QWidget()
            box_layout = QHBoxLayout()
            box_layout.setSpacing(5)
            box_layout.setContentsMargins(2, 2, 2, 2)

            label = QLabel(axis)
            label.setFixedWidth(20)
            label.setStyleSheet("color: white;")

            input_field = QLineEdit("0.000")
            input_field.setFixedWidth(70)
            input_field.returnPressed.connect(lambda idx=i: self.set_cart_from_input(idx))
            self.cart_inputs.append(input_field)

            left = QPushButton("◀")
            left.setFixedWidth(35)
            right = QPushButton("▶")
            right.setFixedWidth(35)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            slider.setFixedHeight(20)

            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    border: none;
                    height: 10px;
                    background: #1c1c1c;
                    border-radius: 5px;
                }
                QSlider::handle:horizontal {
                    background: #3c3c3c;
                    border: none;
                    width: 30px;
                    height: 30px;
                    margin: -10px 0;
                    border-radius: 15px;
                }
                QSlider::sub-page:horizontal,
                QSlider::add-page:horizontal {
                    background: transparent;
                }
            """)

            left.pressed.connect(partial(self.nudge_cart, i, -1))
            right.pressed.connect(partial(self.nudge_cart, i, 1))
            slider.sliderPressed.connect(partial(self.set_slider_active, self.cart_active, i, True))
            slider.sliderReleased.connect(partial(self.release_slider, self.cart_sliders, self.cart_active, i))
            # Add valueChanged connection for cartesian sliders
            slider.valueChanged.connect(partial(self.cart_slider_changed, i))

            self.cart_sliders.append(slider)

            box_layout.addWidget(label)
            box_layout.addWidget(input_field)
            box_layout.addWidget(left)
            box_layout.addWidget(slider)
            box_layout.addWidget(right)

            box.setLayout(box_layout)
            self.cart_boxes.append(box)
            layout.addWidget(box, i, 0)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Cartesian Jog")

    def cart_slider_changed(self, axis_idx: int, value: int) -> None:
        """Handle cartesian slider value changes"""
        if value != 0:
            # Activate the slider when moved from center
            self.set_slider_active(self.cart_active, axis_idx, True)
        else:
            # Deactivate when returned to center
            self.set_slider_active(self.cart_active, axis_idx, False)

    def set_joint_from_input(self, idx: int) -> None:
        """Set joint position from input field value with forbidden zone protection"""
        if self.control_mode != "mouse":
            return

        try:
            current = self.robot.GetJoints()
            if not current:
                self.log("⚠️ Cannot get current joint positions")
                return

            target = float(self.joint_inputs[idx].text())
            min_lim, max_lim = JOINT_LIMITS[idx]

            if not (min_lim <= target <= max_lim):
                self.log(f"⚠️ Joint {idx + 1} out of limit.")
                return

            # Calculate the movement delta
            delta = target - current[idx]
            joint_deltas = [0.0] * 6
            joint_deltas[idx] = delta

            # Check if movement is safe
            if not self.is_movement_safe("joint", joint_deltas):
                # Reset input to current value
                self.joint_inputs[idx].setText(f"{current[idx]:.3f}")
                return

            current[idx] = target
            self.robot.MoveJoints(*current)
            QTimer.singleShot(500, self.update_joint_inputs)

        except ValueError:
            self.log(f"⚠️ Invalid joint value for J{idx+1}. Must be a number.")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def set_cart_from_input(self, idx: int) -> None:
        """Set cartesian position from input field value with forbidden zone protection"""
        if self.control_mode != "mouse":
            return

        try:
            current = self.robot.GetPose()
            if not current:
                self.log("⚠️ Cannot get current robot pose")
                return

            target = float(self.cart_inputs[idx].text())

            # Calculate the movement delta
            delta = target - current[idx]
            cart_deltas = [0.0] * 6
            cart_deltas[idx] = delta

            # Check if movement is safe
            if not self.is_movement_safe("cartesian", cart_deltas):
                # Reset input to current value
                self.cart_inputs[idx].setText(f"{current[idx]:.3f}")
                return

            current[idx] = target
            self.robot.MovePose(*current)
            QTimer.singleShot(500, self.update_pose_display)

        except ValueError:
            self.log(f"⚠️ Invalid value for axis {idx+1}. Must be a number.")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def disable_all_jogging(self) -> None:
        """Disable all jogging operations"""
        self.joint_active = [False] * 6
        self.cart_active = [False] * 6

    def joint_jog_loop(self) -> None:
        """Main loop for joint jogging with forbidden zone protection"""
        try:
            # Check robot error status
            status = self.robot.GetStatusRobot()
            if status.error_status:
                if not self._error_popup_shown:
                    self._error_popup_shown = True
                    self.show_error_popup("⛔ The robot is in error. Please reset using the red button.")
                    self.reset_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")

                self.disable_all_jogging()
                self.set_all_sliders_enabled(False)
                return

            # Process active joint sliders
            for i in range(6):
                if not self.joint_active[i]:
                    continue

                val = self.joint_sliders[i].value()
                if val == 0:
                    continue

                step = (abs(val) / SLIDER_RANGE) * self.joint_step
                step *= 1 if val > 0 else -1

                move = [0.0] * 6
                move[i] = step

                # Check if movement is safe before executing
                if not self.is_movement_safe("joint", move):
                    # Block the movement and reset slider
                    self.joint_active[i] = False
                    self.joint_sliders[i].blockSignals(True)
                    self.joint_sliders[i].setValue(0)
                    self.joint_sliders[i].blockSignals(False)
                    continue

                try:
                    self.robot.MoveJointsRel(*move)
                    joints = self.robot.GetJoints()
                    if not self.joint_inputs[i].hasFocus():
                        self.joint_inputs[i].setText(f"{joints[i]:.3f}")
                except Exception as e:
                    self.log(f"[ERROR] {e}")

        except Exception as e:
            self.log(f"[ERROR] joint_jog_loop: {e}")

    def release_slider(self, sliders: List[QSlider], states: List[bool], idx: int) -> None:
        """Release a slider and reset its value"""
        states[idx] = False

        # Snap back to center
        sliders[idx].blockSignals(True)
        sliders[idx].setValue(0)
        sliders[idx].blockSignals(False)

    def get_gripper_position(self) -> float:
        """Get the current gripper position"""
        try:
            event = self.robot.SendCustomCommand("GetGripper", expected_responses=[defs.MX_ST_GRIPPER_POSITION])
            event.wait(timeout=2)  # blocks until response is received

            # Now parse the result (use latest known value from robot)
            pos = self.robot.GetStatusRobot().gripper_position
            if pos is not None:
                return float(pos)
            else:
                self.log("[WARN] Gripper position unknown.")
                return 0.0
        except Exception as e:
            self.log(f"[ERROR] Getting gripper position: {e}")
            return 0.0

    def get_gripper_percent(self) -> int:
        """Get the current gripper position as a percentage"""
        try:
            event = self.robot.SendCustomCommand("GetGripper", expected_responses=["Gripper"])
            event.wait(timeout=2)

            # Fetch the last known gripper value from robot status log if available
            response = event.responses[0] if event.responses else None
            if response:
                val = int(''.join(filter(str.isdigit, response)))
                return max(0, min(100, val))
            else:
                self.log("⚠️ No response from gripper.")
                return 50
        except Exception as e:
            self.log(f"[ERROR] get_gripper_percent: {e}")
            return 50

    def cartesian_jog_loop(self) -> None:
        """Cartesian jog loop with proper movement handling"""
        try:
            # Skip if robot not ready
            if not self.robot or not self.robot.IsConnected():
                return

            # Process each axis
            for i in range(6):
                val = self.cart_sliders[i].value()
                if val == 0:
                    continue  # Skip inactive axes

                # Determine movement parameters
                direction = 1 if val > 0 else -1
                step_size = self.cart_step_mm if i < 3 else self.cart_step_deg
                move = [0.0] * 6
                move[i] = step_size * direction

                # Get current position for safety checks
                try:
                    current_pose = self.robot.GetPose()
                    if not current_pose:
                        continue
                except Exception as e:
                    self.log(f"Error getting pose: {e}")
                    continue

                # Check if already in forbidden zone
                if self.is_pose_in_forbidden_zone(current_pose):
                    self.log("⚠️ Movement blocked: In forbidden zone")
                    self.cart_sliders[i].setValue(0)
                    continue

                # Check if movement would enter forbidden zone
                new_pose = self.simulate_cartesian_movement(move)
                if new_pose and self.is_pose_in_forbidden_zone(new_pose):
                    self.log(f"❌ Movement blocked: Would enter forbidden zone (axis {i + 1})")
                    self.cart_sliders[i].setValue(0)
                    continue

                # Execute the movement
                try:
                    if i < 3:  # X, Y, Z - linear movement
                        self.robot.MoveLinRelWrf(*move)
                    else:  # Rx, Ry, Rz - rotational movement
                        self.robot.MoveLinRelWrf(*move)

                    # Update position display
                    new_pose = self.robot.GetPose()
                    if new_pose and not self.cart_inputs[i].hasFocus():
                        self.cart_inputs[i].setText(f"{new_pose[i]:.3f}")

                except Exception as e:
                    self.log(f"Movement error (axis {i + 1}): {e}")
                    self.cart_sliders[i].setValue(0)

        except Exception as e:
            self.log(f"Cartesian jog error: {e}")

    def handle_cart_slider_change(self, axis_idx: int, value: int) -> None:
        """Handle cartesian slider changes (call this when slider moves)"""
        if value != 0:
            self.cart_active[axis_idx] = True  # Mark axis as active
            self.cartesian_jog_loop()  # Trigger movement
        else:
            self.cart_active[axis_idx] = False  # Reset when slider returns to center

    def update_joint_and_pose_inputs(self) -> None:
        """Update both joint and pose displays"""
        self.update_joint_inputs()
        self.update_pose_display()

    def update_joint_inputs(self) -> None:
        """Update joint input fields with current values"""
        try:
            joints = self.robot.GetJoints()
            for i in range(6):
                if not self.joint_inputs[i].hasFocus():
                    self.joint_inputs[i].setText(f"{joints[i]:.3f}")
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] {e}")

    def update_gripper_label(self, opened: bool) -> None:
        """Update the gripper/vacuum status label based on tool type"""
        self.gripper_open = opened
        if self.is_vacuum_tool:
            if opened:
                self.gripper_label.setText("🔵 Vacuum: On")
                self.gripper_label.setStyleSheet("color: lightblue;")
            else:
                self.gripper_label.setText("⚫ Vacuum: Off")
                self.gripper_label.setStyleSheet("color: gray;")
        else:
            if opened:
                self.gripper_label.setText("🧲 Gripper: Open")
                self.gripper_label.setStyleSheet("color: lightgreen;")
            else:
                self.gripper_label.setText("🧲 Gripper: Closed")
                self.gripper_label.setStyleSheet("color: red;")

    def open_gripper(self) -> None:
        """Open the gripper"""
        try:
            self.robot.SendCustomCommand("GripperOpen")
            self.update_gripper_label(True)
            QTimer.singleShot(500, self.update_gripper_slider)
            self.log("Gripper opened.")
        except Exception as e:
            self.log(f"[ERROR] Gripper open: {e}")

    def close_gripper(self) -> None:
        """Close the gripper"""
        try:
            self.robot.SendCustomCommand("GripperClose")
            self.update_gripper_label(False)
            QTimer.singleShot(500, self.update_gripper_slider)
            self.log("Gripper closed.")
        except Exception as e:
            self.log(f"[ERROR] Gripper close: {e}")

    def update_pose_display(self) -> None:
        """Update cartesian pose input fields with current values"""
        try:
            pose = self.robot.GetPose()
            for i in range(6):
                if not self.cart_inputs[i].hasFocus():
                    self.cart_inputs[i].setText(f"{pose[i]:.3f}")
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] {e}")

    def nudge_joint(self, idx: int, direction: int) -> None:
        """Move a joint by one increment in the specified direction"""
        try:
            step = self.joint_step * direction
            move = [0] * 6
            move[idx] = step
            self.robot.MoveJointsRel(*move)
            self.update_joint_inputs()
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def nudge_cart(self, idx: int, direction: int) -> None:
        """Move in cartesian space by one increment in the specified direction"""
        try:
            step = self.cart_step_mm if idx < 3 else self.cart_step_deg
            move = [0] * 6
            move[idx] = step * direction
            self.robot.MoveLinRelWrf(*move)
            self.update_pose_display()
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def go_home(self) -> None:
        """Move the robot to home position (bypasses forbidden zone protection)"""
        try:
            # Home command should bypass forbidden zone protection
            # since it's a safety command to return to a known safe position
            self.log("Going home... (bypassing forbidden zone protection)")
            self.robot.MoveJoints(0, 0, 0, 0, 0, 0)
            QTimer.singleShot(1000, self.update_joint_and_pose_inputs)
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def set_slider_active(self, state_list: List[bool], idx: int, val: bool) -> None:
        """Set a slider's active state"""
        state_list[idx] = val

        # Ensure timer is active if needed
        if val and state_list is self.joint_active and not self.joint_timer.isActive():
            self.joint_timer.start(TIMER_INTERVAL)
        elif val and state_list is self.cart_active and not self.cart_timer.isActive():
            self.cart_timer.start(TIMER_INTERVAL)

    def set_velocity(self, v: int) -> None:
        """Set the robot velocity"""
        try:
            self.velocity_percent = v
            self.vel_input.setText(str(v))
            self.robot.SetJointVel(v)
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def connect_robot(self) -> None:
        """Connect to the robot"""
        try:
            self.robot.Connect(DEFAULT_ROBOT_IP)
            self.robot.SetMonitoringInterval(0.1)
            self.conn_label.setText("✅ Connected")
            self.conn_label.setStyleSheet("color: lightgreen;")
            self.log("Connected.")
            self.update_gripper_slider()

            QTimer.singleShot(200, self.check_error_state)
            # --- After connecting, check tool type and update buttons
            QTimer.singleShot(500, self.detect_tool_type)

        except Exception as e:
            self.conn_label.setText("❌ Disconnected")
            self.conn_label.setStyleSheet("color: red;")
            self.log(f"[ERROR] {e}")

    def activate_robot(self) -> None:
        """Activate the robot"""
        try:
            self.robot.ActivateRobot()
            self.robot.SetMonitoringInterval(0.1)
            self.set_velocity(self.velocity_percent)
            self.activate_label.setText("🟢 Activated")
            self.activate_label.setStyleSheet("color: lightgreen; font-weight: bold;")
            self.log("Activated.")

            QTimer.singleShot(200, self.check_error_state)
        except Exception as e:
            self.activate_label.setText("⛔ Inactive")
            self.activate_label.setStyleSheet("color: orange; text-decoration: underline; cursor: pointer;")
            self.log(f"[ERROR] {e}")

    def reset_error(self) -> None:
        """Reset robot error state"""
        try:
            if self.robot.GetStatusRobot().error_status:
                self.robot.ResetError()
                self.robot.ResumeMotion()
                self.log("✅ Error reset.")

                self._error_popup_shown = False
                self.reset_button.setStyleSheet("")
                self.set_all_sliders_enabled(True)

                self.joint_active = [False] * 6
                self.cart_active = [False] * 6
                self.rebind_slider_events()
            else:
                QMessageBox.information(self, "No Error", "ℹ️ No error is currently active.")
        except Exception as e:
            self.log(f"[ERROR] {e}")

    def log(self, msg: str):
        print(msg)  # This goes through ConsoleInterceptor

    def update_increment_from_slider(self, val: int) -> None:
        """Update increment value from slider position"""
        increment = val / 10.0  # 10 → 1.0, 15 → 1.5, etc.
        self.joint_step = increment
        self.cart_step_mm = increment
        self.cart_step_deg = increment
        self.inc_input.setText(f"{increment:.1f}")  # sync back to input box

    def adjust_velocity(self, delta: int) -> None:
        """Adjust velocity by the specified delta"""
        new_val = max(10, min(100, self.velocity_percent + delta))
        self.vel_slider.setValue(new_val)

    def manual_velocity_input(self) -> None:
        """Handle manual velocity input"""
        try:
            val = int(self.vel_input.text())
            val = max(10, min(100, val))  # Clamp to valid range
            self.vel_slider.setValue(val)
        except ValueError:
            self.log("Invalid velocity input. Must be a number between 10 and 100.")
            self.vel_input.setText(str(self.velocity_percent))  # Reset to current value

    def adjust_increment(self, delta: int) -> None:
        """Adjust increment by the specified delta"""
        val = self.inc_slider.value() + delta
        val = max(1, min(50, val))
        self.inc_slider.setValue(val)

    def manual_increment_input(self) -> None:
        """Handle manual increment input"""
        try:
            val = float(self.inc_input.text())
            val = max(0.1, min(5.0, val))  # Clamp to valid range
            self.inc_slider.setValue(int(val * 10))
        except ValueError:
            self.log("Invalid increment input. Must be a number between 0.1 and 5.0.")
            self.inc_input.setText(f"{self.joint_step:.1f}")  # Reset to current value

    def joystick_loop(self) -> None:
        """Main loop for joystick control"""
        if self.control_mode != "joystick":
            return

        try:
            pygame.event.pump()

            if self.joystick is None or not self.joystick.get_init():
                raise RuntimeError("Joystick not connected")

            x, y, z = self.joystick.get_axis(0), self.joystick.get_axis(1), self.joystick.get_axis(2)
            # Check if joystick has additional axes for RX, RY, RZ
            rx, ry, rz = 0, 0, 0
            if self.joystick.get_numaxes() >= 6:
                rx, ry, rz = self.joystick.get_axis(3), self.joystick.get_axis(4), self.joystick.get_axis(5)
            current_buttons = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]

            # Initialize button states if needed
            if not self.last_button_states:
                self.last_button_states = [0] * self.joystick.get_numbuttons()

            # Gripper toggle
            # Gripper/Vacuum toggle
            if current_buttons[0] and not self.last_button_states[0]:
                if self.is_vacuum_tool:
                    # Toggle vacuum state
                    self.gripper_open = not self.gripper_open
                    if self.gripper_open:
                        try:
                            self.robot.SetValveState(0, 1)
                            self.log("Vacuum ON (joystick)")
                        except Exception as e:
                            self.log(f"[ERROR] Vacuum ON (joystick): {e}")
                    else:
                        try:
                            self.robot.SetValveState(0, 0)
                            self.log("Vacuum OFF (joystick)")
                        except Exception as e:
                            self.log(f"[ERROR] Vacuum OFF (joystick): {e}")
                    self.update_gripper_label(self.gripper_open)
                else:
                    # Toggle gripper state
                    self.gripper_open = not self.gripper_open
                    cmd = "GripperOpen" if self.gripper_open else "GripperClose"
                    try:
                        self.robot.SendCustomCommand(cmd)
                        self.log(f"Gripper {'opened' if self.gripper_open else 'closed'} (joystick)")
                    except Exception as e:
                        self.log(f"[ERROR] Gripper toggle (joystick): {e}")
                    self.update_gripper_label(self.gripper_open)
                # Sync slider & value
                percent = 100 if self.gripper_open else 0
                self.gripper_slider.blockSignals(True)
                self.gripper_slider.setValue(percent)
                self.gripper_slider.blockSignals(False)
                self.gripper_value_label.setText(f"{percent}%")

            # Joint group switch
            if current_buttons[1] and not self.last_button_states[1]:
                self.joystick_joint_group = 1 - self.joystick_joint_group
                self.update_joint_highlights()
                self.log(f"🎛️ Now controlling Joints {4 if self.joystick_joint_group else 1}–{6 if self.joystick_joint_group else 3}")

            self.last_button_states = current_buttons

            # Apply deadzone to joystick axes
            deadzone = 0.1
            apply_deadzone = lambda v: v if abs(v) > deadzone else 0.0
            x, y, z = map(apply_deadzone, [x, y, z])
            rx, ry, rz = map(apply_deadzone, [rx, ry, rz])

            if self.joystick_submode == "cartesian":
                self._handle_cartesian_joystick(x, y, z, rx, ry, rz)
            else:  # joint mode
                self._handle_joint_joystick(x, y, z)

        except Exception as e:
            # Don't log disconnection errors repeatedly
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] joystick_loop: {e}")
            self.joystick = None

    def _handle_cartesian_joystick(self, x: float, y: float, z: float, rx: float = 0, ry: float = 0, rz: float = 0) -> None:
        """Handle joystick input in cartesian mode with proactive forbidden zone protection"""
        self.update_cartesian_highlights()

        # Use all 6 axes directly for cartesian movement
        deltas = [
            self.cart_step_mm * -x,  # X (inverted to fix direction)
            self.cart_step_mm * y,   # Y
            self.cart_step_mm * z,   # Z
            self.cart_step_deg * rx, # RX
            self.cart_step_deg * ry, # RY
            self.cart_step_deg * rz  # RZ
        ]

        if any(d != 0 for d in deltas):
            # Check current position first
            current_pose = self.robot.GetPose()
            if current_pose and self.is_pose_in_forbidden_zone(current_pose):
                # Already in forbidden zone - block all movements
                for i in range(6):
                    self.cart_sliders[i].blockSignals(True)
                    self.cart_sliders[i].setValue(0)
                    self.cart_sliders[i].blockSignals(False)
                return

            # Check if movement is safe before executing
            if not self.is_movement_safe("cartesian", deltas):
                # Reset all sliders to center when movement is blocked
                for i in range(6):
                    self.cart_sliders[i].blockSignals(True)
                    self.cart_sliders[i].setValue(0)
                    self.cart_sliders[i].blockSignals(False)
                return

            # Additional check: simulate the movement
            test_pose = self.simulate_cartesian_movement(deltas)
            if test_pose and self.is_pose_in_forbidden_zone(test_pose):
                # Would enter forbidden zone - block movement
                for i in range(6):
                    self.cart_sliders[i].blockSignals(True)
                    self.cart_sliders[i].setValue(0)
                    self.cart_sliders[i].blockSignals(False)
                self.log("❌ BLOCKED: Joystick cartesian movement would enter forbidden zone")
                return

            try:
                self.robot.MoveLinRelWrf(*deltas)
                pose = self.robot.GetPose()
                if pose:
                    for i in range(6):
                        if not self.cart_inputs[i].hasFocus():
                            self.cart_inputs[i].setText(f"{pose[i]:.3f}")
                        # Reflect joystick input on slider visually
                        scale = self.cart_step_mm if i < 3 else self.cart_step_deg
                        raw_val = deltas[i] / scale if scale != 0 else 0
                        slider_val = int(raw_val * SLIDER_RANGE)
                        self.cart_sliders[i].blockSignals(True)
                        self.cart_sliders[i].setValue(slider_val)
                        self.cart_sliders[i].blockSignals(False)
            except Exception as e:
                self.log(f"[ERROR] Cartesian joystick: {e}")
        else:
            # Reset all sliders to center when no movement
            for i in range(6):
                self.cart_sliders[i].blockSignals(True)
                self.cart_sliders[i].setValue(0)
                self.cart_sliders[i].blockSignals(False)

    def _handle_joint_joystick(self, x: float, y: float, z: float) -> None:
        """Handle joystick input in joint mode with forbidden zone protection"""
        base = 3 if self.joystick_joint_group else 0
        for i, axis_val in enumerate([x, y, z]):
            joint_idx = base + i
            move = axis_val * self.joint_step
            if move != 0:
                # Create joint movement delta
                rel = [0.0] * 6
                rel[joint_idx] = move

                # Check if movement is safe before executing
                if not self.is_movement_safe("joint", rel):
                    # Reset slider for this joint
                    self.joint_sliders[joint_idx].blockSignals(True)
                    self.joint_sliders[joint_idx].setValue(0)
                    self.joint_sliders[joint_idx].blockSignals(False)
                    continue

                try:
                    self.robot.MoveJointsRel(*rel)
                    joints = self.robot.GetJoints()
                    if joints and not self.joint_inputs[joint_idx].hasFocus():
                        self.joint_inputs[joint_idx].setText(f"{joints[joint_idx]:.3f}")
                    slider_val = int(axis_val * SLIDER_RANGE)
                    self.joint_sliders[joint_idx].blockSignals(True)
                    self.joint_sliders[joint_idx].setValue(slider_val)
                    self.joint_sliders[joint_idx].blockSignals(False)
                except Exception as e:
                    self.log(f"[ERROR] Joint joystick: {e}")
            else:
                self.joint_sliders[joint_idx].blockSignals(True)
                self.joint_sliders[joint_idx].setValue(0)
                self.joint_sliders[joint_idx].blockSignals(False)

    def update_joystick_status_label(self) -> None:
        """Update the joystick status label"""
        if self.control_mode == "mouse":
            self.mode_label.setText("🖱️ Mouse Mode")
        else:
            if self.joystick is None:
                self.mode_label.setText(f"🕹️ Joystick: {self.joystick_submode.upper()} ❌")
            else:
                self.mode_label.setText(f"🕹️ Joystick: {self.joystick_submode.upper()} ✅")

    def update_joint_highlights(self) -> None:
        """Update joint control highlighting based on current mode"""
        for i, box in enumerate(self.joint_boxes):
            if self.control_mode == "joystick":
                active = (self.joystick_joint_group == 0 and i < 3) or (self.joystick_joint_group == 1 and i >= 3)
                if active:
                    box.setStyleSheet("background-color: #333355;")
                else:
                    box.setStyleSheet("background-color: none;")
            else:
                box.setStyleSheet("background-color: none;")

    def update_cartesian_highlights(self) -> None:
        """Update cartesian control highlighting based on current mode"""
        for i, box in enumerate(self.cart_boxes):
            if self.control_mode == "joystick":
                active = (self.joystick_joint_group == 0 and i < 3) or (self.joystick_joint_group == 1 and i >= 3)
                if active:
                    box.setStyleSheet("background-color: #333355;")
                else:
                    box.setStyleSheet("background-color: none;")
            else:
                box.setStyleSheet("background-color: none;")

    def set_control_mode(self, mode: str) -> None:
        """Set the control mode (mouse or joystick)"""
        self.control_mode = mode
        self.update_joint_highlights()

        if mode == "mouse":
            self.update_control_buttons()
            self.update_joystick_status_label()
            self.mode_label.setText("🖱️ Mouse Mode")
        else:
            self.joystick_submode = "joint"
            self.update_control_buttons()
            self.update_joystick_status_label()
            self.mode_label.setText(f"🕹️ Joystick: {self.joystick_submode.upper()}")

            # Auto-switch to correct tab
            self.tabs.setCurrentIndex(0 if self.joystick_submode == "joint" else 1)

    def toggle_joystick_mode(self) -> None:
        """Toggle between joystick modes"""
        if self.control_mode != "joystick":
            self.set_control_mode("joystick")
        else:
            self.joystick_submode = "cartesian" if self.joystick_submode == "joint" else "joint"
            self.update_control_buttons()
            self.mode_label.setText(f"🕹️ Joystick: {self.joystick_submode.upper()}")

            # Auto-switch to correct tab
            self.tabs.setCurrentIndex(0 if self.joystick_submode == "joint" else 1)

    def update_control_buttons(self) -> None:
        """Update control button styling based on current mode"""
        active_style = "background-color: lightgreen; color: black; font-weight: bold;"
        inactive_style = "background-color: #333; color: white;"

        if self.control_mode == "mouse":
            self.mouse_btn.setStyleSheet(active_style)
            self.joystick_btn.setStyleSheet(inactive_style)
            self.joystick_btn.setText("Joystick Mode")
        else:
            self.mouse_btn.setStyleSheet(inactive_style)
            self.joystick_btn.setStyleSheet(active_style)
            self.joystick_btn.setText(f"Joystick: {self.joystick_submode.upper()}")

        self.update_mouse_controls_enabled()

    def update_mouse_controls_enabled(self) -> None:
        """Update enabled state of mouse controls"""
        # Always allow buttons and sliders regardless of mode
        for i in range(6):
            self.joint_inputs[i].setEnabled(True)
            self.joint_sliders[i].setEnabled(True)
            self.cart_inputs[i].setEnabled(True)
            self.cart_sliders[i].setEnabled(True)

    def check_joystick_connection(self) -> None:
        """Check for joystick connection and update status"""
        try:
            pygame.joystick.quit()
            pygame.joystick.init()

            joystick_count = pygame.joystick.get_count()
            if joystick_count > 0:
                if self.joystick is None or not self.joystick.get_init():
                    self.joystick = pygame.joystick.Joystick(0)
                    self.joystick.init()
                    self.last_button_states = [0] * self.joystick.get_numbuttons()
                    if not self._joystick_was_connected:
                        self.log("🕹️ Joystick connected.")
                        self.log(f"🎮 {self.joystick.get_name()} | Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
                    self._joystick_was_connected = True
            else:
                if self.joystick is not None:
                    self.joystick = None
                if self._joystick_was_connected:
                    self.log("⚠️ Joystick disconnected.")
                    self._joystick_was_connected = False

            # Update joystick status in UI
            self.update_joystick_status_label()

        except Exception as e:
            if self._joystick_was_connected:
                self.log(f"⚠️ Joystick disconnected: {e}")
                self._joystick_was_connected = False
            self.joystick = None
            self.update_joystick_status_label()

    def force_activate_robot(self, event) -> None:
        """Force robot activation when status label is clicked"""
        if not self.robot.IsConnected():
            self.connect_robot()
        self.activate_robot()

    def check_robot_status(self) -> None:
        """Check robot status and update UI accordingly"""
        try:
            status = self.robot.GetStatusRobot()
            if status.error_status:
                if not self._error_popup_shown:
                    self._error_popup_shown = True
                    self.show_error_popup("⛔ The robot is in error. Please reset using the red button.")
                    self.reset_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                self.set_all_sliders_enabled(False)
                self.disable_all_jogging()
            else:
                self._error_popup_shown = False
                self.reset_button.setStyleSheet("")
                self.set_all_sliders_enabled(True)
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] check_robot_status: {e}")

    def check_robot_connection(self) -> None:
        """Check robot connection and update UI accordingly"""
        try:
            if not self.robot.IsConnected():
                self.conn_label.setText("❌ Disconnected")
                self.conn_label.setStyleSheet("color: red;")
                self.activate_label.setText("⛔ Inactive")
                self.activate_label.setStyleSheet("color: orange; text-decoration: underline; cursor: pointer;")
            else:
                self.conn_label.setText("✅ Connected")
                self.conn_label.setStyleSheet("color: lightgreen;")
                self.check_robot_status()
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] check_robot_connection: {e}")
            self.conn_label.setText("❌ Disconnected")
            self.conn_label.setStyleSheet("color: red;")
            self.activate_label.setText("⛔ Inactive")
            self.activate_label.setStyleSheet("color: orange; text-decoration: underline; cursor: pointer;")

    def auto_start_robot(self) -> None:
        """Attempt to automatically connect and activate the robot"""
        QTimer.singleShot(1000, self.update_joint_and_pose_inputs)
        QTimer.singleShot(1500, self.check_robot_status)

        # Check for robot error
        try:
            if self.robot.GetStatusRobot().error_status:
                self.log("⛔ Robot is in error. Please reset.")
                self.reset_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        except Exception:
            pass  # Ignore errors during auto-start

        # Initialize state tracking variables if needed
        if not hasattr(self, "_last_conn_status"):
            self._last_conn_status = None  # None means "first run"
        if not hasattr(self, "_last_activation_status"):
            self._last_activation_status = False
        if not hasattr(self, "_reconnect_message_shown"):
            self._reconnect_message_shown = False

        try:
            # Try to connect if not connected
            if not self.robot.IsConnected():
                if self._last_conn_status is True and not self._reconnect_message_shown:
                    self.log("🔌 Attempting to reconnect...")
                    self._reconnect_message_shown = True

                self.robot.Connect(DEFAULT_ROBOT_IP)
                self.robot.SetMonitoringInterval(0.1)
                self.conn_label.setText("✅ Connected")
                self.conn_label.setStyleSheet("color: lightgreen;")
                if self._last_conn_status is not True:
                    self.log("✅ Connected.")
                self._last_conn_status = True
                self._reconnect_message_shown = False
            else:
                self.conn_label.setText("✅ Connected")
                self.conn_label.setStyleSheet("color: lightgreen;")
                self._last_conn_status = True
                self._reconnect_message_shown = False

            # Try to activate if not activated
            status = self.robot.GetStatusRobot()
            if not status.activation_state and not self._last_activation_status:
                try:
                    self.robot.ActivateRobot()
                    self.set_velocity(self.velocity_percent)
                    self.activate_label.setText("🟢 Activated")
                    self.activate_label.setStyleSheet("color: lightgreen; font-weight: bold;")
                    self.log("🟢 Activated.")
                    self._last_activation_status = True
                except Exception:
                    pass  # don't spam errors
            elif status.activation_state:
                self._last_activation_status = True
                self.activate_label.setText("🟢 Activated")
                self.activate_label.setStyleSheet("color: lightgreen; font-weight: bold;")

            # Update gripper slider to known default or remembered value
            self.update_gripper_slider()

            # Schedule another update
            QTimer.singleShot(1000, self.update_joint_and_pose_inputs)

        except Exception as e:
            self._last_conn_status = False
            self._last_activation_status = False
            self.conn_label.setText("❌ Disconnected")
            self.conn_label.setStyleSheet("color: red;")
            self.activate_label.setText("⛔ Inactive")
            self.activate_label.setStyleSheet("color: orange; text-decoration: underline; cursor: pointer;")
            # Only log connection errors on first attempt
            if self._reconnect_message_shown:
                self.log(f"[ERROR] auto_start_robot: {e}")

    def update_gripper_slider(self) -> None:
        """Update gripper slider to match robot state"""
        try:
            status = self.robot.GetStatusRobot()
            if hasattr(status, "gripper_opened") and status.gripper_opened is not None:
                is_open = status.gripper_opened
                percent = 100 if is_open else 0
                self.gripper_slider.blockSignals(True)
                self.gripper_slider.setValue(percent)
                self.gripper_slider.blockSignals(False)
                self.gripper_value_label.setText(f"{percent}%")
                self.update_gripper_label(is_open)
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] update_gripper_slider: {e}")

    def check_forbidden_zone_status(self) -> None:
        """Check if robot is currently in a forbidden zone and warn user"""
        try:
            current_pose = self.robot.GetPose()
            if current_pose and self.is_pose_in_forbidden_zone(current_pose):
                self.log("⚠️ WARNING: Robot is currently in a forbidden zone!")
                self.log("   Manual movements are blocked until robot exits the zone.")
        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] check_forbidden_zone_status: {e}")

    def check_error_state(self) -> None:
        """Check for robot error state and update UI accordingly"""
        try:
            if self.robot.GetStatusRobot().error_status:
                if not self._error_popup_shown:
                    self._error_popup_shown = True
                    self.show_error_popup("⛔ The robot is in error. Please reset using the red button.")
                    self.reset_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")

                # Disable all sliders
                for slider in self.joint_sliders + self.cart_sliders:
                    slider.setEnabled(False)
                    slider.setValue(0)
            else:
                # Check forbidden zone status when robot is not in error
                self.check_forbidden_zone_status()

        except Exception as e:
            # Don't log connection errors during normal operation
            if "not connected" not in str(e).lower():
                self.log(f"[ERROR] check_error_state: {e}")

    def show_error_popup(self, message: str) -> None:
        """Show an error popup with the specified message"""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("⚠️ Robot Error")
        msg_box.setText(f"<b style='color:red'>{message}</b>")
        msg_box.setStyleSheet("font-size: 14px;")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def set_gripper_percent(self, val: int) -> None:
        """Set gripper opening percentage"""
        try:
            mm_opening = round((val / 100.0) * GRIPPER_MAX_OPENING, 2)  # Stay in 0–5.8 mm
            self.robot.MoveGripper(mm_opening)
            self.update_gripper_label(val > 50)
            self.gripper_value_label.setText(f"{val}%")
        except Exception as e:
            self.log(f"[ERROR] SetGripperPos: {e}")

    def set_all_sliders_enabled(self, enabled: bool) -> None:
        """Enable or disable all sliders"""
        for i, slider in enumerate(self.joint_sliders):
            slider.blockSignals(True)
            slider.setEnabled(enabled)
            if not enabled:
                slider.setValue(0)
            slider.blockSignals(False)

        for i, slider in enumerate(self.cart_sliders):
            slider.blockSignals(True)
            slider.setEnabled(enabled)
            if not enabled:
                slider.setValue(0)
            slider.blockSignals(False)

        # Force a full rebind AFTER enabling
        if enabled:
            self.rebind_slider_events()

    def rebind_slider_events(self) -> None:
        """Rebind all slider events"""
        for i, slider in enumerate(self.joint_sliders):
            try:
                slider.sliderPressed.disconnect()
            except Exception:
                pass
            try:
                slider.sliderReleased.disconnect()
            except Exception:
                pass
            slider.sliderPressed.connect(partial(self.set_slider_active, self.joint_active, i, True))
            slider.sliderReleased.connect(partial(self.release_slider, self.joint_sliders, self.joint_active, i))

        for i, slider in enumerate(self.cart_sliders):
            try:
                slider.sliderPressed.disconnect()
            except Exception:
                pass
            try:
                slider.sliderReleased.disconnect()
            except Exception:
                pass
            slider.sliderPressed.connect(partial(self.set_slider_active, self.cart_active, i, True))
            slider.sliderReleased.connect(partial(self.release_slider, self.cart_sliders, self.cart_active, i))

    def highlight_joint_group(self) -> None:
        """Highlight the active joint group"""
        self.update_joint_highlights()
        self.update_cartesian_highlights()


def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Use Fusion style for consistent cross-platform look

        # Set dark theme
        dark_palette = app.palette()
        dark_palette.setColor(dark_palette.ColorRole.Window, Qt.GlobalColor.black)
        dark_palette.setColor(dark_palette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(dark_palette.ColorRole.Base, Qt.GlobalColor.darkGray)
        dark_palette.setColor(dark_palette.ColorRole.AlternateBase, Qt.GlobalColor.darkGray)
        dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, Qt.GlobalColor.darkGray)
        dark_palette.setColor(dark_palette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(dark_palette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(dark_palette.ColorRole.Button, Qt.GlobalColor.darkGray)
        dark_palette.setColor(dark_palette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(dark_palette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(dark_palette.ColorRole.Link, Qt.GlobalColor.blue)
        dark_palette.setColor(dark_palette.ColorRole.Highlight, Qt.GlobalColor.darkBlue)
        dark_palette.setColor(dark_palette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        app.setPalette(dark_palette)

        # Set stylesheet for better appearance
        app.setStyleSheet("""
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:pressed {
                background-color: #222;
            }
            QLineEdit {
                background-color: #222;
                color: white;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                padding: 5px 10px;
            }
            QTabBar::tab:selected {
                background-color: #444;
                border-bottom: none;
            }
        """)

        window = MecaPendant()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()