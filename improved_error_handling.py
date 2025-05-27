"""
Improved Error Handling for Meca500 Robot Control GUI
----------------------------------------------------
"""
import sys
import time
import threading
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QMessageBox, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QEventLoop
from PyQt6.QtGui import QFont, QIcon

class RobotErrorDialog(QDialog):
    """
    Custom error dialog for robot errors with Reset Error button and optional Home button.
    Non-blocking implementation to prevent UI freezes.
    """
    reset_clicked = pyqtSignal()
    home_clicked = pyqtSignal()

    def __init__(self, parent=None, error_message="", error_code="", show_home_button=False):
        super().__init__(parent)
        self.setWindowTitle("⚠️ Robot Error")
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint | Qt.WindowType.WindowStaysOnTopHint)

        # Make dialog non-modal to prevent blocking
        self.setModal(False)

        # Store error details
        self.error_message = error_message
        self.error_code = error_code

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Error icon and message
        error_layout = QHBoxLayout()

        error_icon = QLabel("⚠️")
        error_icon.setFont(QFont("Arial", 24))
        error_layout.addWidget(error_icon)

        message_layout = QVBoxLayout()
        error_title = QLabel("<b>Robot Error</b>")
        error_title.setFont(QFont("Arial", 12))
        message_layout.addWidget(error_title)

        error_text = QLabel(error_message)
        error_text.setWordWrap(True)
        message_layout.addWidget(error_text)

        if error_code:
            code_text = QLabel(f"<i>Error code: {error_code}</i>")
            code_text.setStyleSheet("color: #666;")
            message_layout.addWidget(code_text)

        error_layout.addLayout(message_layout, stretch=1)
        layout.addLayout(error_layout)

        # Don't show again checkbox
        self.dont_show_again = QCheckBox("Don't show this popup again (errors will still appear in console)")
        layout.addWidget(self.dont_show_again)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset Error")
        self.reset_button.setStyleSheet("background-color: #f80; color: white; font-weight: bold;")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        button_layout.addWidget(self.reset_button)

        if show_home_button:
            self.home_button = QPushButton("Home Robot")
            self.home_button.setStyleSheet("background-color: #26a; color: white; font-weight: bold;")
            self.home_button.clicked.connect(self._on_home_clicked)
            button_layout.addWidget(self.home_button)

        layout.addLayout(button_layout)

        # Auto-close timer to prevent orphaned dialogs
        self.close_timer = QTimer(self)
        self.close_timer.timeout.connect(self.close)
        self.close_timer.setSingleShot(True)
        self.close_timer.start(60000)  # Auto-close after 1 minute if forgotten

    def _on_reset_clicked(self):
        """Handle Reset Error button click"""
        # Disable button immediately to prevent double-clicks
        self.reset_button.setEnabled(False)
        self.reset_button.setText("Resetting...")

        # Emit signal in a non-blocking way
        QTimer.singleShot(0, self.reset_clicked.emit)

        # Close dialog after a short delay
        QTimer.singleShot(500, self.accept)

    def _on_home_clicked(self):
        """Handle Home Robot button click"""
        # Disable button immediately to prevent double-clicks
        self.home_button.setEnabled(False)
        self.home_button.setText("Homing...")

        # Emit signal in a non-blocking way
        QTimer.singleShot(0, self.home_clicked.emit)

        # Close dialog after a short delay
        QTimer.singleShot(500, self.accept)

    def should_suppress_future_popups(self):
        """Return whether future popups should be suppressed"""
        return self.dont_show_again.isChecked()


class SafeRobotCommand(QObject):
    """
    Thread-safe robot command executor that prevents UI freezes.
    """
    command_finished = pyqtSignal(bool, str)

    def __init__(self, robot, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.running = False

    def execute(self, command_name, *args, **kwargs):
        """
        Execute a robot command in a non-blocking way.

        Args:
            command_name: Name of the robot method to call
            *args, **kwargs: Arguments to pass to the method

        Returns:
            bool: True if command was started, False otherwise
        """
        if self.running:
            return False

        self.running = True

        # Start command in a separate thread
        threading.Thread(
            target=self._execute_command,
            args=(command_name, args, kwargs),
            daemon=True
        ).start()

        return True

    def _execute_command(self, command_name, args, kwargs):
        """Execute the command in a background thread"""
        result = False
        error_msg = ""

        try:
            # Get the method from the robot object
            method = getattr(self.robot, command_name)

            # Call the method with the provided arguments
            method(*args, **kwargs)
            result = True

        except Exception as e:
            error_msg = str(e)

        finally:
            self.running = False
            # Emit signal on the main thread
            self.command_finished.emit(result, error_msg)


class ErrorHandler:
    """
    Centralized error handling for the Meca500 robot control GUI.
    """
    def __init__(self, main_gui):
        self.main_gui = main_gui
        self.robot = main_gui.robot
        self.console = main_gui.console
        self.reset_button = main_gui.reset_button

        # Error state tracking
        self.error_popup_shown = False
        self.suppress_popups = False
        self.last_error_code = None
        self.last_error_time = 0
        self.reset_in_progress = False

        # Create safe command executor
        self.safe_command = SafeRobotCommand(self.robot)
        self.safe_command.command_finished.connect(self._on_command_finished)

        # Error debounce timer
        self.error_timer = QTimer()
        self.error_timer.setSingleShot(True)
        self.error_timer.timeout.connect(self._clear_error_state)

        # Dialog tracking
        self.active_dialogs = []

    def handle_error(self, error_message, error_code="", show_home_button=False):
        """
        Handle a robot error with appropriate UI feedback.

        Args:
            error_message: The error message to display
            error_code: Optional error code for reference
            show_home_button: Whether to show the Home Robot button
        """
        # Always log to console
        self._log_to_console(error_message, error_code)

        # Highlight reset button
        self.reset_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")

        # Show popup if not suppressed and not already showing an error
        if not self.suppress_popups and not self.error_popup_shown and not self.reset_in_progress:
            self.error_popup_shown = True
            self._show_error_dialog(error_message, error_code, show_home_button)

    def _show_error_dialog(self, error_message, error_code="", show_home_button=False):
        """Show the custom error dialog in a non-blocking way"""
        # Create dialog
        dialog = RobotErrorDialog(
            self.main_gui,
            error_message=error_message,
            error_code=error_code,
            show_home_button=show_home_button
        )

        # Connect signals using direct connections to prevent event loop issues
        dialog.reset_clicked.connect(self._reset_error, Qt.ConnectionType.QueuedConnection)
        if show_home_button:
            dialog.home_clicked.connect(self._home_robot, Qt.ConnectionType.QueuedConnection)

        # Track dialog to prevent garbage collection
        self.active_dialogs.append(dialog)
        dialog.finished.connect(lambda: self.active_dialogs.remove(dialog) if dialog in self.active_dialogs else None)

        # Show dialog non-modally
        dialog.show()

        # Check if future popups should be suppressed when dialog is closed
        dialog.finished.connect(
            lambda: setattr(self, 'suppress_popups', dialog.should_suppress_future_popups())
        )

        # Reset error popup shown flag when dialog is closed
        dialog.finished.connect(
            lambda: setattr(self, 'error_popup_shown', False)
        )

    def _log_to_console(self, error_message, error_code=""):
        """Log error to the console"""
        if error_code:
            self.main_gui.log(f"⚠️ ERROR ({error_code}): {error_message}")
        else:
            self.main_gui.log(f"⚠️ ERROR: {error_message}")

    def _reset_error(self):
        """Reset robot error state in a non-blocking way"""
        if self.reset_in_progress:
            return

        self.reset_in_progress = True
        self.main_gui.log("🔄 Resetting robot error...")

        # Disable reset button to prevent multiple clicks
        self.reset_button.setEnabled(False)

        try:
            # Execute reset sequence in a non-blocking way
            self.safe_command.execute("ResetError")

            # Schedule ResumeMotion after a delay
            QTimer.singleShot(1000, lambda: self.safe_command.execute("ResumeMotion"))

        except Exception as e:
            self.main_gui.log(f"[ERROR] Failed to reset error: {e}")
            self.reset_in_progress = False
            self.reset_button.setEnabled(True)

    def _on_command_finished(self, success, error_msg):
        """Handle completion of a safe command"""
        if not success and error_msg:
            self.main_gui.log(f"[ERROR] Command failed: {error_msg}")

        # Reset UI state after command completes
        self.reset_in_progress = False
        self.reset_button.setEnabled(True)
        self.reset_button.setStyleSheet("")

        # Update UI
        self.main_gui.set_all_sliders_enabled(True)

        # Reset jogging state
        self.main_gui.joint_active = [False] * 6
        self.main_gui.cart_active = [False] * 6
        self.main_gui.rebind_slider_events()

        # Log success if no error
        if success:
            self.main_gui.log("✅ Error reset complete.")

    def _home_robot(self):
        """Home the robot in a non-blocking way"""
        if self.reset_in_progress:
            return

        self.reset_in_progress = True
        self.main_gui.log("🏠 Homing robot...")

        try:
            # First reset any errors
            self.safe_command.execute("ResetError")

            # Schedule activation after a delay
            QTimer.singleShot(1000, lambda: self.safe_command.execute("ActivateRobot"))

            # Schedule homing after another delay
            QTimer.singleShot(2000, lambda: self.safe_command.execute("Home"))

        except Exception as e:
            self.main_gui.log(f"[ERROR] Failed to home robot: {e}")
            self.reset_in_progress = False

    def _clear_error_state(self):
        """Clear the error state after debounce period"""
        self.error_popup_shown = False
        self.last_error_code = None


def detect_error_type(error_message):
    """
    Detect the type of error from the error message.

    Returns:
        tuple: (error_type, show_home_button)
    """
    error_message = error_message.lower()

    # Only show home button for "not homed" errors
    if "not homed" in error_message or "mx_st_not_homed" in error_message:
        return "not_homed", True
    elif "out of reach" in error_message or "singularity" in error_message:
        return "singularity", False
    elif "limit" in error_message:
        return "limit", False
    elif "socket" in error_message or "connection" in error_message:
        return "connection", False
    else:
        return "general", False


# Modified patch_meca_pendant function
def patch_meca_pendant(MecaPendant):
    """
    Patch the MecaPendant class with improved error handling.

    This function modifies the MecaPendant class to use the improved error handling.
    """
    # Store the original show_error_popup method
    original_show_error_popup = MecaPendant.show_error_popup

    def new_show_error_popup(self, message):
        """
        Enhanced error popup with Reset Error button and improved error handling.
        """
        # Detect error type
        error_type, show_home_button = detect_error_type(message)

        # Extract error code if present
        error_code = ""
        if "(" in message and ")" in message:
            start = message.find("(")
            end = message.find(")")
            if start < end:
                error_code = message[start + 1:end]

        # Use the error handler
        if not hasattr(self, 'error_handler'):
            self.error_handler = ErrorHandler(self)

        # Use QTimer to make this non-blocking
        QTimer.singleShot(0, lambda: self.error_handler.handle_error(message, error_code, show_home_button))

    # Replace the method
    MecaPendant.show_error_popup = new_show_error_popup

    # Store original reset_error method
    original_reset_error = MecaPendant.reset_error if hasattr(MecaPendant, 'reset_error') else None

    def new_reset_error(self):
        """
        Enhanced reset_error method with better error handling.
        """
        # Use the error handler if available
        if hasattr(self, 'error_handler'):
            self.error_handler._reset_error()
        elif original_reset_error:
            original_reset_error(self)

    # Replace the method if it exists
    if original_reset_error:
        MecaPendant.reset_error = new_reset_error

    # We'll modify the instance creation instead of trying to patch ConsoleInterceptor directly
    original_init = MecaPendant.__init__

    def new_init(self, *args, **kwargs):
        # Call the original __init__
        original_init(self, *args, **kwargs)

        # Now patch the console interceptor that was created during initialization
        original_write = sys.stdout.write

        def new_write(msg):
            """
            Enhanced console interceptor that detects specific errors.
            """
            msg = msg.strip()
            if not msg:
                return

            # Call the original method
            original_write(msg + "\n")

            # Check for specific error patterns in a non-blocking way
            if "MX_ST_NOT_HOMED" in msg:
                # This is a "not homed" error, show special dialog
                if hasattr(self, 'error_handler'):
                    QTimer.singleShot(0, lambda: self.error_handler.handle_error(
                        "Robot is not homed. Please home the robot to continue.",
                        "MX_ST_NOT_HOMED",
                        True  # Show home button
                    ))
            elif "MX_ST_SINGULARITY" in msg:
                # This is a singularity error
                if hasattr(self, 'error_handler'):
                    QTimer.singleShot(0, lambda: self.error_handler.handle_error(
                        "Robot encountered a singularity. Try moving to a different position or homing the robot.",
                        "MX_ST_SINGULARITY",
                        False  # Don't show home button
                    ))
            elif "MX_ST_ALREADY_ERR" in msg:
                # Already in error state
                if hasattr(self, 'error_handler'):
                    QTimer.singleShot(0, lambda: self.error_handler.handle_error(
                        "Robot is already in error state. Please reset the error first.",
                        "MX_ST_ALREADY_ERR",
                        False  # Don't show home button
                    ))
            elif "socket" in msg.lower() or "connection" in msg.lower():
                # Connection issue
                if hasattr(self, 'error_handler'):
                    QTimer.singleShot(0, lambda: self.error_handler.handle_error(
                        "Connection to robot lost. Please check the connection and try again.",
                        "CONNECTION_ERROR",
                        False  # Don't show home button
                    ))

        if hasattr(sys.stdout, 'callback') and sys.stdout.callback == self.log:
            sys.stdout.write = new_write

    MecaPendant.__init__ = new_init

    return MecaPendant
