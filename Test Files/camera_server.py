import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGridLayout, QMessageBox, QSizePolicy, QPushButton, QComboBox,
    QSpacerItem
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMetaObject, Q_ARG, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont
import time

# Configuration for camera detection and streaming
MAX_CAMERAS_TO_CHECK = 10  # Check up to 10 potential camera indices (0 to 9)
MAX_CONCURRENT_CAMERAS = 2  # Max number of camera streams to try and display simultaneously initially

# Define common OpenCV backend preferences for Windows
# You might need to adjust these based on your OS and camera types
OPENCV_BACKENDS = {
    "Default": cv2.CAP_ANY,  # Auto-detect (usually the default)
    "DirectShow": cv2.CAP_DSHOW,  # Good for older webcams on Windows
    "Media Foundation": cv2.CAP_MSMF,  # Newer backend for Windows
    "V4L2 (Linux)": cv2.CAP_V4L2,  # For Linux systems
    "AVFoundation (macOS)": cv2.CAP_AVFOUNDATION  # For macOS
}
DEFAULT_BACKEND_KEY = "Default"


class CameraWorker(QThread):
    """
    Worker thread to handle camera capture and emit frames.
    This prevents the UI from freezing while capturing video.
    """
    frame_ready = pyqtSignal(QImage)
    camera_error = pyqtSignal(int, str)  # Emits camera_id and error message
    camera_opened = pyqtSignal(int)  # Emits camera_id when successfully opened
    camera_stopped = pyqtSignal(int)  # Emits camera_id when stopped

    def __init__(self, camera_id: int, backend_api: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.backend_api = backend_api
        self.running = False
        self.cap = None
        self.width = 640  # Default width
        self.height = 480  # Default height
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10  # Increased reconnect attempts
        self.reconnect_delay_ms = 2000  # 2 seconds delay between reconnect attempts

    def _open_camera(self):
        """Attempts to open the camera with the specified backend and set properties."""
        if self.cap:
            self.cap.release()  # Release any existing capture
            self.cap = None

        try:
            self.cap = cv2.VideoCapture(self.camera_id, self.backend_api)
            if not self.cap.isOpened():
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Warm-up: try reading a frame to ensure it's actually streaming
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                self.cap = None
                return False
            return True
        except Exception as e:
            print(f"Error during _open_camera for {self.camera_id}: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False

    def run(self):
        """Main loop for the camera worker thread."""
        self.running = True
        try:
            if not self._open_camera():
                self.camera_error.emit(self.camera_id,
                                       f"Initial open failed for camera {self.camera_id}. Check if in use or accessible.")
                self.running = False
                self.camera_stopped.emit(self.camera_id)
                return

            self.camera_opened.emit(self.camera_id)
            self.reconnect_attempts = 0  # Reset reconnect attempts on successful open

            while self.running:
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        self.reconnect_attempts = 0  # Reset on successful frame read
                        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.frame_ready.emit(qt_image)
                    else:
                        # Frame not read, check if camera is still open or needs reconnection
                        if not self.cap.isOpened():
                            print(f"[CameraWorker {self.camera_id}] Camera lost, attempting to reconnect...")
                            self.camera_error.emit(self.camera_id, "Camera disconnected, attempting to reconnect...")
                            self.msleep(self.reconnect_delay_ms)
                            if not self._open_camera():
                                self.reconnect_attempts += 1
                                if self.reconnect_attempts >= self.max_reconnect_attempts:
                                    self.camera_error.emit(self.camera_id,
                                                           f"Failed to reconnect camera {self.camera_id} after multiple attempts.")
                                    self.running = False
                                    break  # Exit loop if max attempts reached
                        else:
                            # Camera is open but no frame, small delay to prevent busy-waiting
                            self.msleep(10)
                except cv2.error as e:
                    # Specific OpenCV errors (e.g., if camera suddenly becomes unavailable)
                    print(f"[CameraWorker {self.camera_id}] OpenCV error during read: {e}")
                    self.camera_error.emit(self.camera_id, f"Streaming error: {e}, attempting reconnection...")
                    self.msleep(self.reconnect_delay_ms)
                    if not self._open_camera():
                        self.reconnect_attempts += 1
                        if self.reconnect_attempts >= self.max_reconnect_attempts:
                            self.camera_error.emit(self.camera_id,
                                                   f"Failed to reconnect camera {self.camera_id} after streaming error.")
                            self.running = False
                            break
                except Exception as e:
                    # General unexpected errors
                    self.camera_error.emit(self.camera_id, f"An unexpected error occurred: {e}")
                    self.running = False
                    break  # Exit loop on unhandled exception

        except Exception as e:
            self.camera_error.emit(self.camera_id, f"An unhandled error occurred during camera worker startup: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.running = False
            self.camera_stopped.emit(self.camera_id)  # Signal that this worker has fully stopped

    def stop(self):
        """Stop the camera worker thread."""
        self.running = False
        self.wait()  # Wait for the thread to finish gracefully


class CameraFeedWidget(QWidget):  # Changed from QLabel to QWidget to allow more complex layout
    """
    Widget to display a single camera feed with controls.
    It manages its own CameraWorker thread.
    """

    def __init__(self, camera_id: int, backend_api: int, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.backend_api = backend_api
        self.worker = None
        self.is_streaming = False
        self.init_ui()

    def init_ui(self):
        """Initialize the UI for the camera feed widget."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        self.setLayout(main_layout)

        # Camera ID Label
        self.id_label = QLabel(f"Camera {self.camera_id}")
        self.id_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.id_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.id_label.setStyleSheet("color: #ADD8E6;")  # Light blue for ID
        main_layout.addWidget(self.id_label)

        # Video feed QLabel
        self.video_label = QLabel("Initializing...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(320, 240)  # Standard 4:3 aspect ratio minimum
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #ccc;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        self.video_label.setFont(QFont("Arial", 14))
        main_layout.addWidget(self.video_label)

        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 5, 0, 0)  # Top margin for separation

        self.toggle_button = QPushButton("Start Stream")
        self.toggle_button.clicked.connect(self.toggle_stream)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green for Start */
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        control_layout.addWidget(self.toggle_button)

        main_layout.addLayout(control_layout)

        # Initial state setup
        self.update_ui_for_stopped_stream()

    def start_feed(self):
        """Start the camera capture thread."""
        if self.is_streaming:
            return

        self.video_label.setText(f"Camera {self.camera_id}\n(Starting stream...)")
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #ccc;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
            }
        """)

        if self.worker:
            self.worker.stop()
            self.worker.deleteLater()  # Clean up old worker

        self.worker = CameraWorker(self.camera_id, self.backend_api)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.camera_error.connect(self.handle_error)
        self.worker.camera_opened.connect(self.on_camera_opened)
        self.worker.camera_stopped.connect(self.on_camera_stopped)
        self.worker.start()
        self.is_streaming = True
        self.update_ui_for_running_stream()

    def stop_feed(self):
        """Stop the camera feed and its worker thread."""
        if not self.is_streaming:
            return

        if self.worker:
            self.worker.stop()
            self.worker = None
        self.is_streaming = False
        self.update_ui_for_stopped_stream()
        print(f"Camera {self.camera_id} stream stopped.")

    def toggle_stream(self):
        """Toggle between starting and stopping the camera stream."""
        if self.is_streaming:
            self.stop_feed()
        else:
            self.start_feed()

    def update_frame(self, q_image: QImage):
        """Update the QLabel with the new frame."""
        # Scale the image to fit the label size while maintaining aspect ratio
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def handle_error(self, camera_id: int, message: str):
        """Display camera error message."""
        self.video_label.setText(f"Camera {camera_id}\n❌ Error: {message}")
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ff4444;
                border-radius: 8px;
                background-color: #331111;
                color: #ffaaaa;
                font-size: 14px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        # Forward the error to the parent app for centralized logging/status update
        if isinstance(self.parent(), CameraViewerApp):
            self.parent().log_error(f"[Camera {camera_id} ERROR]: {message}")
        self.is_streaming = False  # Mark as not streaming due to error
        self.update_ui_for_stopped_stream()

    def on_camera_opened(self, camera_id: int):
        """Called when a camera is successfully opened."""
        print(f"Camera {camera_id} opened successfully and streaming.")
        self.video_label.setText("")  # Clear initial text once frame starts coming
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #228b22; /* Forest green border for active camera */
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #ccc;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        self.is_streaming = True
        self.update_ui_for_running_stream()
        # Inform the parent app that a camera is now active
        if isinstance(self.parent(), CameraViewerApp):
            self.parent().camera_active_status_changed(camera_id, True)

    def on_camera_stopped(self, camera_id: int):
        """Called when a camera worker thread stops."""
        print(f"Camera {camera_id} worker thread stopped.")
        self.is_streaming = False
        self.update_ui_for_stopped_stream()
        if isinstance(self.parent(), CameraViewerApp):
            self.parent().camera_active_status_changed(camera_id, False)

    def update_ui_for_running_stream(self):
        """Updates the UI to reflect a running stream."""
        self.toggle_button.setText("Stop Stream")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #CC0000; /* Red for Stop */
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #B20000;
            }
            QPushButton:pressed {
                background-color: #990000;
            }
        """)
        # Show active status on the border of the widget container
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #228b22; /* Green border for active widget */
                border-radius: 10px;
                background-color: #2c2c2c;
            }
        """)

    def update_ui_for_stopped_stream(self):
        """Updates the UI to reflect a stopped stream."""
        self.toggle_button.setText("Start Stream")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green for Start */
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        self.video_label.setPixmap(QPixmap())  # Clear any residual image
        self.video_label.setText(f"Camera {self.camera_id}\n(Stream stopped)")
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 8px;
                background-color: #1e1e1e;
                color: #ccc;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #555; /* Gray border for inactive widget */
                border-radius: 10px;
                background-color: #2c2c2c;
            }
        """)


class CameraViewerApp(QWidget):
    """Main application window for displaying multiple camera feeds."""

    def __init__(self):
        super().__init__()
        self.camera_feeds = {}  # Use dict to store CameraFeedWidget instances by camera_id
        self.active_cameras_count = 0
        self.init_ui()
        self.find_and_start_cameras()

    def init_ui(self):
        """Initialize the main application UI."""
        self.setWindowTitle("Multi-Camera Viewer")
        self.setGeometry(100, 100, 1200, 800)  # Default window size
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #2c2c2c; /* Dark background */
                color: #f0f0f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #4CAF50; /* Green */
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                border: none;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
            QMessageBox {
                background-color: #2c2c2c;
                color: #f0f0f0;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: #007acc;
                color: white;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QComboBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAALCAYAAABaGz3uAAAABmJLR0QA/wD/AP+AdzgyAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAFiUAABYlAUlSJPAAAAB+SURBVCjPY2AYBfQxMDDQxcDAwMChpYGJkYfBQAaGhoYsDAx/sTAxMqgBEGZhYGCQZmBgYGQgA6gGkGagwUAmgGQYmBhqYGhgYGFgyMjAwgQjXjAwMBQyMDAwMNiBkYH/MTAwgJGBgaFBhoGBgaHBQAYgXGBiYGCigAAAF3tq3Jq+0fIAAAAASUVORK5CYII=); /* Example tiny base64 encoded down arrow */
                width: 10px;
                height: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                color: #ffffff;
                selection-background-color: #007acc;
            }
        """)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Title Label
        title_label = QLabel("Connected Cameras")
        title_label.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)

        # Controls Layout (Refresh, Backend Selector)
        top_controls_layout = QHBoxLayout()
        top_controls_layout.setContentsMargins(0, 0, 0, 10)

        self.refresh_button = QPushButton("Refresh Cameras")
        self.refresh_button.clicked.connect(self.refresh_cameras)
        top_controls_layout.addWidget(self.refresh_button)

        top_controls_layout.addSpacerItem(
            QSpacerItem(20, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))  # Spacer

        backend_label = QLabel("OpenCV Backend:")
        backend_label.setStyleSheet("font-weight: bold;")
        top_controls_layout.addWidget(backend_label)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(OPENCV_BACKENDS.keys())
        self.backend_combo.setCurrentText(DEFAULT_BACKEND_KEY)
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        top_controls_layout.addWidget(self.backend_combo)

        self.main_layout.addLayout(top_controls_layout)

        # Grid layout for camera feeds
        self.camera_grid_widget = QWidget()
        self.camera_grid_layout = QGridLayout(self.camera_grid_widget)
        self.camera_grid_layout.setSpacing(15)  # Spacing between camera feeds
        self.main_layout.addWidget(self.camera_grid_widget, stretch=1)

        # Status bar
        self.status_label = QLabel("Detecting cameras...")
        self.status_label.setStyleSheet("color: #aaaaaa; font-style: italic; margin-top: 10px;")
        self.main_layout.addWidget(self.status_label)

    def on_backend_changed(self, index: int):
        """Handle selection of a new OpenCV backend."""
        selected_backend_name = self.backend_combo.currentText()
        print(f"OpenCV backend changed to: {selected_backend_name}")
        self.refresh_cameras()  # Re-detect cameras with the new backend

    def find_and_start_cameras(self):
        """
        Attempts to find all connected cameras and start their feeds.
        Iterates through common camera indices as a heuristic.
        Only starts MAX_CONCURRENT_CAMERAS feeds initially.
        """
        self.status_label.setText("Searching for cameras...")
        self.active_cameras_count = 0
        self.available_camera_ids = []  # Store IDs of cameras that opened successfully in detection phase

        # Clear existing feeds
        for camera_id, feed_widget in list(self.camera_feeds.items()):
            feed_widget.stop_feed()
            self.camera_grid_layout.removeWidget(feed_widget)
            feed_widget.deleteLater()
            del self.camera_feeds[camera_id]

        selected_backend_api = OPENCV_BACKENDS[self.backend_combo.currentText()]

        print(f"Starting camera detection with backend: {self.backend_combo.currentText()}")
        for i in range(MAX_CAMERAS_TO_CHECK):
            temp_cap = None
            try:
                # Try to open the camera with the selected backend
                temp_cap = cv2.VideoCapture(i, selected_backend_api)
                if temp_cap.isOpened():
                    # Attempt to read a frame to confirm it's a working camera
                    ret, test_frame = temp_cap.read()
                    if ret:
                        print(f"Found and validated camera at index: {i} (Backend: {self.backend_combo.currentText()})")
                        self.available_camera_ids.append(i)
                    else:
                        print(
                            f"Camera at index {i} opened but failed to read a frame (Backend: {self.backend_combo.currentText()}).")
                else:
                    print(f"Camera at index {i} could not be opened (Backend: {self.backend_combo.currentText()}).")

            except Exception as e:
                print(f"Error checking camera {i}: {e}")
            finally:
                if temp_cap:
                    temp_cap.release()

            # Short delay between camera checks to prevent resource contention during detection
            QThread.msleep(100)  # Sleep for 100 milliseconds

        if len(self.available_camera_ids) == 0:
            self.status_label.setText(
                "No cameras detected. Please ensure cameras are connected and drivers are installed.")
            QMessageBox.warning(self, "No Cameras Found",
                                "No cameras were detected on your system.\n"
                                "Please check if cameras are properly connected and drivers are installed, "
                                "or try a different OpenCV backend.")
        else:
            self.status_label.setText(
                f"Detected {len(self.available_camera_ids)} camera(s). Starting streams for up to {MAX_CONCURRENT_CAMERAS}...")
            # Start feeds for a limited number of cameras
            for idx, camera_id in enumerate(self.available_camera_ids):
                camera_feed_widget = CameraFeedWidget(camera_id, selected_backend_api, self)
                self.camera_feeds[camera_id] = camera_feed_widget

                # Add to grid layout (dynamic arrangement)
                row = idx // 2  # 2 cameras per row
                col = idx % 2
                self.camera_grid_layout.addWidget(camera_feed_widget, row, col)

                # If we're within the concurrent limit, start the feed immediately
                if idx < MAX_CONCURRENT_CAMERAS:
                    camera_feed_widget.start_feed()
                    # Add a small delay after starting each feed
                    QThread.msleep(500)  # Sleep for 500 milliseconds

            self.update_status_label()

    def refresh_cameras(self):
        """Clears current camera feeds and restarts detection."""
        print("Refreshing camera list...")
        self.find_and_start_cameras()

    def log_error(self, message: str):
        """Logs errors to the console and potentially updates status bar."""
        print(message)
        self.status_label.setText(f"Error: {message}")

    def camera_active_status_changed(self, camera_id: int, is_active: bool):
        """Updates the count of active cameras and the status label."""
        if is_active:
            self.active_cameras_count += 1
        else:
            self.active_cameras_count = max(0, self.active_cameras_count - 1)  # Ensure not negative

        self.update_status_label()

    def update_status_label(self):
        """Updates the main status label with current camera counts."""
        total_detected = len(self.available_camera_ids)
        if total_detected == 0:
            self.status_label.setText("No cameras detected.")
        elif self.active_cameras_count == 0 and total_detected > 0:
            self.status_label.setText(f"Detected {total_detected} camera(s). No streams active.")
        else:
            self.status_label.setText(
                f"Streaming from {self.active_cameras_count} of {total_detected} detected camera(s).")

    def closeEvent(self, event):
        """Ensure all camera threads are stopped when the application closes."""
        for feed_widget in self.camera_feeds.values():
            feed_widget.stop_feed()
        super().closeEvent(event)


def main():
    """Main entry point for the camera viewer application."""
    app = QApplication(sys.argv)
    # Apply a dark fusion style for better aesthetics
    app.setStyle('Fusion')
    viewer = CameraViewerApp()
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
