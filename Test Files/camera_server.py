import cv2
import time
import threading
import numpy as np
import math

# Configuration for camera detection and streaming
MAX_CAMERAS_TO_CHECK = 4  # We are aiming for a 2x2 grid, so check up to 4 potential camera indices (0, 1, 2, 3)

# Configuration for testing a single camera
ENABLE_SINGLE_CAMERA_TESTING = False  # Set to True to test only one specific camera
TEST_SINGLE_CAMERA_ID = 0  # Change this to 0, 1, 2, 3 to test different cameras

# Desired resolution for individual camera streams (HD)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Desired size for each sub-frame in the composite window
# To achieve "max" quality, this now matches the CAMERA_WIDTH/HEIGHT
SUB_FRAME_DISPLAY_WIDTH = CAMERA_WIDTH
SUB_FRAME_DISPLAY_HEIGHT = CAMERA_HEIGHT

# Dictionary to hold the latest frame from each active camera
# Protected by a lock for thread-safe access
latest_frames_buffer = {}
buffer_lock = threading.Lock()

# List to hold camera worker threads
camera_threads = []
running = True  # Global flag to control threads and main loop

# Global state for UI view mode and maximized camera
current_maximized_camera_id = None  # Stores the 0-indexed ID of the maximized camera
in_maximized_view = False
composite_window_name = "Multi-Camera Feed"  # Name of the single OpenCV window


def create_text_overlay(frame, text, color=(255, 255, 255)):
    """Adds text to a frame, centered, with a small border for visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    # Add a black outline for better readability on varying backgrounds
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    return frame


def create_placeholder_frame(width: int, height: int, text: str, border_color=(50, 50, 50)):
    """Creates a black frame with white text and a border for placeholder display."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a border
    border_thickness = 5
    frame = cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, border_thickness)

    return create_text_overlay(frame, text, (200, 200, 200))  # Lighter text for placeholders


def capture_and_buffer(camera_id: int, backend_api: int = cv2.CAP_ANY):
    """
    Captures video frames from a specific camera and places them in a shared buffer.
    Runs in its own thread. Does NOT display anything directly.
    """
    global running
    cap = None

    print(f"Attempting to open camera {camera_id} with backend {backend_api}...")
    try:
        cap = cv2.VideoCapture(camera_id, backend_api)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_id}. It might be in use or not accessible.")
            with buffer_lock:
                latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                           f"Camera {camera_id + 1}\nX Failed to Open",
                                                                           border_color=(0, 0, 200))
            return

        # Attempt to set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Camera {camera_id} opened. Requested: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, Actual: {actual_width}x{actual_height}")

        # Warm-up: read a few frames to stabilize the stream
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                print(f"Camera {camera_id} failed to read warm-up frames. Releasing.")
                cap.release()
                cap = None
                with buffer_lock:
                    latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                               f"Camera {camera_id + 1}\nNo Frames",
                                                                               border_color=(0, 0, 200))
                return

        with buffer_lock:
            latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                       f"Camera {camera_id + 1}\nLoading...")  # Initial placeholder

        while running and cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Add "Camera X" label directly on the frame for consistency with image
                frame = create_text_overlay(frame, f"Camera {camera_id + 1}")
                with buffer_lock:
                    latest_frames_buffer[camera_id] = frame
            else:
                # Frame not read, check if camera is still open or needs reconnection
                if not cap.isOpened():
                    print(f"Camera {camera_id} lost, attempting to re-open...")
                    with buffer_lock:
                        latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                   f"Camera {camera_id + 1}\nReconnecting...",
                                                                                   border_color=(0, 100,
                                                                                                 200))  # Blue border for reconnect
                    time.sleep(2)  # Wait before retrying
                    cap.release()
                    cap = cv2.VideoCapture(camera_id, backend_api)
                    if not cap.isOpened():
                        print(f"Failed to re-open camera {camera_id}.")
                        with buffer_lock:
                            latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                       f"Camera {camera_id + 1}\nX Failed Reconnect",
                                                                                       border_color=(0, 0, 200))
                        break  # Exit loop if re-open fails
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        print(f"Camera {camera_id} re-opened successfully.")
                        continue  # Skip to next loop iteration to get a valid frame
                else:
                    # Camera is open but no frame, small delay to prevent busy-waiting
                    time.sleep(0.01)  # 10ms
            time.sleep(0.005)  # Small sleep to yield CPU and prevent busy-waiting

    except Exception as e:
        print(f"An error occurred with camera {camera_id}: {e}")
        with buffer_lock:
            latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                       f"Camera {camera_id + 1}\nERROR: {e}",
                                                                       border_color=(0, 0, 200))
    finally:
        if cap:
            cap.release()
        with buffer_lock:
            if camera_id in latest_frames_buffer:
                latest_frames_buffer[camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                           f"Camera {camera_id + 1}\nDisconnected",
                                                                           border_color=(0, 0, 200))
        print(f"Camera {camera_id} capture thread terminated.")


# Mouse callback function for the main OpenCV window
def on_mouse_click(event, x, y, flags, param):
    global current_maximized_camera_id, in_maximized_view, running

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_maximized_view:
            # If already maximized, click anywhere to return to grid view
            in_maximized_view = False
            current_maximized_camera_id = None
            print("Returning to grid view.")
        else:
            # In grid view, calculate which quadrant was clicked
            col = x // SUB_FRAME_DISPLAY_WIDTH
            row = y // SUB_FRAME_DISPLAY_HEIGHT

            clicked_index_in_grid = row * 2 + col  # For a 2x2 grid

            # Map grid index to potential_camera_indices (0-3)
            if 0 <= clicked_index_in_grid < MAX_CAMERAS_TO_CHECK:
                clicked_camera_id = clicked_index_in_grid  # Assuming our cameras are 0,1,2,3 for quadrants
                if clicked_camera_id in latest_frames_buffer:  # Ensure it's a camera we attempted to open
                    # Check if the camera is actually streaming (not just a placeholder for failed/no camera)
                    with buffer_lock:
                        current_frame_data = latest_frames_buffer.get(clicked_camera_id)
                        # Check if it's a default placeholder frame (meaning not actively streaming)
                        # This is a heuristic and might need refinement
                        if current_frame_data is not None and \
                                not (np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nX Failed to Open")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nNo Frames")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nLoading...")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nReconnecting...")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nX Failed Reconnect")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nERROR:")) or \
                                     np.array_equal(current_frame_data,
                                                    create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                             f"Camera {clicked_camera_id + 1}\nDisconnected"))):

                            current_maximized_camera_id = clicked_camera_id
                            in_maximized_view = True
                            print(f"Maximizing camera {clicked_camera_id + 1}")
                        else:
                            print(
                                f"Camera {clicked_camera_id + 1} is not streaming or failed to open. Cannot maximize.")
                else:
                    print(f"No active camera at this position ({clicked_index_in_grid}).")


def main():
    """
    Main function to detect cameras, start capture threads, and display a composite view.
    """
    global running, in_maximized_view, current_maximized_camera_id

    # We will specifically try to get cameras 0, 1, 2, 3 for the 2x2 grid
    potential_camera_indices = list(range(MAX_CAMERAS_TO_CHECK))  # [0, 1, 2, 3]

    detected_camera_ids = []

    # Using CAP_ANY for backend for basic OpenCV viewer
    selected_backend_api = cv2.CAP_ANY

    # --- Camera Detection ---
    if ENABLE_SINGLE_CAMERA_TESTING:
        print(f"Single camera testing mode: Checking Camera {TEST_SINGLE_CAMERA_ID}")
        temp_cap = cv2.VideoCapture(TEST_SINGLE_CAMERA_ID, selected_backend_api)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()  # Try to read a frame to validate
            if ret:
                detected_camera_ids.append(TEST_SINGLE_CAMERA_ID)
                print(f"Found and validated Camera {TEST_SINGLE_CAMERA_ID}")
            else:
                print(f"Camera {TEST_SINGLE_CAMERA_ID} opened but could not read a frame.")
            temp_cap.release()
        else:
            print(f"Camera {TEST_SINGLE_CAMERA_ID} could not be opened.")
        time.sleep(0.3)  # Short delay
    else:
        print("Auto-detecting specified camera indices for 2x2 grid (0-3)...")
        for i in potential_camera_indices:
            temp_cap = cv2.VideoCapture(i, selected_backend_api)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()  # Try to read a frame to validate
                if ret:
                    detected_camera_ids.append(i)
                    print(f"Found and validated Camera {i}")
                else:
                    print(f"Camera {i} opened but could not read a frame.")
                temp_cap.release()
            else:
                print(f"Camera {i} could not be opened.")
            time.sleep(0.3)  # Increased delay for detection

    if not detected_camera_ids and not ENABLE_SINGLE_CAMERA_TESTING:
        print("No cameras detected. Please ensure cameras are connected and drivers are installed.")
        input("Press Enter to exit...")  # Keep console open for user to read message
        return
    elif ENABLE_SINGLE_CAMERA_TESTING and not detected_camera_ids:
        print(f"Camera {TEST_SINGLE_CAMERA_ID} not detected or failed to open in single test mode.")
        input("Press Enter to exit...")
        return

    print(f"Starting capture threads for detected cameras: {detected_camera_ids}")

    # Initialize placeholders for all 4 quadrants, mapping them to camera IDs or "empty" status
    # This ensures a consistent 2x2 grid even if fewer than 4 cameras are found.

    # Map camera_ids to their display order (0 -> top-left, 1 -> top-right, etc.)
    display_quadrants = {}
    for i in range(MAX_CAMERAS_TO_CHECK):
        if i in detected_camera_ids:
            display_quadrants[i] = i  # Use actual camera ID if detected
        else:
            display_quadrants[i] = None  # Placeholder if not detected

    # Start a thread for each detected camera to capture frames
    for cam_id in detected_camera_ids:
        thread = threading.Thread(target=capture_and_buffer, args=(cam_id, selected_backend_api))
        thread.daemon = True  # Allow main program to exit even if threads are running
        thread.start()
        camera_threads.append(thread)
        time.sleep(0.5)  # Small delay between starting each camera thread

    # Give threads a moment to start capturing
    time.sleep(2)

    # Create the single main display window
    cv2.namedWindow(composite_window_name, cv2.WINDOW_NORMAL)
    # Set the mouse callback for the main window
    cv2.setMouseCallback(composite_window_name, on_mouse_click)

    # Main loop to keep the program running and display composite view
    while running:
        if in_maximized_view and current_maximized_camera_id is not None:
            # Display maximized single camera feed
            max_frame_to_display = None
            with buffer_lock:
                max_frame = latest_frames_buffer.get(current_maximized_camera_id)
                if max_frame is not None:
                    max_frame_to_display = max_frame
                else:
                    # If maximized camera's buffer is empty/failed, return to grid
                    print(f"Maximized camera {current_maximized_camera_id + 1} frame not available. Returning to grid.")
                    in_maximized_view = False
                    current_maximized_camera_id = None
                    max_frame_to_display = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                    f"Camera {current_maximized_camera_id + 1}\nError/No Data",
                                                                    border_color=(0, 0, 200))

            if max_frame_to_display is not None:
                cv2.imshow(composite_window_name, max_frame_to_display)
                cv2.resizeWindow(composite_window_name, CAMERA_WIDTH, CAMERA_HEIGHT)  # Resize window to single HD frame
        else:
            # Display composite grid view
            quadrant_frames = []
            for i in range(MAX_CAMERAS_TO_CHECK):  # Iterate through the 4 quadrants
                cam_id_for_quadrant = display_quadrants[i]

                frame_to_display = None
                if cam_id_for_quadrant is not None:
                    with buffer_lock:
                        frame = latest_frames_buffer.get(cam_id_for_quadrant)
                        if frame is not None:
                            # Frame is already at CAMERA_WIDTH/HEIGHT, no resizing needed here if matching SUB_FRAME_DISPLAY_WIDTH/HEIGHT
                            frame_to_display = frame
                        else:
                            # Fallback for detected but not yet streaming or errored
                            frame_to_display = create_placeholder_frame(SUB_FRAME_DISPLAY_WIDTH,
                                                                        SUB_FRAME_DISPLAY_HEIGHT,
                                                                        f"Camera {cam_id_for_quadrant + 1}\nLoading...")
                else:
                    # No camera assigned to this quadrant
                    if i == 0:  # Top-Left
                        label = "Camera 1"
                    elif i == 1:  # Top-Right
                        label = "Camera 2"
                    elif i == 2:  # Bottom-Left
                        label = "Camera 3"
                    elif i == 3:  # Bottom-Right
                        label = "Camera 4"

                    # Check if camera was detected but failed to open permanently or is disconnected
                    if i in detected_camera_ids and cam_id_for_quadrant is None:
                        frame_to_display = create_placeholder_frame(SUB_FRAME_DISPLAY_WIDTH, SUB_FRAME_DISPLAY_HEIGHT,
                                                                    f"{label}\nX Failed to open",
                                                                    border_color=(0, 0, 200))
                    else:
                        frame_to_display = create_placeholder_frame(SUB_FRAME_DISPLAY_WIDTH, SUB_FRAME_DISPLAY_HEIGHT,
                                                                    f"{label}\nNo Camera Here")

                quadrant_frames.append(frame_to_display)

            # Arrange into a 2x2 grid
            top_row = cv2.hconcat([quadrant_frames[0], quadrant_frames[1]])
            bottom_row = cv2.hconcat([quadrant_frames[2], quadrant_frames[3]])
            composite_image = cv2.vconcat([top_row, bottom_row])

            # Display the composite image
            cv2.imshow(composite_window_name, composite_image)
            cv2.resizeWindow(composite_window_name, SUB_FRAME_DISPLAY_WIDTH * 2,
                             SUB_FRAME_DISPLAY_HEIGHT * 2)  # Fixed 2x2 size

        # Check for 'q' key press globally, and also if window was closed via X button
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(composite_window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

        time.sleep(0.01)  # Small sleep to prevent busy-waiting

    print("Exiting application. Signalling threads to stop...")
    # Signal all threads to stop
    running = False
    for thread in camera_threads:
        thread.join(timeout=5)  # Wait for threads to finish gracefully

    print("Releasing cameras and closing windows...")
    cv2.destroyAllWindows()
    print("All cameras released and windows closed.")


if __name__ == "__main__":
    main()
