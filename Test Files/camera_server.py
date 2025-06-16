import cv2
import time
import threading
import numpy as np
import math

# --- Configuration ---
MAX_CAMERAS_TO_CHECK = 4  # We aim for a 2x2 grid, so check up to 4 potential camera indices (e.g., 0, 1, 2, 3, 4 etc.)

# Configuration for testing a single camera
ENABLE_SINGLE_CAMERA_TESTING = False  # Set to True to test only one specific camera
# IMPORTANT: When ENABLE_SINGLE_CAMERA_TESTING is True, TEST_SINGLE_CAMERA_ID should be 1, 2, 3, etc.
TEST_SINGLE_CAMERA_ID = 1  # Change this to 1, 2, 3, etc., to test different cameras

# Desired resolution for individual camera streams (HD)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Desired size for each sub-frame in the composite window
# To maintain "max" quality, this now matches the CAMERA_WIDTH/HEIGHT
SUB_FRAME_DISPLAY_WIDTH = CAMERA_WIDTH
SUB_FRAME_DISPLAY_HEIGHT = CAMERA_HEIGHT

# Define common OpenCV backend preferences for Windows (can be useful for troubleshooting)
# The order here is important for the iterative testing strategy
OPENCV_BACKENDS = [
    ("Default", cv2.CAP_ANY),  # Auto-detect (usually the default)
    ("Media Foundation", cv2.CAP_MSMF),  # Newer backend for Windows (sometimes more stable for multiple cams)
    ("DirectShow", cv2.CAP_DSHOW),  # Good for older webcams on Windows
]

# --- Global State ---
# Stores the latest frame from each active camera thread, keyed by its *internal OpenCV index* (e.g., 0, 1, 2, 3...)
latest_frames_buffer = {}
buffer_lock = threading.Lock()  # Protects access to latest_frames_buffer

camera_threads = []  # List to hold camera worker threads
running = True  # Global flag to control threads and main loop

# Stores the internal OpenCV ID of the currently maximized camera (NOT 1-based display ID)
current_maximized_camera_id = None
in_maximized_view = False  # Flag to indicate if a camera is maximized
composite_window_name = "Multi-Camera Feed"  # Name of the single OpenCV window


# --- Utility Functions ---
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


# --- Camera Capture Thread ---
def capture_and_buffer(internal_camera_id: int, backend_api: int):
    """
    Captures video frames from a specific camera (identified by its 0-based internal_camera_id)
    and places them in a shared buffer. Runs in its own thread.
    """
    global running
    cap = None

    # Use 1-based index for user-facing prints, but use internal_camera_id directly for VideoCapture
    print(
        f"Thread for Camera {internal_camera_id}: Attempting to open (internal index {internal_camera_id}) with backend {backend_api}...")

    total_open_attempts = 0
    max_total_open_attempts = 5  # Max attempts to fully open and warm up the camera

    while total_open_attempts < max_total_open_attempts and running:
        total_open_attempts += 1
        print(f"Thread for Camera {internal_camera_id}: Open attempt {total_open_attempts}/{max_total_open_attempts}")
        try:
            # Try to open the camera
            cap = cv2.VideoCapture(internal_camera_id, backend_api)
            if not cap.isOpened():
                print(f"Thread for Camera {internal_camera_id}: Failed to open VideoCapture. Retrying in 1s...")
                time.sleep(1)
                continue  # Try opening again if failed

            # Set desired resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            # Check if resolution was actually set (cameras might not support requested resolution)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width != CAMERA_WIDTH or actual_height != CAMERA_HEIGHT:
                print(
                    f"Thread for Camera {internal_camera_id}: Warning! Requested resolution {CAMERA_WIDTH}x{CAMERA_HEIGHT}, got {actual_width}x{actual_height}.")

            # Warm-up phase: try to read several frames to ensure stable stream
            warmup_frames_read = 0
            max_warmup_frames_to_read = 30  # Try to read more frames to ensure stability

            for i in range(max_warmup_frames_to_read):
                ret, _ = cap.read()
                if ret:
                    warmup_frames_read += 1
                else:
                    # Frame read failed during warm-up, might be temporary or a sign of issues
                    print(
                        f"Thread for Camera {internal_camera_id}: Warm-up frame {i + 1}/{max_warmup_frames_to_read} failed. Retrying frame read...")
                    time.sleep(0.05)  # Small pause

                # If we've successfully read a reasonable number of frames, consider warm-up complete
                if warmup_frames_read >= 5:  # Successfully read at least 5 frames
                    break

            if warmup_frames_read == 0:
                print(
                    f"Thread for Camera {internal_camera_id}: Failed to read any frames during warm-up. Releasing and retrying open...")
                cap.release()
                cap = None
                time.sleep(1)  # Give system a break before the next open attempt
                continue  # Go to next `total_open_attempts`

            print(f"Thread for Camera {internal_camera_id}: Warmed up successfully. Read {warmup_frames_read} frames.")
            break  # Camera successfully opened and warmed up, exit outer `while total_open_attempts` loop

        except Exception as e:
            # Catch general exceptions during open/warm-up
            print(f"Thread for Camera {internal_camera_id}: Error during open/warm-up: {e}. Retrying in 1s...")
            if cap: cap.release()  # Ensure camera is released on error
            cap = None
            time.sleep(1)  # Give system a break

    # After the loop, check if camera is finally opened
    if not cap or not cap.isOpened():
        print(
            f"Thread for Camera {internal_camera_id}: Failed to open after {max_total_open_attempts} total attempts. Marking as failed.")
        with buffer_lock:
            latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                f"Camera {internal_camera_id}\nX Failed to Open",
                                                                                border_color=(0, 0, 200))
        return  # Exit the thread if camera couldn't be opened

    # Initial placeholder indicating streaming has started (before first frame arrives)
    with buffer_lock:
        latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                            f"Camera {internal_camera_id}\nStreaming...")

    try:  # Main streaming loop try-except-finally block
        while running and cap and cap.isOpened():
            ret, frame = cap.read()  # This is a blocking call, waits for a new frame
            if ret:
                # Add "Camera X" label directly on the frame for consistency with image
                frame = create_text_overlay(frame, f"Camera {internal_camera_id}")
                with buffer_lock:
                    latest_frames_buffer[internal_camera_id] = frame
            else:
                # Frame not read, check if camera is still open or needs reconnection
                if not cap.isOpened():
                    print(f"Thread for Camera {internal_camera_id}: Lost, attempting to re-open...")
                    with buffer_lock:
                        latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                            f"Camera {internal_camera_id}\nReconnecting...",
                                                                                            border_color=(0, 100,
                                                                                                          200))  # Blue border for reconnect
                    time.sleep(2)  # Wait before retrying
                    cap.release()
                    cap = cv2.VideoCapture(internal_camera_id, backend_api)  # Attempt to re-open
                    if not cap.isOpened():
                        print(f"Thread for Camera {internal_camera_id}: Failed to re-open after loss.")
                        with buffer_lock:
                            latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH,
                                                                                                CAMERA_HEIGHT,
                                                                                                f"Camera {internal_camera_id}\nX Failed Reconnect",
                                                                                                border_color=(0, 0,
                                                                                                              200))
                        break  # Exit loop if re-open fails
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)  # Re-apply resolution
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        print(f"Thread for Camera {internal_camera_id}: Re-opened successfully after loss.")
                        continue  # Skip to next loop iteration to get a valid frame

    except Exception as e:
        # Catch unexpected errors during the main streaming loop
        print(f"Thread for Camera {internal_camera_id}: An error occurred: {e}")
        with buffer_lock:
            latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                f"Camera {internal_camera_id}\nERROR: {e}",
                                                                                border_color=(0, 0, 200))
    finally:
        # This block always executes when the thread finishes
        if cap:
            cap.release()  # Ensure camera resource is released
        with buffer_lock:
            if internal_camera_id in latest_frames_buffer:
                latest_frames_buffer[internal_camera_id] = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                                    f"Camera {internal_camera_id}\nDisconnected",
                                                                                    border_color=(0, 0, 200))
        print(f"Thread for Camera {internal_camera_id}: Capture thread terminated.")


# --- Mouse Event Handling ---
def on_mouse_click(event, x, y, flags, param):
    """Callback function for mouse clicks on the main OpenCV window."""
    global current_maximized_camera_id, in_maximized_view, running

    if event == cv2.EVENT_LBUTTONDOWN:
        if in_maximized_view:
            # If already maximized, clicking anywhere returns to grid view
            in_maximized_view = False
            current_maximized_camera_id = None
            print("Returning to grid view.")
        else:
            # In grid view, calculate which quadrant was clicked
            col = x // SUB_FRAME_DISPLAY_WIDTH
            row = y // SUB_FRAME_DISPLAY_HEIGHT

            # Convert 0-indexed grid position (0-3) to internal grid index
            clicked_grid_position_index = row * 2 + col

            # Get the actual internal_camera_id that is mapped to this grid position
            internal_camera_id_for_quadrant = display_quadrants_mapping.get(clicked_grid_position_index)

            if internal_camera_id_for_quadrant is not None:
                # Check if the camera is actually streaming (not just a placeholder for failed/no camera)
                with buffer_lock:
                    current_frame_data = latest_frames_buffer.get(internal_camera_id_for_quadrant)

                    # Heuristic to determine if the frame is a real stream or a placeholder
                    is_streaming_or_valid = True
                    placeholder_texts = [
                        "Loading...", "Failed to Open", "No Frames", "Reconnecting...",
                        "Failed Reconnect", "Disconnected", "ERROR:", "No Camera Here",
                        "Warm-up Failed"
                    ]
                    if current_frame_data is not None:
                        # Convert numpy array to string for text searching
                        frame_text_content = current_frame_data.tobytes().decode(errors='ignore')
                        for text_part in placeholder_texts:
                            if text_part in frame_text_content:
                                is_streaming_or_valid = False
                                break
                    else:
                        is_streaming_or_valid = False  # No frame data at all

                    if is_streaming_or_valid:
                        current_maximized_camera_id = internal_camera_id_for_quadrant
                        in_maximized_view = True
                        print(f"Maximizing camera with internal ID {internal_camera_id_for_quadrant}")
                    else:
                        print(
                            f"Camera with internal ID {internal_camera_id_for_quadrant} is not streaming or failed to open. Cannot maximize.")
            else:
                print(f"No active camera assigned to this position (Grid Index {clicked_grid_position_index}).")


# --- Main Application Logic ---
# Global mapping from 0-indexed grid position (0-3) to the actual internal_camera_id.
# This will be populated after detection to ensure unique working cameras map to unique quadrants.
display_quadrants_mapping = {}


def main():
    """
    Main function to detect cameras, start capture threads, and display a composite view.
    """
    global running, in_maximized_view, current_maximized_camera_id, display_quadrants_mapping

    # User wants 1-based display, but internal OpenCV indices are 0-based or arbitrary.
    # We will iterate through a range of common internal IDs starting from 0 up to a limit.
    # This range attempts to discover all possible cameras on the system.
    internal_camera_id_scan_range = list(range(MAX_CAMERAS_TO_CHECK + 5))  # Scan a few more indices just in case

    # List of (internal_camera_id, backend_name_string, backend_api_value) tuples that successfully opened
    successfully_detected_unique_cameras = []

    print("--- Camera Detection Phase ---")
    if ENABLE_SINGLE_CAMERA_TESTING:
        # In single test mode, only check the specified TEST_SINGLE_CAMERA_ID (which is 1-based for user input)
        # We need to use this as the internal ID for VideoCapture
        internal_id_to_test = TEST_SINGLE_CAMERA_ID
        print(
            f"Single camera testing mode: Checking Camera {internal_id_to_test} (internal index {internal_id_to_test})...")

        found_test_camera = False
        for backend_name, backend_api_val in OPENCV_BACKENDS:
            temp_cap = cv2.VideoCapture(internal_id_to_test, backend_api_val)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()  # Try to read a frame to validate
                if ret:
                    successfully_detected_unique_cameras.append(
                        (internal_id_to_test, backend_name, backend_api_val))  # Store name and value
                    print(f"Found and validated Camera {internal_id_to_test} with backend: {backend_name}")
                    found_test_camera = True
                    temp_cap.release()
                    break  # Found it, no need to try other backends for this single camera
                temp_cap.release()
            print(f"Camera {internal_id_to_test} failed with backend {backend_name}. Trying next...")
            time.sleep(0.1)  # Small delay between backend tries

        if not found_test_camera:
            print(f"Camera {internal_id_to_test} could not be opened with any backend.")
            input("Press Enter to exit...")
            return

    else:
        # Auto-detect cameras within the specified scan range, trying all backends
        print(
            f"Auto-detecting cameras by scanning internal indices {internal_camera_id_scan_range[0]} to {internal_camera_id_scan_range[-1]}...")

        # Use a set to keep track of physical device IDs to avoid adding duplicates
        # based on cv2.CAP_PROP_GUID or similar, but for simplicity, we'll rely on index uniqueness for now

        for i in internal_camera_id_scan_range:
            print(f"Checking internal camera index {i}...")
            found_for_this_index = False
            for backend_name, backend_api_val in OPENCV_BACKENDS:
                temp_cap = cv2.VideoCapture(i, backend_api_val)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()  # Try to read a frame to validate
                    if ret:
                        successfully_detected_unique_cameras.append(
                            (i, backend_name, backend_api_val))  # Store name and value
                        print(f"  --> Found and validated internal index {i} with backend: {backend_name}")
                        found_for_this_index = True
                        temp_cap.release()
                        break  # Found a working backend for this index, move to next internal index
                    temp_cap.release()  # Release if opened but no frame
                print(f"  Internal index {i} failed with backend {backend_name}. Trying next...")
                time.sleep(0.1)  # Small delay between backend tries
            time.sleep(0.3)  # Increased delay between checking each internal camera index

    if not successfully_detected_unique_cameras:
        print("No cameras detected. Please ensure cameras are connected and drivers are installed.")
        input("Press Enter to exit...")  # Keep console open for user to read message
        return

    print(f"\n--- Detected {len(successfully_detected_unique_cameras)} Unique Working Cameras ---")
    for internal_id, backend_name_str, backend_api_val in successfully_detected_unique_cameras:
        print(f"  Internal ID: {internal_id}, Backend: {backend_name_str}")  # Print name here

    print("\n--- Assigning Cameras to Display Quadrants (1-4) ---")

    assigned_count = 0
    for grid_pos in range(MAX_CAMERAS_TO_CHECK):  # Iterate through 4 fixed grid positions
        if assigned_count < len(successfully_detected_unique_cameras):
            # Assign the next available unique camera to this grid position
            internal_camera_id, backend_name_str, backend_api_val = successfully_detected_unique_cameras[assigned_count]
            display_quadrants_mapping[grid_pos] = internal_camera_id  # Store only the internal ID in mapping

            # Start a thread for this assigned camera
            thread = threading.Thread(target=capture_and_buffer, args=(internal_camera_id, backend_api_val))
            thread.daemon = True  # Allow main program to exit even if threads are running
            camera_threads.append(thread)
            thread.start()
            print(
                f"  Quadrant {grid_pos + 1} (Camera {grid_pos + 1} label) assigned internal Camera {internal_camera_id}.")
            time.sleep(0.75)  # Increased delay between starting each camera thread to prevent contention
            assigned_count += 1
        else:
            # No more unique cameras to assign, this quadrant will be blank
            display_quadrants_mapping[grid_pos] = None
            print(f"  Quadrant {grid_pos + 1} (Camera {grid_pos + 1} label) has no camera assigned.")

    # Give threads a moment to start capturing and buffering frames
    time.sleep(2)

    # Create the single main display window
    cv2.namedWindow(composite_window_name, cv2.WINDOW_NORMAL)
    # Set the mouse callback for the main window to handle maximize/minimize clicks
    cv2.setMouseCallback(composite_window_name, on_mouse_click)

    print("\n--- Starting Main Display Loop ---")
    # --- Main Display Loop ---
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
                    print(
                        f"Maximized internal Camera {current_maximized_camera_id} frame not available. Returning to grid.")
                    in_maximized_view = False
                    current_maximized_camera_id = None
                    # Show a generic error placeholder if we can't get the maximized frame
                    max_frame_to_display = create_placeholder_frame(CAMERA_WIDTH, CAMERA_HEIGHT,
                                                                    f"Camera {current_maximized_camera_id}\nError/No Data",
                                                                    border_color=(0, 0, 200))

            if max_frame_to_display is not None:
                cv2.imshow(composite_window_name, max_frame_to_display)
                # Resize window to single HD frame when maximized
                cv2.resizeWindow(composite_window_name, CAMERA_WIDTH, CAMERA_HEIGHT)
        else:
            # Display composite grid view
            quadrant_frames = []
            for i in range(MAX_CAMERAS_TO_CHECK):  # Iterate through the 4 fixed quadrants (0, 1, 2, 3)
                internal_camera_id_for_quadrant = display_quadrants_mapping.get(i)  # Get internal ID from mapping

                frame_to_display = None
                if internal_camera_id_for_quadrant is not None:
                    # If a camera is assigned to this quadrant, try to get its frame from the buffer
                    with buffer_lock:
                        frame = latest_frames_buffer.get(internal_camera_id_for_quadrant)  # Fetch using internal ID
                        if frame is not None:
                            # Frame is already at CAMERA_WIDTH/HEIGHT, no resizing needed here if matching SUB_FRAME_DISPLAY_WIDTH/HEIGHT
                            frame_to_display = frame
                        else:
                            # Fallback for assigned camera but not yet streaming or errored
                            frame_to_display = create_placeholder_frame(SUB_FRAME_DISPLAY_WIDTH,
                                                                        SUB_FRAME_DISPLAY_HEIGHT,
                                                                        f"Camera {internal_camera_id_for_quadrant}\nLoading...",
                                                                        border_color=(150, 150,
                                                                                      0))  # Yellowish for loading
                else:
                    # No camera assigned to this quadrant in `display_quadrants_mapping`
                    # Determine the label based on the fixed quadrant position (1-based for user)
                    label = f"Camera {i + 1}"
                    frame_to_display = create_placeholder_frame(SUB_FRAME_DISPLAY_WIDTH, SUB_FRAME_DISPLAY_HEIGHT,
                                                                f"{label}\nNo Camera Here", border_color=(50, 50, 50))

                quadrant_frames.append(frame_to_display)

            # Arrange the 4 quadrant frames into a 2x2 grid
            top_row = cv2.hconcat([quadrant_frames[0], quadrant_frames[1]])
            bottom_row = cv2.hconcat([quadrant_frames[2], quadrant_frames[3]])
            composite_image = cv2.vconcat([top_row, bottom_row])

            # Display the composite image
            cv2.imshow(composite_window_name, composite_image)
            # Resize window to fit the 2x2 grid of HD frames
            cv2.resizeWindow(composite_window_name, SUB_FRAME_DISPLAY_WIDTH * 2, SUB_FRAME_DISPLAY_HEIGHT * 2)

        # Check for 'q' key press globally, and also if the window was closed via the X button
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(composite_window_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False

        time.sleep(0.01)  # Small sleep to prevent busy-waiting, but not introduce significant lag

    print("\n--- Exiting Application ---")
    running = False  # Set the global flag to False to stop all threads
    for thread in camera_threads:
        thread.join(timeout=5)  # Wait for each thread to finish, with a timeout

    print("Releasing cameras and closing windows...")
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print("All cameras released and windows closed.")


if __name__ == "__main__":
    main()
