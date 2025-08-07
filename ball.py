import cv2
from pyzbar import pyzbar
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Define the central crop percentage (e.g., 40% of the frame)
# You may need to make this smaller for more extreme fisheye lenses.
CROP_PERCENT = 0.4

print("Scanning central region for QR codes... Press 'q' to quit.")
print("Place the QR code inside the blue rectangle.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape

    # Calculate the crop dimensions
    crop_w = int(w * CROP_PERCENT)
    crop_h = int(h * CROP_PERCENT)

    # Calculate top-left corner for a centered crop
    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2

    # Create the cropped frame for detection
    cropped_frame = frame[y_start:y_start + crop_h, x_start:x_start + crop_w]

    # Draw a blue rectangle on the original frame to guide the user
    cv2.rectangle(frame, (x_start, y_start), (x_start + crop_w, y_start + crop_h), (255, 0, 0), 3)

    # --- Use pyzbar to detect QR codes in the CROPPED frame ---
    decoded_objects = pyzbar.decode(cropped_frame)

    for obj in decoded_objects:
        # Get the decoded data
        data = obj.data.decode("utf-8")
        print(f"Decoded data: {data}")

        # Get the bounding box of the QR code (relative to the cropped frame)
        points = obj.polygon

        # If the points are just a list of (x,y), convert to a NumPy array
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))

            # --- Adjust points to the original frame's coordinate system ---
            pts[:, :, 0] += x_start
            pts[:, :, 1] += y_start

            # Draw the bounding box on the original frame
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Put the decoded text on the original frame
            text_pos = (pts[0][0][0], pts[0][0][1] - 10)
            cv2.putText(frame, data, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Fisheye QR Scanner (No Calibration)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()