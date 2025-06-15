# camera_server.py

import time
import socket
import cv2
import imagezmq

# Use either a specific IP address or 'tcp://*:5555' to bind to all available interfaces
# If using a specific IP, clients must connect to that IP.
# If using '*', clients can connect to any IP of this machine.
SENDER_IP_ADDRESS = 'tcp://*:5555'

# Create a sender for each camera
# This allows for future expansion where different cameras might send to different clients
# For now, we'll just use one sender.
sender = imagezmq.ImageSender(connect_to=SENDER_IP_ADDRESS)

# Get the hostname of this machine to include in the stream
host_name = socket.gethostname()

# Initialize cameras
# This will attempt to open camera IDs 0, 1, 2, and 3
caps = {}
for i in range(4):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            caps[i] = cap
            print(f"✅ Camera {i} opened successfully.")
        else:
            print(f"⚠️ Could not open camera {i}.")
    except Exception as e:
        print(f"❌ Error opening camera {i}: {e}")

if not caps:
    print("❌ No cameras found. Exiting.")
    exit()

print(f"🚀 Starting camera server on {host_name} at {SENDER_IP_ADDRESS}...")

try:
    while True:
        # Loop through all successfully opened cameras
        for camera_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ Could not read frame from camera {camera_id}. Re-opening...")
                cap.release()
                caps[camera_id] = cv2.VideoCapture(camera_id)
                continue

            # The message sent is a tuple: (camera_id, frame)
            # This allows the client to know which camera the frame is from.
            sender.send_image(camera_id, frame)

except (KeyboardInterrupt, SystemExit):
    print("🛑 Stopping camera server.")
finally:
    # Clean up resources
    for cap in caps.values():
        cap.release()
    sender.close()
    cv2.destroyAllWindows()