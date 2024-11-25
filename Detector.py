import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import os
import json
import time

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Paths
model_path = "models/face_landmarker.task"

# Function to save extracted data
def save_extracted_data(data, output_path="output.json"):
    """
    Saves the extracted facial data to a JSON file.
    """
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    existing_data.append(data)

    with open(output_path, "w") as file:
        json.dump(existing_data, file, indent=4)
    print(f"[INFO] Data saved to {output_path}")

# Function to extract and save data
def extract_face_data(result, timestamp_ms):
    """
    Extracts required face data from detection results and saves it.
    """
    try:
        face_data = []
        for face_landmarks in result.face_landmarks:
            # Extract landmark data
            landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z}
                for lm in face_landmarks
            ]

            # Extract connection data (Mesh structure)
            connections = [
                {"start": conn[0], "end": conn[1]}
                for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION
            ]

            # Create a data structure for each face
            face_data.append({
                "timestamp": timestamp_ms,
                "landmarks": landmarks,
                "connections": connections
            })

        # Save the data
        save_extracted_data(face_data)

    except Exception as e:
        print(f"[ERROR] Failed to extract face data: {e}")

# Main face detection function
def detect():
    """
    Detects faces and extracts data (stops automatically after 5 seconds).
    """
    print("[INFO] Starting face detection...")
    output_path = "output.json"

    def result_callback(result, image, timestamp_ms):
        """
        Processes results asynchronously and saves data.
        """
        print(f"[DEBUG] Received results at timestamp {timestamp_ms}")
        if result.face_landmarks:
            print(f"[DEBUG] Detected {len(result.face_landmarks)} faces")
            extract_face_data(result, timestamp_ms)
        else:
            print("[DEBUG] No face detected")

    # Initialize Face Landmarker
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=result_callback,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Start webcam feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("[INFO] Webcam opened. The program will stop after 5 seconds.")

        # Start timer
        start_time = time.time()
        current_timestamp = int(time.time() * 1000)  # Current time in ms

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to capture frame")
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send frame to the landmarker
            landmarker.detect_async(mp_image, current_timestamp)

            # Increment timestamp
            current_timestamp += 30  # 30ms per frame (~30 FPS)

            # Stop the loop after 5 seconds
            if time.time() - start_time > 5:
                print("[INFO] Time limit reached. Stopping face detection...")
                break

        cap.release()
        print("[INFO] Webcam closed.")
