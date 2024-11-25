import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import os
import gdown
import time

# Download the YuNet Model into models folder
model_url = r"https://drive.google.com/uc?id=16LiSCv1z2sf--XNWBzz6_cDSRbbrgW4X"
output_dir = "models"
output_path = os.path.join(output_dir, "face_landmarker.task")
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(output_path):
    gdown.download(model_url, output_path, fuzzy=True)

# Paths and MediaPipe setup
model_path = output_path
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Function to visualize landmarks on the frame
def draw_landmarks_on_image(rgb_image, face_landmarks_list):
    """
    Visualize facial landmarks on the given image.
    """
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Convert landmarks to a MediaPipe NormalizedLandmarkList
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        # Draw the landmarks and connections
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style()
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style()
        )

    return annotated_image

# Callback function for processing results
def result_callback(result, image, output_frame_ref):
    """
    Processes the results asynchronously and visualizes landmarks.
    """
    annotated_frame = draw_landmarks_on_image(image.numpy_view(), result.face_landmarks)
    output_frame_ref[0] = annotated_frame

# Start webcam feed and detection
def detect_with_camera():
    """
    Main function to initialize webcam feed and process live face landmarks.
    """
    # Shared reference for the output frame
    output_frame = [None]

    # Initialize Face Landmarker with the result callback
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=lambda result, image, timestamp_ms: result_callback(result, image, output_frame)
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        # Open webcam feed
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to exit.")

        # Start with the current timestamp
        current_timestamp = int(time.time() * 1000)  # Milliseconds

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image from RGB frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Send frame to the landmarker with monotonically increasing timestamp
            landmarker.detect_async(mp_image, current_timestamp)

            # Increment timestamp for the next frame
            current_timestamp += 30  # Assume ~30ms between frames (~30 FPS)

            # Annotate and display the frame if available
            if output_frame[0] is not None:
                cv2.imshow("Face Landmarker", output_frame[0])

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
