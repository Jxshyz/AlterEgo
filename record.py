import os
import uuid
import csv
import cv2 as cv
import gdown
from common import ROOT_FOLDER, YuNet

# This is the data recording pipeline
def record(args):

    # Exit if folder is None
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

    # Create folder for recorded person
    target_folder = os.path.join(ROOT_FOLDER, args.folder)
    os.makedirs(target_folder, exist_ok=True)


    # Google drive download
    directory = os.path.join(os.path.dirname(__file__), "models")
    output_path = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
    os.makedirs(directory, exist_ok=True)

    if not os.path.isfile(output_path):
        print(f"{output_path} not found. Downloading YuNet model...")
        gdown.download(YuNet, output_path, fuzzy=True)
        print(f"Download complete")


    # Open webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera!")
        exit()

    # Variable to detect when to save frame
    frames_since_detection = 0
    save_frames = True

    while True:
        # Capture frame
        ret, frame = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_with_rectangle = frame.copy()

        #directory = r"C:\Users\Josch\Documents\Uni\5. Semester\advanced ai\AlterEgo\models"
        weights = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
        face_detector = cv.FaceDetectorYN_create(weights, "", (0, 0))

        # Change to grayscale

        # Get the dimensions of the frame and set the input size for YuNet
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        # Start cascade
        _, faces = face_detector.detect(frame)

        if faces is not None:  # If faces are detected
            for face in faces:
                # Extract bounding box coordinates (first 4 values)
                x, y, w, h = map(int, face[:4])  # Only take the first 4 values for the bounding box

                # Draw rectangle around the face
                cv.rectangle(frame_with_rectangle, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(frame_with_rectangle, args.folder, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

                # Save the cropped face matrix for further processing
                face_matrix = frame[y:y + h, x:x + w]  # Crop the face region from the frame
                filename = f"face_{args.folder}_{uuid.uuid4()}"
                cv.imwrite(os.path.join(target_folder, f"{filename}_face.jpg"), face_matrix)

                # Save the bounding box coordinates in a CSV file
                with open(os.path.join(target_folder, f"{filename}.csv"), "w", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=",")
                    csv_writer.writerow([x, y, w, h])

            # Change save_frames status and reset counter
            save_frames = False
            frames_since_detection = 0

        # Dont save face for 30 frames after face was saved
        else:
            frames_since_detection += 1

            if frames_since_detection >= 30:
                save_frames = True

        # Display frame
        cv.imshow('frame', frame_with_rectangle)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
