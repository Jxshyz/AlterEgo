import os
import json

def get_person_folder():
    """
    Create a new folder named 'Person i' in the './trash' directory.
    Increment i if the folder already exists.
    """
    base_path = "./trash"
    os.makedirs(base_path, exist_ok=True)  # Ensure './trash' exists

    i = 1
    while True:
        folder_path = os.path.join(base_path, f"Person {i}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path
        i += 1


def save_face_data(landmarks, timestamp, folder_path):
    """
    Saves face landmarks and associated data to a JSON file in the specified folder.
    """
    output_file = os.path.join(folder_path, "face_data.json")  # Save in the correct folder
    data = {
        "timestamp": timestamp,
        "landmarks": [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z} for landmark in landmarks
        ],
    }

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = json.load(f)
        existing_data.append(data)
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(output_file, "w") as f:
            json.dump([data], f, indent=4)