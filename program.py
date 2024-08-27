import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from PIL import Image

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Check if video capture opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Path to the folder containing student images
folder_path = "photos/"

# Initialize lists to hold known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
        # Get the full path to the image
        image_path = os.path.join(folder_path, filename)
        
        # Load and convert the image to RGB
        try:
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')
            image_rgb = np.array(pil_image).astype(np.uint8)  # Ensure it's in 8-bit RGB format

            # Get face encodings from the image
            face_encodings = face_recognition.face_encodings(image_rgb)
            if face_encodings:
                # Use the first face found in the image
                known_face_encodings.append(face_encodings[0])

                # Extract the student name from the filename (without the extension)
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"Loaded encoding for {name}")
            else:
                print(f"No face found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Copy known_face_names to students for tracking attendance
students = known_face_names.copy()

# Initialize lists for face locations, encodings, and names
face_locations = []
face_encodings = []
face_names = []

# Get the current date for the CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Dictionary to keep track of attendance status
attendance_status = {name: "Absent" for name in students}

# Video capture and face recognition loop
while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Check if the frame was captured correctly
    if not ret:
        print("Error: Could not read frame from video capture.")
        break

    # Resize the frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame to RGB
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compute the face distance for all known faces
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Check if the best match is within a reasonable distance
        if face_distances[best_match_index] < 0.6:  # Default threshold for reasonable matching
            name = known_face_names[best_match_index]
            face_names.append(name)

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # If the recognized face is in the list of students, mark them as present
            if name in students:
                students.remove(name)
                attendance_status[name] = "Present"
                print(f"{name} marked present")

        else:
            face_names.append("Unknown")
            print("Face does not match any known faces.")

    # Display the resulting frame with the title "Attendance System"
    cv2.imshow("Attendance System", frame)

    # Break the loop if the 'q' key is pressed or if all students are marked
    if cv2.waitKey(1) & 0xFF == ord('q') or not students:
        break

# Write attendance status to the CSV file
with open(f'{current_date}_attendance.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    # Write the header row
    lnwriter.writerow(["Name", "Date", "Time", "Status"])
    for name, status in attendance_status.items():
        # For each student, record the date, time, and status
        lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S"), status])

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

print("Attendance process complete. Camera closed.")
