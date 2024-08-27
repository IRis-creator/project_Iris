import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime, timedelta
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
attendance_status = {name: {"Period 1": "Absent", "Period 2": "Absent"} for name in students}
entry_times = {}

# Define the two periods
period_duration = timedelta(minutes=2)
start_time_period_1 = now
start_time_period_2 = start_time_period_1 + period_duration + timedelta(seconds=1)

# Function to update attendance based on the entry time
def update_attendance(name, entry_time, period):
    if entry_time <= start_time_period_1 + timedelta(seconds=20):
        attendance_status[name][period] = "Present"
    else:
        attendance_status[name][period] = "Late Present"
    print(f"{name} marked as {attendance_status[name][period]} for {period}")

# Function to handle attendance for a specific period
def process_period(period_name, start_time, duration):
    period_end_time = start_time + duration
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame from video capture.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
                
                if name in students:
                    current_time = datetime.now()
                    update_attendance(name, current_time, period_name)
                    students.remove(name)  # Remove the student once they are marked

                    # Draw a rectangle around the face
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                else:
                    print(f"{name} already marked for {period_name}")

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or not students or datetime.now() > period_end_time:
            break
    
    # After the period ends, mark remaining students as absent
    for name in students:
        print(f"{name} marked as Absent for {period_name}")

# Process Period 1
process_period("Period 1", start_time_period_1, period_duration)

# Wait until Period 2 starts
print("Waiting for Period 2 to start...")
while datetime.now() < start_time_period_2:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Process Period 2
process_period("Period 2", start_time_period_2, period_duration)

# Write attendance status to the CSV file
with open(f'{current_date}_attendance.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    # Write the header row
    lnwriter.writerow(["Name", "Period", "Date", "Time", "Status"])
    for name, periods in attendance_status.items():
        for period, status in periods.items():
            # For each student, record the date, time, and status
            lnwriter.writerow([name, period, current_date, now.strftime("%H:%M:%S"), status])

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

print("Attendance process complete. Camera closed.")
