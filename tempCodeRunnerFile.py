import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime, timedelta
from PIL import Image
import dlib

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Check if video capture opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load the face detector and shape predictor for blink detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio to detect blinking
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Path to the folder containing student images
folder_path = "photos/"

# Initialize lists to hold known face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Load and encode student images
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        try:
            pil_image = Image.open(image_path)
            pil_image = pil_image.convert('RGB')
            image_rgb = np.array(pil_image).astype(np.uint8)

            # Get face encodings from the image
            face_encodings = face_recognition.face_encodings(image_rgb)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"Loaded encoding for {name}")
            else:
                print(f"No face found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Copy known_face_names to students for tracking attendance
students = known_face_names.copy()

# Initialize variables for attendance and blinking detection
attendance_status = {name: "Absent" for name in students}
blink_threshold = 0.22
consecutive_frames = 2
blink_counter = {name: 0 for name in students}

# Time tracking for the class period
start_time = datetime.now()
cutoff_time = start_time + timedelta(seconds=15)
class_end_time = start_time + timedelta(minutes=2)

# Get the current day of the week (e.g., "Monday", "Tuesday")
current_day_of_week = start_time.strftime('%A')

# Initialize total attendance dictionary
total_attendance = {name: {"Present": 0, "Late Present": 0, "Absent": 0, "Daily Attendance": {}} for name in students}

# Load existing total attendance from CSV if it exists
if os.path.exists('total_attendance_summary.csv'):
    with open('total_attendance_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['Name']
            total_attendance[name] = {
                "Present": int(row.get("Total Present Days", 0)),
                "Late Present": int(row.get("Total Late Present Days", 0)),
                "Absent": int(row.get("Total Absent Days", 0)),
                "Daily Attendance": {day: count for day, count in eval(row.get("Daily Attendance", "{}")).items()}
            }

# Function to send a message to absent students
def send_absent_message(name):
    print(f"Sending message: 'Why are you absent?' to {name}")

# Video capture and face recognition loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from video capture.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    current_time = datetime.now()

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.6:
            name = known_face_names[best_match_index]

            # Check if the student is already marked
            if attendance_status[name] != "Absent":
                print(f"{name} is already marked as {attendance_status[name]}. Skipping further checks.")
                continue

            # Check for eye blinks
            rects = detector(frame, 0)
            for rect in rects:
                shape = predictor(frame, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < blink_threshold:
                    blink_counter[name] += 1
                else:
                    blink_counter[name] = 0

                # Confirm a blink if the counter exceeds the threshold
                if blink_counter[name] >= consecutive_frames:
                    if current_time <= cutoff_time:
                        attendance_status[name] = "Present"
                    elif current_time <= class_end_time:
                        attendance_status[name] = "Late Present"
                    else:
                        attendance_status[name] = "Absent"
                    print(f"{name} marked as {attendance_status[name]}")
                    blink_counter[name] = 0

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        else:
            print("Face does not match any known faces.")

    # Display the frame
    cv2.imshow("Attendance System", frame)

    # Check if class period is over
    if current_time > class_end_time or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Send messages to absent students
for name, status in attendance_status.items():
    if status == "Absent":
        send_absent_message(name)

# Write daily attendance status to the CSV file
current_date = start_time.strftime("%Y-%m-%d")
with open(f'{current_date}_attendance.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Date", "Time", "Status", "Day of the Week"])
    for name, status in attendance_status.items():
        lnwriter.writerow([name, current_date, start_time.strftime("%H:%M:%S"), status, current_day_of_week])

        # Update total attendance based on today's attendance
        if status == "Present":
            total_attendance[name]["Present"] += 1
        elif status == "Late Present":
            total_attendance[name]["Late Present"] += 1
        elif status == "Absent":
            total_attendance[name]["Absent"] += 1

        # Update daily attendance count
        if current_day_of_week not in total_attendance[name]["Daily Attendance"]:
            total_attendance[name]["Daily Attendance"][current_day_of_week] = 0
        total_attendance[name]["Daily Attendance"][current_day_of_week] += 1

# Write total attendance summary to a CSV file (overwrites the previous file with updated totals)
with open('total_attendance_summary.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Total Present Days", "Total Late Present Days", "Total Absent Days"])
    for name, counts in total_attendance.items():
        lnwriter.writerow([name, counts["Present"], counts["Late Present"], counts["Absent"]])

video_capture.release()
cv2.destroyAllWindows()
print("Attendance process complete. Camera closed.")
