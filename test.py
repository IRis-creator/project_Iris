import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime, timedelta
from PIL import Image
import dlib

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # Use the correct index for the external webcam

# Check if video capture opened successfully
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Load the face detector and shape predictor for blink detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists in your directory

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
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
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
blink_threshold = 0.22  # Eye aspect ratio threshold for detecting a blink
consecutive_frames = 2  # Number of consecutive frames to confirm a blink
blink_counter = {name: 0 for name in students}  # Track consecutive frames below the threshold

# Time tracking for 2-minute period and 15-second cutoff
start_time = datetime.now()
cutoff_time = start_time + timedelta(seconds=15)
class_end_time = start_time + timedelta(minutes=2)

# Initialize the motion detection variables
prev_frame = None
motion_threshold = 5000  # Threshold to determine if significant motion is detected

# Path for attendance CSV file
current_date = start_time.strftime("%Y-%m-%d")
attendance_file = f'{current_date}_attendance.csv'

# Function to send a message to absent students
def send_absent_message(name):
    # This is a placeholder function for sending a message.
    # You can replace this with actual messaging code (SMS, email, etc.)
    print(f"Sending message: 'Why are you absent?' to {name}")

# Function to check attendance for a specific student
def check_student_attendance(student_name):
    present_days = 0
    absent_days = 0
    late_days = 0

    for csv_file in os.listdir('.'):  # Assuming all attendance files are in the current directory
        if csv_file.endswith('.csv') and csv_file.startswith('attendance'):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Name'] == student_name:
                        if row['Status'] == "Present":
                            present_days += 1
                        elif row['Status'] == "Absent":
                            absent_days += 1
                        elif row['Status'] == "Late Present":
                            late_days += 1

    print(f"{student_name}'s Attendance:")
    print(f"Present: {present_days} days")
    print(f"Absent: {absent_days} days")
    print(f"Late Present: {late_days} days")

# Video capture and face recognition loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from video capture.")
        break

    # Resize the frame for faster processing and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Detect face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    # Convert frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        # Compute the absolute difference between the current frame and previous frame
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        # Compute the sum of the absolute differences
        motion_score = np.sum(frame_diff)
        # Check if motion is detected
        if motion_score < motion_threshold:
            print("No significant motion detected. Skipping frame.")
            prev_frame = gray_frame
            continue

    current_time = datetime.now()

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.6:  # Confidence threshold
            name = known_face_names[best_match_index]
            face_names.append(name)

            # Check if the student is already marked
            if attendance_status[name] != "Absent":
                print(f"{name} is already marked as {attendance_status[name]}. Skipping further checks.")
                continue  # Skip if the student is already marked as present or late present

            # Check for eye blinks using dlib
            rects = detector(gray_frame, 0)

            for rect in rects:
                shape = predictor(gray_frame, rect)
                shape = np.array([[p.x, p.y] for p in shape.parts()])
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear)/2.5

                # Print EAR values for debugging
                print(f"{name}: Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Avg EAR: {avg_ear:.2f}")

                # Check if a blink is detected
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
                    blink_counter[name] = 0  # Reset the blink counter

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        else:
            face_names.append("Unknown")
            print("Face does not match any known faces.")

    # Display the frame
    cv2.imshow("Attendance System", frame)

    # Check if class period is over
    if current_time > class_end_time or cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_frame = gray_frame

# Mark absent students and send messages
for name, status in attendance_status.items():
    if status == "Absent":
        send_absent_message(name)

# Write attendance status to the CSV file
with open(attendance_file, 'a+', newline='') as f:
    lnwriter = csv.writer(f)
    f.seek(0)
    if f.read(100) == '':  # Write header if the file is empty
        lnwriter.writerow(["Name", "Date", "Time", "Status"])

    for name, status in attendance_status.items():
        lnwriter.writerow([name, current_date, start_time.strftime("%H:%M:%S"), status])

video_capture.release()
cv2.destroyAllWindows()
print("Attendance process complete. Camera closed.")

# After attendance collection, query the student's attendance
student_name = input("Enter the name of the student to check attendance: ")
check_student_attendance(student_name)
