import cv2
import face_recognition
import os
from datetime import datetime
import csv
import random

# Load known faces
known_encodings = []
known_names = []

path = 'known_faces'
for file in os.listdir(path):
    img_path = os.path.join(path, file)
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(file)[0])
    else:
        print(f"[WARNING] No face found in: {file}")

video = cv2.VideoCapture(0)

unknown_encodings = []
unknown_labels = []
unknown_count = 0
csv_filename = "daily_visitors.csv"
seen_unknowns_today = []

# CSV file setup
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Gender", "AgeGroup", "Date", "Time"])

# Dummy Gender and Age
def dummy_gender():
    return random.choice(["Male", "Female"])

def dummy_age():
    return "Adult"

# Remove from CSV
def remove_unknown_from_csv(label):
    if not os.path.exists(csv_filename):
        return
    rows = []
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in rows:
            if row and row[0] != label:
                writer.writerow(row)

# Check if already counted today
def has_seen_today(face_encoding, current_date):
    for enc, date in seen_unknowns_today:
        if date == current_date:
            distance = face_recognition.face_distance([enc], face_encoding)[0]
            if distance < 0.45:
                return True
    return False

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = []
    valid_face_locations = []

    for loc in face_locations:
        encoding = face_recognition.face_encodings(rgb_small_frame, [loc])
        if encoding:
            top, right, bottom, left = loc
            face_image = rgb_small_frame[top:bottom, left:right]
            landmarks_list = face_recognition.face_landmarks(face_image)
            if landmarks_list:
                landmarks = landmarks_list[0]
                required = ['left_eye', 'right_eye', 'nose_tip', 'top_lip', 'chin']
                if all(f in landmarks for f in required):
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    nose = landmarks['nose_tip']

                    left_eye_x = sum(pt[0] for pt in left_eye) / len(left_eye)
                    right_eye_x = sum(pt[0] for pt in right_eye) / len(right_eye)
                    nose_x = sum(pt[0] for pt in nose) / len(nose)

                    eye_center = (left_eye_x + right_eye_x) / 2
                    symmetry_offset = abs(nose_x - eye_center)

                    if symmetry_offset < 5:
                        face_encodings.append(encoding[0])
                        valid_face_locations.append(loc)

    for (top, right, bottom, left), face_encoding in zip(valid_face_locations, face_encodings):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if face_distances[best_match_index] < 0.45:
                name = known_names[best_match_index]

                # Remove from unknown if matched later
                for i, unknown_enc in enumerate(unknown_encodings):
                    distance = face_recognition.face_distance([unknown_enc], face_encoding)[0]
                    if distance < 0.45:
                        unknown_count -= 1
                        label = unknown_labels[i]
                        remove_unknown_from_csv(label)
                        unknown_encodings.pop(i)
                        unknown_labels.pop(i)
                        break
            else:
                name = "Unknown"
        else:
            name = "Unknown"

        if name == "Unknown":
            if has_seen_today(face_encoding, current_date):
                continue

            duplicate = False
            for enc in unknown_encodings:
                distance = face_recognition.face_distance([enc], face_encoding)[0]
                if distance < 0.45:
                    duplicate = True
                    break

            if not duplicate:
                unknown_count += 1
                unknown_encodings.append(face_encoding)
                unknown_label = f"Unknown_{unknown_count}"
                unknown_labels.append(unknown_label)

                seen_unknowns_today.append((face_encoding, current_date))

                gender = dummy_gender()
                age_group = dummy_age()

                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([unknown_label, gender, age_group, current_date, current_time])

        display_text = f"{name} | {current_time} | {current_date}"
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Unknown Count: {unknown_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Human Visitor Logger', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
 