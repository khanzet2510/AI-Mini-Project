import cv2
import csv
import os
import numpy as np


face_cap=cv2.CascadeClassifier("C:/Users/Admin/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap=cv2.VideoCapture(0)




#-------------------------------------------------------
# Path of the directory containing images of known people
KNOWN_FACES_DIR = 'C:\\Users\\Admin\\OneDrive\\Python+SQL\\AI Mini Project\\DatasetGPT'

# Load the pre-trained recognizer for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()


# Function to save the current face as an image
def save_face():
    # Open the camera and capture a frame
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If exactly one face is detected, save it as an image
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        cv2.imwrite('current_face.jpg', face)

    # Release the camera
    cap.release()

# Function to compare the recently saved face with the known faces

def compare_faces():
    # Load saved faces
    known_faces = []
    labels = []
    for name in os.listdir(KNOWN_FACES_DIR):
        for filename in os.listdir(os.path.join(KNOWN_FACES_DIR, name)):
            image = cv2.imread(os.path.join(KNOWN_FACES_DIR, name, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(gray, (100, 100))
            known_faces.append(face)
            labels.append(name)

    # Train face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.array(labels)
    recognizer.train(known_faces, labels)

    # Capture image from camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return "Unknown"

    # Detect face in image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Compare detected face with saved faces
    for (x, y, w, h) in faces:
        current_face = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
        (label, confidence) = recognizer.predict(current_face)
        if confidence < 100:
            name = labels[label]
            return name
        else:
            return "Unknown"

# Save the current face and compare it with the known faces
save_face()
name = compare_faces()

# Print the result
print('Recognized person: ' + name)

video_cap.release()
cv2.destroyAllWindows()

