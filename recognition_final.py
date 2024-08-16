import numpy as np
import cv2 as cv

def rescale_frame(frame, scale=0.75):
    # Rescale the frame (image, video, or live video) by the given scale factor
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load pre-trained Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# List of people for label mapping
people = ['Cillian Murphy', 'Emma Watson', 'Rupert Grint']

# Initialize the face recognizer and load the trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load and rescale the input image
img = cv.imread("C:\\Users\\User\\Downloads\\1685815019_gagaru-club-p-killian-merfi-glaza-pinterest-36.jpg")
img = rescale_frame(img, 0.5)
# cv.imshow('Rescaled', img)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 10)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]  # Region of interest for the face

    # Predict the label and confidence for the detected face
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Draw label and rectangle around the detected face
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected face
cv.imshow('Detected face', img)
cv.waitKey(0)
