import os
import cv2 as cv
import numpy as np


# Directory containing images of people
DIR = 'C:\\Users\\User\\Desktop\\persons'


# List to hold names of people (folders)
people = []
for folder in os.listdir(DIR):
    people.append(folder)


# Load the pre-trained Haar Cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Lists to hold face features and corresponding labels
features = []
labels = []


def create_train():
    # Loop through each person's folder
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)  # Assign a label based on the index of the person

        # Loop through each image in the person's folder
        for image in os.listdir(path):
            img_path = os.path.join(path, image)

            img_array = cv.imread(img_path)
            if img_array is not None:
                # Convert image to grayscale
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                # Loop through detected faces
                for (x, y, w, h) in faces_rect:
                    faces_roi = gray[y:y+h, x:x + w]  # Region of interest (ROI) for the face
                    features.append(faces_roi)  # Append the ROI to features list
                    labels.append(label)  # Append the corresponding label to labels list


# Call the function to create the training data
create_train()

print('Training done -----------')

# Convert features and labels lists to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create and train the LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save the trained face recognizer and data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)