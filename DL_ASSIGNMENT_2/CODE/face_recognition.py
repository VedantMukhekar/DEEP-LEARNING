import cv2
import os
import numpy as np

# Load the Haar Cascade for face detection.
#file contains data required to detect faces in images

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images from the dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

# loops through each person's folder in the dataset
    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if os.path.isdir(person_path):
            label_dict[current_label] = person

#loops through each image in the person's folder
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #cv2.imread=reads the img
                
                if img is not None: #Checks if the image is successfully read not
                    images.append(img)
                    labels.append(current_label)
            current_label += 1

    return images, labels, label_dict #returns the lists of images, labels, and the dictionary

# Load images and labels from specific folder(datset)
images, labels, label_dict = load_images_from_folder('dataset')

# Create the LBPH (local binary patterns histograms) face recognizer using opencv.
# built-in face_recognition algorithm 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer 
recognizer.train(images, np.array(labels))

# Function to recognize faces
def recognize_face(recognizer, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Uses the Haar Cascade classifier to detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        label_id, confidence = recognizer.predict(face)
    

       

        # Adjust the confidence threshold as needed
        if confidence < 80:  # confidence score is below 80  it retrieves the name corresponding to the label_id 
            name = label_dict[label_id]
            confidence_text = f"{name} ({round(100 - confidence)}%)"
        else:
            confidence_text = "Unknown" #confidence is too high then face is labled as "unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return frame

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# continuosly reads frames from the webcam 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Recognize faces in the frame
    recognized_frame = recognize_face(recognizer, frame)

    # Display the result
    cv2.imshow('Face Recognition', recognized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the webcam and closes all opencv windows
cap.release()
cv2.destroyAllWindows()