import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained models for emotion and age detection
emotion_model_path = 'F:\Downlod\Driver-Drowsiness-Detection-master\test'
age_model_path = 'path/to/age_model.h5'
emotion_model = load_model(emotion_model_path)
age_model = load_model(age_model_path)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Random professions (since there's no standard model for profession prediction)
professions = ['Engineer', 'Doctor', 'Artist', 'Teacher', 'Student', 'Lawyer', 'Scientist']

def detect_emotion_and_age(face):
    # Prepare the face for emotion and age prediction
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = cv2.resize(face_gray, (48, 48))
    face_gray = face_gray.astype("float") / 255.0
    face_gray = img_to_array(face_gray)
    face_gray = np.expand_dims(face_gray, axis=0)

    # Emotion prediction
    emotion_prediction = emotion_model.predict(face_gray)[0]
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]

    # Age prediction
    face_color = cv2.resize(face, (200, 200))
    face_color = face_color.astype("float") / 255.0
    face_color = img_to_array(face_color)
    face_color = np.expand_dims(face_color, axis=0)
    age_prediction = age_model.predict(face_color)[0]
    age = int(age_prediction[0])

    # Random profession prediction
    profession = np.random.choice(professions)

    return emotion_label, age, profession

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y+h, x:x+w]

        # Predict emotion, age, and profession
        emotion, age, profession = detect_emotion_and_age(face)

        # Draw the face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the predictions
        cv2.putText(frame, f'Emotion: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'Age: {age}', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'Profession: {profession}', (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion, Age and Profession Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
