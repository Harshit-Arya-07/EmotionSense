import cv2
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional

# Load the model
try:
    print("Loading model...")
    model: tf.keras.Model = tf.keras.models.load_model("facialemotionmodel.h5")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    exit(1)

# Load Haar cascade for face detection
try:
    print("Loading Haar cascade classifier...")
    haar_file: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(haar_file)
    if face_cascade.empty():
        raise ValueError("Failed to load Haar cascade classifier")
    print("Haar cascade classifier loaded successfully")
except Exception as e:
    print(f"Error loading Haar cascade: {str(e)}")
    exit(1)

# Function to process images
def extract_features(image: np.ndarray) -> np.ndarray:
    feature: np.ndarray = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0  # Normalize pixel values

# Initialize webcam
print("Initializing webcam...")
webcam: cv2.VideoCapture = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam")
    exit(1)
print("Webcam initialized successfully")

labels: Dict[int, str] = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    success: bool
    im: np.ndarray
    success, im = webcam.read()
    if not success:
        print("Error accessing webcam.")
        break

    gray: np.ndarray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces: np.ndarray = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image: np.ndarray = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)

        image = cv2.resize(image, (48, 48))
        img: np.ndarray = extract_features(image)

        pred: np.ndarray = model.predict(img)
        prediction_label: str = labels[pred.argmax()]

        # Display prediction text
        cv2.putText(im, prediction_label, (p - 10, q - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow("Emotion Detection", im)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()
