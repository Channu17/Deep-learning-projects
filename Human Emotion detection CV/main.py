import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load Haar Cascade for face detection
path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + path)

# Load the pre-trained emotion detection model
model = load_model('Human Emotion detection CV/Final_model.keras')  # Replace with your model's path

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','neutral', 'Sad', 'Surprise']

# Font settings
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)

# Create a black image for testing
img = np.zeros((500, 500, 3), dtype="uint8")
text = "Some in a box!"

(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))

# Start video capture for real-time emotion detection
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
    # Extract and preprocess the face ROI
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))  # Resize grayscale face ROI
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        face_roi = face_roi / 255.0  # Normalize pixel values
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(face_roi)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        # Display emotion label above the face
        cv2.putText(frame, emotion, (x, y - 10), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

