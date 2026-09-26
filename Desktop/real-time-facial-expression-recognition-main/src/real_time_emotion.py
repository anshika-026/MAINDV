import cv2
import numpy as np
import tensorflow as tf
from collections import deque


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/emotion_model_v3.keras"
)


# ==========================================
# 2. EMOTION LABELS
# ==========================================

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ==========================================
# 3. PREDICTION SMOOTHING
# ==========================================

# Store predictions from last 10 frames

prediction_history = deque(maxlen=10)


# ==========================================
# 4. LOAD FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break


    # ======================================
    # CONVERT TO GRAYSCALE
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================
    # DETECT FACES
    # ======================================

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(50, 50)

    )


    # ======================================
    # PROCESS FACE
    # ======================================

    for (x, y, w, h) in faces:


        # Crop face

        face = gray[
            y:y + h,
            x:x + w
        ]


        # Resize

        face = cv2.resize(
            face,
            (48, 48)
        )


        # Normalize

        face = face / 255.0


        # Add channel dimension

        face = np.expand_dims(
            face,
            axis=-1
        )


        # Add batch dimension

        face = np.expand_dims(
            face,
            axis=0
        )


        # ==================================
        # MODEL PREDICTION
        # ==================================

        prediction = model.predict(
            face,
            verbose=0
        )[0]


        # ==================================
        # STORE PREDICTION
        # ==================================

        prediction_history.append(
            prediction
        )


        # ==================================
        # CALCULATE AVERAGE PREDICTION
        # ==================================

        average_prediction = np.mean(
            prediction_history,
            axis=0
        )


        # Get final emotion

        predicted_index = np.argmax(
            average_prediction
        )


        emotion = emotion_labels[
            predicted_index
        ]


        confidence = (
            average_prediction[
                predicted_index
            ] * 100
        )


        # ==================================
        # DRAW RECTANGLE
        # ==================================

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )


        # ==================================
        # DISPLAY EMOTION
        # ==================================

        text = (
            f"{emotion}: "
            f"{confidence:.1f}%"
        )


        cv2.putText(

            frame,

            text,

            (x, y - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2

        )


    # ======================================
    # DISPLAY WEBCAM
    # ======================================

    cv2.imshow(

        "Real-Time Facial Emotion Recognition",

        frame

    )


    # Press Q to exit

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# RELEASE WEBCAM
# ==========================================

cap.release()

cv2.destroyAllWindows()