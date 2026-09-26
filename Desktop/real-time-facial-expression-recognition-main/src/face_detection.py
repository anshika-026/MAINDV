import cv2


# ==========================================
# LOAD FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================================
# START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)


while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break


    # Convert frame to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces
    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(50, 50)

    )


    # Draw rectangle around faces
    for (x, y, w, h) in faces:

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )


    # Show webcam
    cv2.imshow(
        "Face Detection",
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