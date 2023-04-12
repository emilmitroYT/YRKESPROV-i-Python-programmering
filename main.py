from allimports import *

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the emotion classifier
classifier = cv2.ml.SVM_load("emotion_classifier.xml")

# Capture the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = detector(frame)

    # Loop over the detected faces
    for face in faces:
        # Extract the facial landmarks
        landmarks = predictor(frame, face)

        # Convert the landmarks to a feature vector
        features = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            features.append(x)
            features.append(y)

        # Classify the emotion
        features = np.array(features).reshape(1, -1)
        emotion = classifier.predict(features)[1][0]

        # Display the emotion on the screen
        cv2.putText(frame, emotion, (face.left(), face.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame on the screen
    cv2.imshow("Frame", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()