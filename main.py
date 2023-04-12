from allimports import *
# Load the emotion classifier
classifier = cv2.ml.SVM_load("haarcascade_frontalface_default.xml")

# Capture the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the facial landmarks
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(68):
            landmarks[i] = (x + predictor_data[i][0] * w // 100, y + predictor_data[i][1] * h // 100)

        # Convert the landmarks to a feature vector
        features = []
        for i in range(68):
            x = landmarks[i][0]
            y = landmarks[i][1]
            features.append(x)
            features.append(y)

        # Classify the emotion
        features = np.array(features).reshape(1, -1)
        emotion = classifier.predict(features)[1][0]

        # Display the emotion on the screen
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame on the screen
    cv2.imshow("Frame", frame)

    # Exit if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()