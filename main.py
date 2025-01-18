import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam
cam = cv2.VideoCapture(0)
# Initialize Mediapipe Face Mesh with refined landmarks
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# Get screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    # Capture a frame from the webcam
    _, frame = cam.read()  # read whatever coming in camera
    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    # Convert frame from BGR to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the frame to detect face landmarks
    output = face_mesh.process(rgb_frame)
    # Get detected landmarks
    landmark_points = output.multi_face_landmarks
    # Get frame dimensions
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        # Access the first detected face landmarks
        landmarks = landmark_points[0].landmark
        # Highlight landmarks used for cursor control
        for id, landmark in enumerate(landmarks[474:478]):  # Eye region landmarks
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Draw landmark points
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:  # Use landmark ID 1 for cursor movement
                # Calculate screen coordinates for cursor movement
                screen_X = screen_w / frame_w * x
                screen_Y = screen_h / frame_h * y
                pyautogui.moveTo(screen_X, screen_Y)  # Move cursor
        # Detect blinking using two specific eye landmarks
        left = [landmarks[145], landmarks[159]]  # Top and bottom of the left eye
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # Draw eye landmarks
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        # Check if the eye is blinking (distance between two points is small)
        if (left[0].y - left[1].y) < 0.006:
            pyautogui.click()  # Simulate a mouse click
            pyautogui.sleep(1)  # Prevent rapid consecutive clicks
    # Display the processed frame
    cv2.imshow('Eye controlled Mouse', frame)
    # Wait for a key press to prevent freezing
    cv2.waitKey(1)
