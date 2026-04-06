import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# Start video (0 = webcam or replace with video file)
cap = cv2.VideoCapture(r'C:\Users\riddh_wqixihs\Desktop\ExcerciseFormCorrectionCoach\data\push-up\incorrect\9.mp4')

# Window setup
cv2.namedWindow('Exercise Form Demo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Exercise Form Demo', 1280, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display angle near elbow
            cv2.putText(image, f'Angle: {int(angle)}',
                        tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Feedback logic
            feedback = "Good Form"
            if angle > 160:
                feedback = "Lower your arm"
            elif angle < 50:
                feedback = "Raise your arm"

            # Display feedback (BIG)
            cv2.putText(image, f'Feedback: {feedback}',
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            # Display coordinates
            cv2.putText(image, f'Elbow: {elbow}',
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ----------- CENTER + RESIZE FRAME ----------- #
        window_w, window_h = 1280, 720
        h, w = image.shape[:2]

        scale = min(window_w / w, window_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)

        x_offset = (window_w - new_w) // 2
        y_offset = (window_h - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        # -------------------------------------------- #

        cv2.imshow('Exercise Form Demo', canvas)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()