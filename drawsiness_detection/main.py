import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import sys

# --- Configuration ---
EYE_AR_THRESH = 0.22      # EAR threshold (below this indicates eyes are closed)[reference:4]
EYE_AR_CONSEC_FRAMES = 10 # Number of frames eyes must be closed to trigger alarm[reference:5]

# Indices for eye landmarks from MediaPipe Face Mesh
# These points outline the opening of each eye[reference:6]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- Helper Function to Calculate Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye_landmarks, landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR).
    The EAR is a value that becomes very small when the eye is closed.[reference:7]
    """
    # Extract the points for the six landmarks around the eye
    points = []
    for idx in eye_landmarks:
        point = landmarks[idx]
        points.append([point.x, point.y])

    # Compute the Euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # Compute the distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    # Return the final EAR value
    ear = (A + B) / (2.0 * C)
    return ear

# --- Main Program ---
def main():
    # 1. Initialize Pygame mixer for playing sound
    pygame.mixer.init()
    
    # Provide a path to a sound file (e.g., 'alarm.wav' or 'alarm.mp3')
    # You can find free sound files online, or the code will work silently if the file is missing.
    try:
        alarm_sound = pygame.mixer.Sound("kushalave.mp3")
    except pygame.error:
        print("Warning: 'alarm.wav' not found. Audio alert disabled.")
        alarm_sound = None

    # 2. Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    # 'static_image_mode=False' optimizes for video.
    # 'max_num_faces=1' assumes only one face (the driver) is in frame.
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      refine_landmarks=True, min_detection_confidence=0.5)

    # 3. Start video capture from the default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    frame_counter = 0
    alarm_playing = False

    print("Drowsiness Detection System Started. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        # Flip frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame to detect face landmarks
        results = face_mesh.process(rgb_frame)

        current_ear = 0.0
        status_text = "Awake :)" 
        status_color = (0, 255, 0) # Green

        # If a face is found with landmarks
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Get normalized coordinates of the detected landmarks
            h, w, _ = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append((int(lm.x * w), int(lm.y * h), lm))

            # Convert MediaPipe landmarks to a format our EAR function can use
            landmark_points = face_landmarks.landmark

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(LEFT_EYE_INDICES, landmark_points)
            right_ear = eye_aspect_ratio(RIGHT_EYE_INDICES, landmark_points)
            current_ear = (left_ear + right_ear) / 2.0

            # --- Drowsiness Logic and Alerting ---
            # If the average EAR is below the threshold, eyes are likely closed.
            if current_ear < EYE_AR_THRESH:
                frame_counter += 1
                # If eyes have been closed for a sustained number of frames, it's drowsiness.
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    status_text = "DROWSINESS DETECTED!"
                    status_color = (0, 0, 255) # Red
                    if alarm_sound is not None and not alarm_playing:
                        alarm_sound.play(loops=-1) # Loop the alarm
                        alarm_playing = True
            else:
                # If eyes are open, reset the counter and stop the alarm.
                frame_counter = 0
                if alarm_playing:
                    if alarm_sound is not None:
                        alarm_sound.stop()
                    alarm_playing = False

            # --- (Optional) Draw eye landmarks on the face ---
            # This helps visualize what the code is detecting
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                if idx < len(landmarks):
                    cv2.circle(frame, (landmarks[idx][0], landmarks[idx][1]), 2, (0, 255, 0), -1)

        else:
            # If no face is detected, reset the counter and stop any playing alarm
            frame_counter = 0
            if alarm_playing:
                if alarm_sound is not None:
                    alarm_sound.stop()
                alarm_playing = False
            status_text = "No Face Detected"
            status_color = (0, 0, 255)

        # --- Display status on the video window ---
        # Show the current Eye Aspect Ratio
        cv2.putText(frame, f"EAR: {current_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Show the frame counter (debugging)
        cv2.putText(frame, f"Closed Frames: {frame_counter}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Show the main status message
        cv2.putText(frame, status_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

        # Show the video window
        cv2.imshow("Driver Drowsiness Detection System", frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if alarm_playing and alarm_sound is not None:
        alarm_sound.stop()
    pygame.quit()
    print("System shut down.")

if __name__ == "__main__":
    main()
main()