# eye_contact_detector.py

import numpy as np
import cv2
import mediapipe as mp
import time

class EyeContactDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.define_mesh()
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.cap = cv2.VideoCapture(0)
        self.calibrated_x, self.calibrated_y, self.calibrated_z = 6, 0, 0
        self.blinking = False
        self.EAR_THRESHOLD = 0.2

    def define_mesh(self):
        return self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def draw_gaze_arrow(frame, start_point, gaze_direction, length=100, color=(0, 0, 255), thickness=2):
        gaze_direction_2d = gaze_direction[:2]  # Extract the x, y components
        end_point = tuple(map(int, (start_point + gaze_direction_2d * length)))
        start_point = tuple(map(int, start_point))
        cv2.arrowedLine(frame, start_point, end_point, color, thickness)

    def calculate_gaze_direction(iris_landmarks, eye_landmarks):
        iris_coords = np.array([(lm.x, lm.y, lm.z) for lm in iris_landmarks])
        eye_coords = np.array([(lm.x, lm.y, lm.z) for lm in eye_landmarks])
    
        iris_center = np.mean(iris_coords, axis=0)
        eye_center = np.mean(eye_coords, axis=0)
    
        gaze_vector = iris_center - eye_center
        return gaze_vector / np.linalg.norm(gaze_vector)

    def calculate_EAR(eye_landmarks):
        def landmark_to_point(landmark):
            return np.array([landmark.x, landmark.y])

        A = np.linalg.norm(landmark_to_point(eye_landmarks[1]) - landmark_to_point(eye_landmarks[5]))
        B = np.linalg.norm(landmark_to_point(eye_landmarks[2]) - landmark_to_point(eye_landmarks[4]))
        C = np.linalg.norm(landmark_to_point(eye_landmarks[0]) - landmark_to_point(eye_landmarks[3]))
        EAR = (A + B) / (2.0 * C)
        return EAR

    def detect_eye_contact(self):
        ret, image = self.cap.read()
        start = time.time()
        # Process image and detect eye contact
        # ...

        # After processing
        end = time.time()
        totalTime = end - start
        fps = 1 / (totalTime + 0.00001)

        return image, self.eye_contact

    def calibrate(self):
        # Re-calibrate based on current face position
        # ...

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
