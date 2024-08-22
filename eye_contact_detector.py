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
        self.EAR_THRESHOLD = 0.18
        self.eye_contact = False
        self._needs_calibration = False  # Private variable to track calibration status

    def define_mesh(self):
        return self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def calculate_gaze_direction(self, iris_landmarks, eye_landmarks):
        iris_coords = np.array([(lm.x, lm.y, lm.z) for lm in iris_landmarks])
        eye_coords = np.array([(lm.x, lm.y, lm.z) for lm in eye_landmarks])
        
        iris_center = np.mean(iris_coords, axis=0)
        eye_center = np.mean(eye_coords, axis=0)
        
        gaze_vector = iris_center - eye_center
        return gaze_vector / np.linalg.norm(gaze_vector)

    def calculate_EAR(self, eye_landmarks):
        def landmark_to_point(landmark):
            return np.array([landmark.x, landmark.y])

        A = np.linalg.norm(landmark_to_point(eye_landmarks[1]) - landmark_to_point(eye_landmarks[5]))
        B = np.linalg.norm(landmark_to_point(eye_landmarks[2]) - landmark_to_point(eye_landmarks[4]))
        C = np.linalg.norm(landmark_to_point(eye_landmarks[0]) - landmark_to_point(eye_landmarks[3]))
        EAR = (A + B) / (2.0 * C)
        return EAR

    def draw_gaze_arrow(self, frame, start_point, gaze_direction, length=100, color=(0, 0, 255), thickness=2):
        gaze_direction_2d = gaze_direction[:2]  # Extract the x, y components
        end_point = tuple(map(int, (start_point + gaze_direction_2d * length)))
        start_point = tuple(map(int, start_point))
        cv2.arrowedLine(frame, start_point, end_point, color, thickness)

    def detect_eye_contact(self):
        success, image = self.cap.read()
        if not success:
            return image, False

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append(([x, y, lm.z]))

                # Get 2d Coord
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # Getting rotational angles of face
                rmat, _ = cv2.Rodrigues(rotation_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Check if calibration is needed
                if self._needs_calibration:
                    self.calibrated_x = x
                    self.calibrated_y = y
                    self.calibrated_z = z
                    self._needs_calibration = False  # Calibration done

                # Adjusting angles based on calibration
                x -= self.calibrated_x
                y -= self.calibrated_y
                z -= self.calibrated_z

                # Determine head pose direction
                if y < -8:
                    head_text = "Left"
                elif y > 8:
                    head_text = "Right"
                elif x < -15:
                    head_text = "Down"
                elif x > 8:
                    head_text = "Up"
                else:
                    head_text = "Forward"

                # Create an overlay for the mesh with transparent drawing
                mesh_overlay = np.zeros_like(image, dtype=np.uint8)

                # Draw face mesh on the overlay
                self.mp_drawing.draw_landmarks(image=mesh_overlay,
                                               landmark_list=face_landmarks,
                                               connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec=self.drawing_spec,
                                               connection_drawing_spec=self.drawing_spec)

                # Blend the mesh overlay with the original image
                mesh_alpha = 0.4  # Mesh transparency factor
                image = cv2.addWeighted(mesh_overlay, mesh_alpha, image, 1 - mesh_alpha, 0)

                ########## Calculate gaze direction

                left_iris = [face_landmarks.landmark[468]]
                right_iris = [face_landmarks.landmark[473]]

                left_eye = [face_landmarks.landmark[i] for i in [33, 173]]
                right_eye = [face_landmarks.landmark[i] for i in [398, 263]]

                complete_left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                complete_right_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]

                # Calculate EAR for blink detection
                left_ear = self.calculate_EAR(complete_left_eye)
                right_ear = self.calculate_EAR(complete_right_eye)
                avg_ear = (left_ear + right_ear) / 2

                if (avg_ear < self.EAR_THRESHOLD) & (head_text != "Up"):
                    self.blinking = True
                    head_text = "blinking"
                else:
                    self.blinking = False

                # Calculate gaze direction
                left_gaze = self.calculate_gaze_direction(left_iris, left_eye)
                right_gaze = self.calculate_gaze_direction(right_iris, right_eye)
                gaze_offset = [0, 0.31, 0]
                gaze_direction = (left_gaze + right_gaze) / 2 + gaze_offset

                if gaze_direction[0] < -0.27:
                    gaze_text = "Left"
                elif gaze_direction[0] > 0.27:
                    gaze_text = "Right"
                elif gaze_direction[1] < -0.25:
                    gaze_text = "Up"
                elif gaze_direction[1] > 0.25:
                    gaze_text = "Down"
                else:
                    gaze_text = "Forward"

                # Determine if eye contact is made with smoother transitions
                self.eye_contact = (
                    not self.blinking and (
                        (head_text == "Forward" and gaze_text == "Forward") or
                        (3 <= y <= 8 and -0.9 <= gaze_direction[0] <= -0.2) or  # Face slightly right, gaze slightly left
                        (-8 <= y <= -3 and 0.2 <= gaze_direction[0] <= 0.9) or  # Face slightly left, gaze slightly right
                        (-20 <= x <= -12 and -0.4 <= gaze_direction[1] <= -0.14) or  # Face slightly down, gaze slightly up
                        (head_text == "Up" and gaze_text == "Down") or
                        (head_text == "Up" and gaze_text == "Forward") or
                        (head_text == "Down" and gaze_text == "Up") or
                        (head_text == "Left" and gaze_text == "Right") or
                        (head_text == "Right" and gaze_text == "Left"))
                )

                nose_tip = face_landmarks.landmark[4]
                start_point = np.array([nose_tip.x * image.shape[1], nose_tip.y * image.shape[0]])
                self.draw_gaze_arrow(image, start_point, gaze_direction)

                # Add text overlay for gaze directions and head pose
                cv2.putText(image, f'Gaze X: {gaze_direction[0]:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'Gaze Y: {gaze_direction[1]:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'Face X: {x:.2f}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'Face Y: {y:.2f}', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'blink ?: {self.blinking}', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'head: {head_text}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(image, f'gaze: {gaze_text}', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / (totalTime + 0.00001)  # Compute FPS with zero division sec

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        return image, self.eye_contact

    def calibrate(self):
        self._needs_calibration = True

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
