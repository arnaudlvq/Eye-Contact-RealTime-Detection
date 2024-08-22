import numpy as np
import cv2
import mediapipe as mp
import time

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

# Variables to store the calibration angles, depending on your face geometry might need calibration
calibrated_x = 0
calibrated_y = 0
calibrated_z = 0
calibrated = False

# EAR threshold for blink detection
EAR_THRESHOLD = 0.25
blink_counter = 0
blinking = False

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view

    image.flags.writeable = False

    results = face_mesh.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_2d = []
    face_3d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
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

            success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

            # Getting rotational angles of face
            rmat, jac = cv2.Rodrigues(rotation_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Press 'c' to calibrate the straight position
            if cv2.waitKey(5) & 0xFF == ord('c'):
                calibrated_x = x
                calibrated_y = y
                calibrated_z = z

            # Adjusting angles based on calibration
            x -= calibrated_x
            y -= calibrated_y
            z -= calibrated_z

            # Determine head pose direction
            if y < -6:
                head_text = "Left"
            elif y > 6:
                head_text = "Right"
            elif x < -10:
                head_text = "Down"
            elif x > 10:
                head_text = "Up"
            else:
                head_text = "Forward"

            # Create an overlay for the mesh with transparent drawing
            mesh_overlay = np.zeros_like(image, dtype=np.uint8)

            # Draw face mesh on the overlay
            mp_drawing.draw_landmarks(image=mesh_overlay,
                                      landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,
                                      connection_drawing_spec=drawing_spec)

            # Blend the mesh overlay with the original image
            mesh_alpha = 0.4  # Mesh transparency factor
            image = cv2.addWeighted(mesh_overlay, mesh_alpha, image, 1 - mesh_alpha, 0)



            ########## Calculate gaze direction


            left_iris = [face_landmarks.landmark[468]]
            right_iris = [face_landmarks.landmark[473]]
            
            left_eye = [face_landmarks.landmark[i] for i in [ 33, 173]]
            right_eye = [face_landmarks.landmark[i] for i in [ 398, 263]]


            complete_left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            complete_right_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            
            # Calculate EAR for blink detection
            left_ear = calculate_EAR(complete_left_eye)
            right_ear = calculate_EAR(complete_right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            if (avg_ear < EAR_THRESHOLD) & (head_text != "Up"):
                blinking = True
                head_text = "blinking"
            else:
                blinking = False

            # In the while loop after gaze detection
            left_gaze = calculate_gaze_direction(left_iris, left_eye)
            right_gaze = calculate_gaze_direction(right_iris, right_eye)
            gaze_offset = [0, 0.2, 0]
            gaze_direction = (left_gaze + right_gaze) / 2 + gaze_offset
            
            if gaze_direction[0] < -0.4:
                gaze_text = "Left"
            elif gaze_direction[0] > 0.4:
                gaze_text = "Right"
            elif gaze_direction[1] < -0.3:
                gaze_text = "Up"
            elif gaze_direction[1] > 0.3:
                gaze_text = "Down"
            else:
                gaze_text = "Forward"
            
            cv2.putText(image, f"Gaze: {gaze_text}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)


            nose_tip = face_landmarks.landmark[4]
            start_point = np.array([nose_tip.x * image.shape[1], nose_tip.y * image.shape[0]])
            draw_gaze_arrow(image, start_point, gaze_direction)

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, f"Head: {head_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            # Determine if eye contact is made
            eye_contact = (
                not blinking and (
                (head_text == "Forward" and gaze_text == "Forward") or
                (head_text == "Up" and gaze_text == "Down") or
                (head_text == "Up" and gaze_text == "Forward") or
                (head_text == "Down" and gaze_text == "Up") or
                (head_text == "Left" and gaze_text == "Right") or
                (head_text == "Right" and gaze_text == "Left"))
            )
            
            if eye_contact:
                cv2.putText(image, "Eye Contact", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / (totalTime + 0.00001) ##Compute FPS with zero division sec

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose Detection', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
