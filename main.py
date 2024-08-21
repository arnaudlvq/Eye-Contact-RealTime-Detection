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
calibrated_x = 4
calibrated_y = 2
calibrated_z = 0
calibrated = False

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
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Create an overlay for transparent drawing
            overlay = np.zeros_like(image, dtype=np.uint8)

            # Calculate gaze direction
            left_iris = [face_landmarks.landmark[i] for i in range(468, 472)]
            right_iris = [face_landmarks.landmark[i] for i in range(473, 477)]
            
            left_eye = [face_landmarks.landmark[i] for i in [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]]
            right_eye = [face_landmarks.landmark[i] for i in [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]]
            
            left_gaze = calculate_gaze_direction(left_iris, left_eye)
            right_gaze = calculate_gaze_direction(right_iris, right_eye)
            
            gaze_direction = (left_gaze + right_gaze) / 2

            nose_tip = face_landmarks.landmark[4]
            start_point = np.array([nose_tip.x * overlay.shape[1], nose_tip.y * overlay.shape[0]])
            draw_gaze_arrow(overlay, start_point, gaze_direction)

            # Blend the overlay with the original image
            alpha = 0.6  # Transparency factor
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=drawing_spec,
                                  connection_drawing_spec=drawing_spec)
    cv2.imshow('Head Pose Detection', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
