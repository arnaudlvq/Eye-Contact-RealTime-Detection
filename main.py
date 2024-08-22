import numpy as np
import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Face Mesh
def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    return face_mesh

# Draw gaze direction arrow on the frame
def draw_gaze_arrow(frame, start_point, gaze_direction, length=100, color=(0, 0, 255), thickness=2):
    gaze_direction_2d = gaze_direction[:2]  # Extract x, y components
    end_point = tuple(map(int, (start_point + gaze_direction_2d * length)))
    start_point = tuple(map(int, start_point))
    cv2.arrowedLine(frame, start_point, end_point, color, thickness)

# Calculate gaze direction based on iris and eye landmarks
def calculate_gaze_direction(iris_landmarks, eye_landmarks):
    iris_coords = np.array([(lm.x, lm.y, lm.z) for lm in iris_landmarks])
    eye_coords = np.array([(lm.x, lm.y, lm.z) for lm in eye_landmarks])
    
    iris_center = np.mean(iris_coords, axis=0)
    eye_center = np.mean(eye_coords, axis=0)
    
    gaze_vector = iris_center - eye_center
    return gaze_vector / np.linalg.norm(gaze_vector)

# Calculate Eye Aspect Ratio (EAR) for blink detection
def calculate_EAR(eye_landmarks):
    def landmark_to_point(landmark):
        return np.array([landmark.x, landmark.y])

    A = np.linalg.norm(landmark_to_point(eye_landmarks[1]) - landmark_to_point(eye_landmarks[5]))
    B = np.linalg.norm(landmark_to_point(eye_landmarks[2]) - landmark_to_point(eye_landmarks[4]))
    C = np.linalg.norm(landmark_to_point(eye_landmarks[0]) - landmark_to_point(eye_landmarks[3]))
    EAR = (A + B) / (2.0 * C)
    return EAR

# Process face landmarks to calculate head pose and gaze direction
def process_face_landmarks(face_landmarks, img_w, img_h, calibrated_angles, EAR_THRESHOLD):
    face_2d = []
    face_3d = []
    nose_2d, nose_3d = None, None

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

            if idx == 1:  # Nose tip
                nose_2d = (x, y)
                nose_3d = (x, y, lm.z * 3000)

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

    rmat, jac = cv2.Rodrigues(rotation_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360 - calibrated_angles['x']
    y = angles[1] * 360 - calibrated_angles['y']
    z = angles[2] * 360 - calibrated_angles['z']

    head_text = determine_head_pose(x, y)

    left_iris = [face_landmarks.landmark[468]]
    right_iris = [face_landmarks.landmark[473]]
    left_eye = [face_landmarks.landmark[i] for i in [33, 173]]
    right_eye = [face_landmarks.landmark[i] for i in [398, 263]]
    
    gaze_direction = calculate_gaze_direction(left_iris, left_eye)
    gaze_text = determine_gaze_direction(gaze_direction)
    
    complete_left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
    complete_right_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
    
    avg_ear = (calculate_EAR(complete_left_eye) + calculate_EAR(complete_right_eye)) / 2
    blinking = avg_ear < EAR_THRESHOLD and head_text != "Up"

    eye_contact = check_eye_contact(blinking, head_text, gaze_text)
    
    return head_text, gaze_text, x, y, z, nose_2d, nose_3d, rotation_vec, translation_vec, gaze_direction, eye_contact, blinking

# Determine head pose direction based on angles
def determine_head_pose(x, y):
    if y < -6:
        return "Left"
    elif y > 6:
        return "Right"
    elif x < -10:
        return "Down"
    elif x > 10:
        return "Up"
    else:
        return "Forward"

# Determine gaze direction based on gaze vector
def determine_gaze_direction(gaze_direction):
    if gaze_direction[0] < -0.4:
        return "Left"
    elif gaze_direction[0] > 0.4:
        return "Right"
    elif gaze_direction[1] < -0.3:
        return "Up"
    elif gaze_direction[1] > 0.3:
        return "Down"
    else:
        return "Forward"

# Check if eye contact is made
def check_eye_contact(blinking, head_text, gaze_text):
    return not blinking and (
        (head_text == "Forward" and gaze_text == "Forward") or
        (head_text == "Up" and gaze_text == "Down") or
        (head_text == "Up" and gaze_text == "Forward") or
        (head_text == "Down" and gaze_text == "Up") or
        (head_text == "Left" and gaze_text == "Right") or
        (head_text == "Right" and gaze_text == "Left")
    )

# Main loop for capturing video and processing frames
def main():
    face_mesh = init_face_mesh()
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    calibrated_angles = {'x': 0, 'y': 0, 'z': 0}
    EAR_THRESHOLD = 0.2
    mesh_alpha = 0.4  # Mesh transparency factor

    while cap.isOpened():
        success, image = cap.read()
        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                head_text, gaze_text, x, y, z, nose_2d, nose_3d, rotation_vec, translation_vec, gaze_direction, eye_contact, blinking = process_face_landmarks(
                    face_landmarks, img_w, img_h, calibrated_angles, EAR_THRESHOLD)

                # Handle key press for calibration
                if cv2.waitKey(5) & 0xFF == ord('c'):
                    calibrated_angles['x'] = x
                    calibrated_angles['y'] = y
                    calibrated_angles['z'] = z

                # Create an overlay for the mesh with transparent drawing
                mesh_overlay = np.zeros_like(image, dtype=np.uint8)
                mp_drawing.draw_landmarks(image=mesh_overlay,
                                          landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=drawing_spec,
                                          connection_drawing_spec=drawing_spec)

                # Blend the mesh overlay with the original image
                image = cv2.addWeighted(mesh_overlay, mesh_alpha, image, 1 - mesh_alpha, 0)

                # Draw gaze arrow
                nose_tip = face_landmarks.landmark[4]
                start_point = np.array([nose_tip.x * image.shape[1], nose_tip.y * image.shape[0]])
                draw_gaze_arrow(image, start_point, gaze_direction)

                # Draw head pose angle and gaze information
                cv2.putText(image, f"Head: {head_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, f"Gaze: {gaze_text}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                # Display eye contact status
                if eye_contact:
                    cv2.putText(image, "Eye Contact", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / (totalTime + 0.00001)  # Compute FPS

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow('Head Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
