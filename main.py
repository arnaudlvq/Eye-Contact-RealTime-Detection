import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and Iris
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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

def draw_info_box(frame, face_orientation, gaze_direction, global_gaze):
    # Box parameters
    box_width = 250
    box_height = 100
    box_color = (50, 50, 50)  # Dark gray
    text_color = (255, 255, 255)  # White
    padding = 10
    text_y_offset = 20
    
    # Draw the box
    cv2.rectangle(frame, (padding, padding), (box_width + padding, box_height + padding), box_color, -1)
    
    # Write the text inside the box
    cv2.putText(frame, f"Face Orientation: {face_orientation}", (padding + 10, padding + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (padding + 10, padding + text_y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    cv2.putText(frame, f"Global Gaze: {global_gaze}", (padding + 10, padding + text_y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Create an overlay for transparent drawing
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # Extract face orientation
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = frame.shape[1]
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        left_iris = [results.multi_face_landmarks[0].landmark[i] for i in range(468, 472)]
        right_iris = [results.multi_face_landmarks[0].landmark[i] for i in range(473, 477)]
        
        left_eye = [results.multi_face_landmarks[0].landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [results.multi_face_landmarks[0].landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        
        left_gaze = calculate_gaze_direction(left_iris, left_eye)
        right_gaze = calculate_gaze_direction(right_iris, right_eye)
        
        face_orientation = np.array([angles[0], angles[1], angles[2]])
        gaze_direction = (left_gaze + right_gaze) / 2
        global_gaze = face_orientation + gaze_direction

        mp_drawing.draw_landmarks(
            overlay,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

        nose_tip = face_landmarks.landmark[4]
        start_point = np.array([nose_tip.x * overlay.shape[1], nose_tip.y * overlay.shape[0]])
        draw_gaze_arrow(overlay, start_point, gaze_direction)

        blended_frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)
        
        # Draw the info box with text
        draw_info_box(blended_frame, face_orientation, gaze_direction, global_gaze)
        
        cv2.imshow('Global Gaze Direction', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
