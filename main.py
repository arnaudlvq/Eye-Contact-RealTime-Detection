import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
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

def draw_info_box(frame, pitch, yaw, roll, gaze_direction, global_gaze):
    # Convert radians to degrees
    gaze_direction_deg = np.degrees(gaze_direction)
    global_gaze_deg = np.degrees(global_gaze)
    
    # Box parameters
    box_width = 450
    box_height = 110
    box_color = (50, 50, 50)  # Dark gray
    text_color = (255, 255, 255)  # White
    padding = 10
    text_y_offset = 20
    
    # Draw the box
    cv2.rectangle(frame, (padding, padding), (box_width + padding, box_height + padding), box_color, -1)
    
    # Write the text inside the box with the orientation values in degrees
    cv2.putText(frame, f"Pitch: {pitch:.2f}", (padding + 10, padding + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (padding + 10, padding + text_y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    cv2.putText(frame, f"Roll: {roll:.2f}", (padding + 10, padding + text_y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    cv2.putText(frame, f"Gaze Direction: {gaze_direction_deg}", (padding + 10, padding + text_y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    cv2.putText(frame, f"Global Gaze: {global_gaze_deg}", (padding + 10, padding + text_y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

def draw_orientation_marker(frame, rmat, marker_size=50):
    origin = np.array([marker_size, frame.shape[0] - marker_size, 0], dtype=np.float64)

    # Define the axes in 3D space
    axis_x = np.array([marker_size, 0, 0], dtype=np.float64)
    axis_y = np.array([0, -marker_size, 0], dtype=np.float64)
    axis_z = np.array([0, 0, -marker_size], dtype=np.float64)

    # Project 3D axes to 2D using the rotation matrix
    x_end = origin + rmat.dot(axis_x)
    y_end = origin + rmat.dot(axis_y)
    z_end = origin + rmat.dot(axis_z)

    # Convert coordinates to integers
    origin = tuple(origin[:2].astype(int))
    x_end = tuple(x_end[:2].astype(int))
    y_end = tuple(y_end[:2].astype(int))
    z_end = tuple(z_end[:2].astype(int))

    # Draw the axes on the frame
    cv2.line(frame, origin, x_end, (0, 0, 255), 2)  # X-axis in red
    cv2.line(frame, origin, y_end, (0, 255, 0), 2)  # Y-axis in green
    cv2.line(frame, origin, z_end, (255, 0, 0), 2)  # Z-axis in blue

def calculate_face_orientation(face_landmarks, frame_shape):

    image_points = []
    for idx, lm in enumerate(face_landmarks.landmark):
        # Convert landmark x and y to pixel coordinates
        x, y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])

        # Add the 2D coordinates to an array
        image_points.append((x, y))

    # Get relevant landmarks for headpose estimation
    face_2d_head = np.array([
        image_points[1],      # Nose
        image_points[199],    # Chin
        image_points[33],     # Left eye left corner
        image_points[263],    # Right eye right corner
        image_points[61],     # Left mouth corner
        image_points[291]     # Right mouth corner
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left Mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ])

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ])

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, face_2d_head, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Convert rotation matrix to quaternion
    quaternion = cv2.RQDecomp3x3(rotation_matrix)[1]

    # Compute Euler angles (yaw, pitch, and roll)
    yaw = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0])
    pitch = np.arcsin(-rotation_matrix[2][0])
    roll = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2])

    return pitch, yaw, roll, rotation_matrix, translation_vector


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Calculate face orientation
        pitch, yaw, roll, rotation_matrix, translation_vector = calculate_face_orientation(face_landmarks, frame.shape)
        if rotation_matrix is not None:
            # Create an overlay for transparent drawing
            overlay = np.zeros_like(frame, dtype=np.uint8)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                overlay,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            # Calculate gaze direction
            left_iris = [face_landmarks.landmark[i] for i in range(468, 472)]
            right_iris = [face_landmarks.landmark[i] for i in range(473, 477)]
            
            left_eye = [face_landmarks.landmark[i] for i in [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]]
            right_eye = [face_landmarks.landmark[i] for i in [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]]
            
            left_gaze = calculate_gaze_direction(left_iris, left_eye)
            right_gaze = calculate_gaze_direction(right_iris, right_eye)
            
            gaze_direction = (left_gaze + right_gaze) / 2
            global_gaze = np.array([pitch, yaw, roll]) + gaze_direction

            nose_tip = face_landmarks.landmark[4]
            start_point = np.array([nose_tip.x * overlay.shape[1], nose_tip.y * overlay.shape[0]])
            draw_gaze_arrow(overlay, start_point, gaze_direction)

            blended_frame = cv2.addWeighted(frame, 1.0, overlay, 0.1, 0)
            
            draw_info_box(blended_frame, pitch, yaw, roll, gaze_direction, global_gaze)

            # Draw the 3D orientation marker
            draw_orientation_marker(blended_frame, rotation_matrix)
            
            cv2.imshow('Global Gaze Direction', blended_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
