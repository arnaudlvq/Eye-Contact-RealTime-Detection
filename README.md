# Eye Contact Detection (OpenCV, Mediapipe)

This project implements a real-time eye contact detection system using Python, OpenCV, and MediaPipe. The application detects the user's gaze direction and head pose, determining if eye contact is made. It also includes a graphical user interface (GUI) built with `tkinter`, which displays the eye contact status and provides controls for calibration and quitting the application.

## Features

- **Real-time Eye Contact Detection:** Detects whether the user is making eye contact with the camera.
- **Head Pose Estimation:** Determines the orientation of the user's head (e.g., Left, Right, Up, Down, Forward).
- **Gaze Direction Analysis:** Analyzes the direction of the user's gaze to detect where they are looking.
- **Blink Detection:** Incorporates blink detection to ensure accurate gaze analysis.
- **Calibration Functionality:** Allows for calibration of the system to adapt to different face geometries.
- **Graphical User Interface (GUI):** A simple and intuitive interface that displays eye contact status and includes buttons for calibration and quitting the application.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- `pip` (Python package manager)

### Python Libraries

Install the required Python libraries using the following command:

```bash
pip install numpy opencv-python mediapipe pillow
```

## Usage

### Running the Application

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/eye-contact-detection.git
   cd eye-contact-detection
   ```
2. **Run the GUI application:**
   ```bash
   python main_app.py
   ```

3. **The application window will open, showing the video feed from your webcam, along with text indicating the eye contact status.**

## Controls

- **Eye Contact Status:** The text in the application window displays "Eye Contact" in green if eye contact is detected, or "No Eye Contact" in red if not.
- **Calibrate Button:** Press this button to calibrate the system based on your current head position. This is useful if the system needs to be adjusted to better detect your face.
- **Quit Button:** Press this button to exit the application.

## Code Structure

- **`eye_contact_detector.py`:** Contains the main logic for detecting eye contact, gaze direction, head pose, and handling calibration.
- **`main_app.py`:** Handles the graphical user interface (GUI) and interacts with the `EyeContactDetector` class to display real-time results.
- **`README.md`:** This file, providing an overview of the project.

## Customization

### Adjusting Thresholds

- **EAR Threshold:** The Eye Aspect Ratio (EAR) threshold for blink detection can be adjusted in the `EyeContactDetector` class (default is `0.2`).
- **Head Pose and Gaze Direction Sensitivity:** You can fine-tune the sensitivity for head pose and gaze direction detection by modifying the thresholds in the `detect_eye_contact` method.

### Adding Features

The modular structure of the code makes it easy to add new features, such as additional head pose classifications or more sophisticated gaze tracking.

## Troubleshooting

- **Calibration Issues:** If the system is not detecting eye contact accurately, try recalibrating by pressing the "Calibrate" button while facing the camera directly.
- **Performance:** Ensure that your webcam is functioning correctly and that your environment is well-lit for optimal detection.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, feel free to open an issue or to contact me.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses [MediaPipe](https://google.github.io/mediapipe/) for face mesh detection.
- Thanks to the open-source community for providing excellent tools and libraries.
