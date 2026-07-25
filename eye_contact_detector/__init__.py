"""Geometric eye-contact detection built on the MediaPipe FaceLandmarker.

One neural network (the FaceLandmarker, ~3.7 MB, CPU) turns each camera frame
into measurements; everything after it is plain 3D geometry:

  1. The facial transformation matrix gives the head rotation R and the head
     position t in cm, in camera space (X right, Y up, the face sits at Z < 0).
  2. The eyeLook* blendshapes give the eyes' rotation inside the head; they
     are mapped to angles through the physiological maximum eye rotation
     (about 40 deg horizontally, 25 deg vertically - same for every human).
  3. The world gaze ray is R applied to that eye-in-head direction, so
     head/eye compensation (head turned right, eyes turned left, ...) falls
     out of the matrix product instead of hand-tuned thresholds.
  4. The gaze is compared to the head->target direction and decomposed into a
     horizontal and a vertical angular error. Contact <=> both errors fall
     inside the detection window (settings.window_h_deg / window_v_deg),
     adjustable live from the SETTINGS screen. The CALIBRATE button absorbs
     the camera-to-clock offset and any residual bias in one corrective
     rotation: look at the clock, press it, the current gaze becomes zero.

Temporal stability: EMA smoothing of the errors, hysteresis (the window grows
20% while contact is held, a Schmitt trigger, so the state cannot flicker at
the boundary), and a freeze during blinks (closed lids have no measurable
gaze).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

HYSTERESIS_FACTOR = 1.2  # window grows by this much while contact is held


class GstCapture:
    """A cv2.VideoCapture drop-in that reads frames from a GStreamer subprocess.
    Useful for MIPI/CSI cameras that OpenCV's V4L2 backend can't drive (multiplanar
    or ISP-only pipelines) but a GStreamer `v4l2src` can. We take raw NV12 off an
    fdsink and convert in Python (cheaper than a gst videoconvert), which also
    makes grayscale free - the NV12 luma plane IS the grayscale image. Grayscale
    is the default: it drops the ISP's imperfect chroma (colour-layer artifacts)
    with < 1 px effect on the landmarks, and is slightly faster. The pipeline
    auto-restarts if it stalls or ends.
    """

    _ROT: ClassVar[dict[int, int]] = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def __init__(
        self,
        device: str,
        sensor_w: int,
        sensor_h: int,
        out_w: int,
        out_h: int,
        framerate: int = 30,
        rotate: int = 0,
        grayscale: bool = True,
        largemode: int = 0,
    ) -> None:
        self.device = device
        self.sensor_w, self.sensor_h = sensor_w, sensor_h
        self.out_w, self.out_h = out_w, out_h
        self.framerate = framerate
        self.rotate = rotate
        self.grayscale = grayscale
        self.largemode = largemode
        self.frame_bytes = out_w * out_h * 3 // 2  # NV12: W*H luma + W*H/2 chroma
        self.proc: subprocess.Popen | None = None
        self._fr = None
        self._buf = b""
        self._spawn()

    def _spawn(self) -> None:
        read_fd, write_fd = os.pipe()
        cmd = [
            "gst-launch-1.0", "v4l2src", f"device={self.device}",
            "en-awisp=1", f"en-largemode={self.largemode}",
            "!",
            f"video/x-raw,format=NV12,width={self.sensor_w},"
            f"height={self.sensor_h},framerate={self.framerate}/1",
            "!", "videoscale",
            "!", f"video/x-raw,format=NV12,width={self.out_w},height={self.out_h}",
            "!", "fdsink", f"fd={write_fd}",
        ]
        # ISP chatter goes to gst's stdout/stderr (discarded); frames go to the pipe
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(write_fd,)
        )
        os.close(write_fd)
        self._fr = os.fdopen(read_fd, "rb", buffering=0)

    def isOpened(self) -> bool:  # cv2 API name
        return self.proc is not None and self.proc.poll() is None

    def set(self, *args) -> bool:  # cv2 API no-op
        return True

    READ_TIMEOUT_S = 2.0  # a live pipeline delivers every frame interval

    def read(self):
        """Return the NEWEST complete frame, not the oldest queued one.

        The pipeline produces at the sensor rate while the caller may read
        much slower: without draining, frames pile up in the gst/pipe
        buffers and every read returns an ever-staler image (a reader at
        5 fps on a 15 fps stream settles around a second of lag). So block
        for the first complete frame, then drain whatever else has already
        arrived and keep only the latest.
        """
        if self._fr is None:
            return False, None
        fd = self._fr.fileno()
        try:
            # Block (bounded) until at least one complete frame is buffered.
            # select() keeps a dead-but-alive gst process (cable wiggle, ISP
            # hiccup) from freezing the caller forever.
            while len(self._buf) < self.frame_bytes:
                ready, _, _ = select.select([fd], [], [], self.READ_TIMEOUT_S)
                if not ready:
                    self._restart()
                    return False, None
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    self._restart()
                    return False, None
                self._buf += chunk
            # Drain everything already available without blocking.
            while True:
                ready, _, _ = select.select([fd], [], [], 0)
                if not ready:
                    break
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                self._buf += chunk
            n = len(self._buf) // self.frame_bytes
            buf = self._buf[(n - 1) * self.frame_bytes : n * self.frame_bytes]
            self._buf = self._buf[n * self.frame_bytes :]
        except OSError:
            self._restart()
            return False, None
        yuv = np.frombuffer(buf, np.uint8).reshape((self.out_h * 3 // 2, self.out_w))
        if self.grayscale:
            frame = cv2.cvtColor(yuv[: self.out_h], cv2.COLOR_GRAY2BGR)  # luma plane only
        else:
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if self.rotate in self._ROT:
            frame = cv2.rotate(frame, self._ROT[self.rotate])
        return True, frame

    def _restart(self) -> None:
        logger.warning("GStreamer camera stalled; restarting pipeline")
        self._buf = b""
        self.release()
        try:
            self._spawn()
        except OSError:
            logger.exception("GStreamer camera restart failed")

    def release(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self._fr is not None:
            with contextlib.suppress(OSError):
                self._fr.close()
            self._fr = None


class GazeSettings(Protocol):
    """Live-adjustable parameters (owned by the app, persisted to JSON)."""

    window_h_deg: float
    window_v_deg: float


@dataclass(frozen=True)
class DetectorConfig:
    """Static parameters."""

    camera_index: int = 0
    frame_width: int = 640  # modest resolution keeps single-board computers happy
    frame_height: int = 480

    # MIPI/CSI camera via GStreamer. When use_gst_camera is True, the OpenCV
    # capture is replaced by a gst-launch subprocess, for sensors whose ISP path
    # a plain v4l2src can't drive (some need a vendor v4l2src, e.g. en-awisp).
    use_gst_camera: bool = False
    gst_device: str = "/dev/video0"
    gst_sensor_size: tuple[int, int] = (1920, 1080)
    gst_output_size: tuple[int, int] = (960, 540)
    gst_rotate: int = 0  # 0/90/180/270 clockwise, for a sideways-mounted sensor
    gst_grayscale: bool = True  # luma-only: drops the ISP colour artifacts, faster
    gst_framerate: int = 30  # sensor rate; lower it to shed ISP + copy work

    # Physiological constants: human eyes rotate roughly this far when a
    # blendshape saturates at 1.0 (same for everyone, not tuning knobs)
    max_eye_yaw_deg: float = 40.0
    max_eye_pitch_deg: float = 25.0

    blink_threshold: float = 0.5  # eyeBlink blendshape score
    blink_hold_s: float = 0.4  # keep the previous state this long while lids are shut
    ema_alpha: float = 0.45  # smoothing factor for the angular errors


@dataclass
class GazeFrame:
    """Per-frame measurements, used by the calibration screen."""

    gaze: np.ndarray | None = None
    yaw_err_deg: float | None = None
    pitch_err_deg: float | None = None
    eye_yaw_deg: float = 0.0
    eye_pitch_deg: float = 0.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    distance_cm: float = 0.0
    nose_px: tuple[float, float] | None = None
    latency_ms: float = 0.0


class EyeContactDetector:
    """Detects whether the person in front of the camera looks at the clock."""

    def __init__(
        self,
        settings: GazeSettings,
        config: DetectorConfig | None = None,
        model_path: Path = DEFAULT_MODEL_PATH,
        landmarker: object | None = None,
    ) -> None:
        """`landmarker` is the pluggable inference backend. Leave it None to use
        the default **CPU MediaPipe FaceLandmarker** (portable, works anywhere).
        Inject any object exposing `detect_for_video(mp.Image, timestamp_ms) ->
        FaceLandmarkerResult` to swap in an accelerated backend (custom hardware,
        a delegate, a remote service…). The gaze geometry below is
        backend-agnostic, it never assumes what runs the model."""
        self.settings = settings
        self.config = config or DetectorConfig()

        if landmarker is not None:
            self.landmarker = landmarker
        else:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)

        if self.config.use_gst_camera:
            sw, sh = self.config.gst_sensor_size
            ow, oh = self.config.gst_output_size
            self.cap = GstCapture(
                self.config.gst_device, sw, sh, ow, oh,
                rotate=self.config.gst_rotate, grayscale=self.config.gst_grayscale,
                framerate=self.config.gst_framerate,
            )
            logger.info("Camera via GStreamer (%s -> %dx%d BGR)", self.config.gst_device, ow, oh)
        else:
            self.cap = cv2.VideoCapture(self.config.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        if not self.cap.isOpened():
            logger.warning("Camera could not be opened")

        self.eye_contact = False
        self.blinking = False
        self.last_reading: GazeFrame | None = None
        self.smoothed_yaw_err: float | None = None
        self.smoothed_pitch_err: float | None = None
        self._blink_started_at = 0.0
        self._last_blink_time = 0.0
        self._needs_calibration = False
        self._calib_rotation = np.eye(3)  # absorbs camera->clock offset + bias
        self._last_timestamp_ms = 0
        self._camera_failures = 0

    def __enter__(self) -> EyeContactDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    # ------------------------------------------------------------------ gaze

    @staticmethod
    def _blendshape_scores(blendshapes: list) -> dict[str, float]:
        return {b.category_name: b.score for b in blendshapes}

    def _eye_in_head_direction(self, bs: dict[str, float]) -> tuple[np.ndarray, float, float]:
        """Unit gaze direction in the head frame from the eye blendshapes."""
        # Conjugate eye movement: looking toward the subject's left raises
        # lookOutLeft + lookInRight; both eyes are averaged for robustness.
        horizontal = (
            bs.get("eyeLookOutLeft", 0.0)
            + bs.get("eyeLookInRight", 0.0)
            - bs.get("eyeLookInLeft", 0.0)
            - bs.get("eyeLookOutRight", 0.0)
        ) / 2.0
        vertical = (
            bs.get("eyeLookUpLeft", 0.0)
            + bs.get("eyeLookUpRight", 0.0)
            - bs.get("eyeLookDownLeft", 0.0)
            - bs.get("eyeLookDownRight", 0.0)
        ) / 2.0

        yaw = math.radians(horizontal * self.config.max_eye_yaw_deg)  # + = subject's left = +X
        pitch = math.radians(vertical * self.config.max_eye_pitch_deg)  # + = up = +Y

        direction = np.array(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ]
        )
        return direction, math.degrees(yaw), math.degrees(pitch)

    @staticmethod
    def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Smallest rotation matrix taking unit vector a onto unit vector b."""
        axis = np.cross(a, b)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return np.eye(3)
        angle = math.atan2(norm, float(np.dot(a, b)))
        rotation, _ = cv2.Rodrigues(axis / norm * angle)
        return rotation

    # ------------------------------------------------------------- main loop

    def detect_eye_contact(self) -> tuple[np.ndarray, bool]:
        """Grab one camera frame; return (annotated mirrored frame, eye contact)."""
        success, frame = self.cap.read()
        if not success or frame is None:
            # A dead camera must not hold the last gaze state: the clock
            # would stay frozen on a stale eye contact until repair.
            self._camera_failures += 1
            if self._camera_failures >= 3:
                self.eye_contact = False
                self.smoothed_yaw_err = None
                self.smoothed_pitch_err = None
            return np.zeros((self.config.frame_height, self.config.frame_width, 3), np.uint8), (
                self.eye_contact
            )
        self._camera_failures = 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # detect_for_video requires strictly increasing timestamps
        ts = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = ts

        t_start = time.perf_counter()
        result = self.landmarker.detect_for_video(mp_image, ts)
        reading = self._process_result(result, frame)
        reading.latency_ms = (time.perf_counter() - t_start) * 1000
        self.last_reading = reading

        # Mirror for a selfie-style display, then draw the gaze arrow (so it
        # matches what the user sees).
        display = cv2.flip(frame, 1)
        self._draw_arrow(display, reading)
        return display, self.eye_contact

    def _process_result(self, result, frame: np.ndarray) -> GazeFrame:
        reading = GazeFrame()

        if not (result.face_blendshapes and result.facial_transformation_matrixes):
            self.eye_contact = False
            self.smoothed_yaw_err = None
            self.smoothed_pitch_err = None
            return reading

        bs = self._blendshape_scores(result.face_blendshapes[0])
        matrix = np.array(result.facial_transformation_matrixes[0])
        rotation, translation = matrix[:3, :3], matrix[:3, 3]

        if result.face_landmarks:
            nose = result.face_landmarks[0][4]
            reading.nose_px = (nose.x * frame.shape[1], nose.y * frame.shape[0])

        # Blink: gaze blendshapes are meaningless with the lids closed,
        # so freeze the current state for a short grace period.
        blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
        was_blinking = self.blinking
        self.blinking = blink > self.config.blink_threshold
        now = time.monotonic()
        if self.blinking:
            if not was_blinking:
                self._blink_started_at = now
            self._last_blink_time = now
            if (now - self._blink_started_at) > self.config.blink_hold_s:
                self.eye_contact = False  # eyes closed for a while is not contact
            return reading
        if (now - self._last_blink_time) <= 0.05:
            return reading  # let the gaze settle right after a blink

        # Gaze ray in camera space
        eye_dir, eye_yaw, eye_pitch = self._eye_in_head_direction(bs)
        gaze = self._calib_rotation @ (rotation @ eye_dir)
        gaze /= np.linalg.norm(gaze)

        # Target direction: toward the camera; CALIBRATE bends it onto the
        # actual clock position (camera above/below the screen, bias, ...)
        to_target = -translation / np.linalg.norm(translation)

        if self._needs_calibration:
            # Whatever we measure now should count as perfect contact
            self._calib_rotation = (
                self._rotation_aligning(gaze, to_target) @ self._calib_rotation
            )
            gaze = to_target.copy()
            self._needs_calibration = False
            logger.info("Gaze calibrated")

        # Decompose the angular error around the target direction into a
        # horizontal (left/right) and a vertical (up/down) component
        h_axis = np.cross(np.array([0.0, 1.0, 0.0]), to_target)
        h_axis /= np.linalg.norm(h_axis)
        v_axis = np.cross(to_target, h_axis)
        forward = float(np.dot(gaze, to_target))
        yaw_err = math.degrees(math.atan2(float(np.dot(gaze, h_axis)), forward))
        pitch_err = math.degrees(math.atan2(float(np.dot(gaze, v_axis)), forward))

        alpha = self.config.ema_alpha
        if self.smoothed_yaw_err is None:
            self.smoothed_yaw_err = yaw_err
            self.smoothed_pitch_err = pitch_err
        else:
            self.smoothed_yaw_err = alpha * yaw_err + (1 - alpha) * self.smoothed_yaw_err
            self.smoothed_pitch_err = alpha * pitch_err + (1 - alpha) * self.smoothed_pitch_err

        factor = HYSTERESIS_FACTOR if self.eye_contact else 1.0
        self.eye_contact = (
            abs(self.smoothed_yaw_err) < self.settings.window_h_deg * factor
            and abs(self.smoothed_pitch_err) < self.settings.window_v_deg * factor
        )

        head_fwd = rotation @ np.array([0.0, 0.0, 1.0])
        reading.gaze = gaze
        reading.yaw_err_deg = self.smoothed_yaw_err
        reading.pitch_err_deg = self.smoothed_pitch_err
        reading.eye_yaw_deg = eye_yaw
        reading.eye_pitch_deg = eye_pitch
        reading.head_yaw_deg = math.degrees(math.atan2(head_fwd[0], head_fwd[2]))
        reading.head_pitch_deg = math.degrees(math.asin(float(np.clip(head_fwd[1], -1, 1))))
        reading.distance_cm = float(np.linalg.norm(translation))
        return reading

    def _draw_arrow(self, image: np.ndarray, reading: GazeFrame) -> None:
        if reading.nose_px is None or reading.gaze is None:
            return
        width = image.shape[1]
        nose_x, nose_y = reading.nose_px
        start = (int(width - 1 - nose_x), int(nose_y))  # mirrored x
        # Project the 3D gaze onto the mirrored image plane: x is flipped by
        # the mirror, y is down in pixel coordinates.
        end = (int(start[0] - reading.gaze[0] * 120), int(start[1] - reading.gaze[1] * 120))
        color = (0, 255, 0) if self.eye_contact else (0, 0, 255)
        cv2.arrowedLine(image, start, end, color, 3)

    # ------------------------------------------------------------------ misc

    def calibrate(self) -> None:
        """Treat the gaze measured on the next frame as perfect contact."""
        self._needs_calibration = True

    def reset_calibration(self) -> None:
        self._calib_rotation = np.eye(3)

    def release(self) -> None:
        self.cap.release()
        self.landmarker.close()
