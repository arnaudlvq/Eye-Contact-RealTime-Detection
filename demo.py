#!/usr/bin/env python3
"""Minimal demo, prints whether someone is looking at the camera, live.

    pip install git+https://github.com/arnaudlvq/Eye-Contact-RealTime-Detection
    python demo.py
"""
from types import SimpleNamespace

from eye_contact_detector import DetectorConfig, EyeContactDetector

# window_*_deg = how wide the "looking at me" cone is (degrees), horizontal / vertical
settings = SimpleNamespace(window_h_deg=12.0, window_v_deg=10.0)
det = EyeContactDetector(settings, config=DetectorConfig(camera_index=0))

print("Look at the camera…  (Ctrl-C to quit)")
try:
    while True:
        _frame, contact = det.detect_eye_contact()
        print("  👁️  EYE CONTACT   " if contact else "  · · ·           ", end="\r", flush=True)
except KeyboardInterrupt:
    pass
finally:
    det.release()
    print()
