# main_app.py

import tkinter as tk
from eye_contact_detector import EyeContactDetector
import cv2
from PIL import Image, ImageTk

class EyeContactApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Contact Detection")
        self.detector = EyeContactDetector()

        self.label = tk.Label(root, text="Eye Contact", font=("Helvetica", 24))
        self.label.pack()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.quit_button = tk.Button(root, text="Quit", command=self.quit_program)
        self.quit_button.pack(side="left")

        self.calibrate_button = tk.Button(root, text="Calibrate", command=self.calibrate)
        self.calibrate_button.pack(side="right")

        self.update_frame()

    def update_frame(self):
        image, eye_contact = self.detector.detect_eye_contact()

        if eye_contact:
            self.label.config(text="Eye Contact", fg="green")
        else:
            self.label.config(text="No Eye Contact", fg="red")

        # Convert the image to PhotoImage format for Tkinter
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.img_tk = img_tk  # Keep a reference

        self.root.after(10, self.update_frame)  # Update frame every 10 ms

    def calibrate(self):
        self.detector.calibrate()

    def quit_program(self):
        self.detector.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeContactApp(root)
    root.mainloop()
