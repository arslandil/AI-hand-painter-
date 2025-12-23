import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np 
import cv2
from PIL import Image, ImageTk

# Create tkinter application window
app = tk.Tk()
app.geometry('600x600')
app.title('DROWSY')
ctk.set_appearance_mode('dark')

# Create frame for video display
vid_frame = tk.Frame(app)
vid_frame.pack(expand=True, fill="both")  # To make the frame expand in the window
vid_frame.config(width=600, height=480)   # Set the width and height after packing

# Create custom Tkinter label for displaying video
vid = ctk.CTkLabel(vid_frame)
vid.pack()

# Counter initialization and label for counter
counter = 0
counter_label = ctk.CTkLabel(master=app, height=40, width=120, text_color='white', fg_color='teal')
counter_label.pack(pady=10)

# Function to reset counter
def reset_counter():
    global counter
    counter = 0
    counter_label.configure(text=counter)

# Button to reset counter
reset_button = ctk.CTkButton(master=app, text='Reset Counter', command=reset_counter, height=40, width=120,
                             text_color='white', fg_color='teal')
reset_button.pack()

# Load YOLOv5 model (Ensure correct path to your weights)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_weights/last.pt', force_reload=True)
model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='path_to_your_weights/best.pt', force_reload=True)

# Load Haar cascade classifier for eye detection
cascade_path = "path_to_your_opencv/haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(cascade_path)

# Function to detect faces and check for drowsiness
def detect():
    global counter
    ret, frame = cap.read()

    if ret:  # Check if frame was read successfully
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model1(frame)
        img = np.squeeze(results.render())

        # Get bounding boxes and confidences for faces
        faces = results.xyxy[0]

        # Iterate over detected faces
        for face in faces:
            x1, y1, x2, y2, conf, cls = face
            # Draw rectangle around face
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Eye detection
            roi_gray = frame[int(y1):int(y2), int(x1):int(x2)]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangles around eyes
                cv2.rectangle(frame, (int(x1) + int(ex), int(y1) + int(ey)),
                              (int(x1) + int(ex) + int(ew), int(y1) + int(ey) + int(eh)), (0, 255, 0), 2)

                # Calculate eye aspect ratio (EAR) to detect drowsiness
                ear = calculate_ear(eyes)
                if ear < 0.2:  # Example threshold for drowsiness, adjust as needed
                    counter += 1
                    counter_label.configure(text=counter)

        # Convert image to PIL format for display
        img_arr = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img_arr)
        vid.imgtk = img_tk
        vid.configure(image=img_tk)

    # Call detect function again after 10 ms
    vid.after(10, detect)

# Function to calculate eye aspect ratio (EAR)
def calculate_ear(eyes):
    if len(eyes) >= 2:
        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[:2]  # Assuming only two eyes are detected
        # Calculate euclidean distance between vertical eye landmarks
        d_ver = np.linalg.norm(np.array((ex1 + ew1 // 2, ey1 + eh1 // 2)) - np.array((ex2 + ew2 // 2, ey2 + eh2 // 2)))
        # Calculate euclidean distance between horizontal eye landmarks
        d_hor1 = np.linalg.norm(np.array((ex1, ey1 + eh1 // 2)) - np.array((ex1 + ew1, ey1 + eh1 // 2)))
        d_hor2 = np.linalg.norm(np.array((ex2, ey2 + eh2 // 2)) - np.array((ex2 + ew2, ey2 + eh2 // 2)))
        # Calculate eye aspect ratio (EAR)
        ear = (d_ver / (d_hor1 + d_hor2)) * 2
        return ear
    else:
        return 0

# Initialize video capture
# Replace with the URL provided by the IP Webcam app or use a local camera (0 for the default camera)
url = "http://your_ip_address:8080/video"  # Example IP Webcam stream
cap = cv2.VideoCapture(url)

# Start detection
detect()

# Start tkinter main loop
app.mainloop()
