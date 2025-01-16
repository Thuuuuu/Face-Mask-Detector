import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import tkinter as tk
from tkinter import Button, Label, Frame, filedialog
from PIL import Image, ImageTk

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def select_image():
    global panel
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if len(file_path) > 0:
        # Load the image
        image = cv2.imread(file_path)
        frame = imutils.resize(image, width=400)  # Resize for display
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # Process detections
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX - 10, endY + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 4)

        # Convert frame to display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img)

        # Display image in the new frame
        if panel is None:
            panel = Label(image_frame, image=img_tk)
            panel.image = img_tk
            panel.pack(padx=5, pady=5)
        else:
            panel.configure(image=img_tk)
            panel.image = img_tk

        # Update UI
        home_frame.pack_forget()
        video_frame.pack_forget()
        image_frame.pack()

def start_video_processing(video_path=None):
    global cap, panel, video_on, video_paused
    video_on = True
    video_paused = False

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            stop_video()
            return
    else:
        cap = VideoStream(src=0).start()

    process_video_frame()

def process_video_frame():
    global cap, panel, video_on, video_paused

    if not video_on or video_paused:
        return

    if isinstance(cap, cv2.VideoCapture):
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from video.")
            stop_video()
            return
    else:
        frame = cap.read()

    if frame is None:
        print("Error: Frame is None.")
        stop_video()
        return

    frame = imutils.resize(frame, width=400)  # Resize for smaller display
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frame, label, (startX - 10, endY + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 4)

    # Convert frame to display in Tkinter
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    if panel is None:
        panel = Label(video_frame, image=image)
        panel.image = image
        panel.pack(padx=5, pady=5)
    else:
        panel.configure(image=image)
        panel.image = image

    root.after(10, process_video_frame)

def pause_video():
    global video_paused
    if video_paused:
        video_paused = False  # Tiếp tục video
        process_video_frame()  # Gọi lại xử lý khung hình
    else:
        video_paused = True  # Tạm dừng video

def stop_video():
    global cap, panel, video_on
    video_on = False
    if isinstance(cap, cv2.VideoCapture):
        cap.release()
    else:
        cap.stop()
    if panel:
        panel.pack_forget()
        panel = None
    video_frame.pack_forget()
    home_frame.pack()

def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if len(file_path) > 0:
        home_frame.pack_forget()
        video_frame.pack()
        start_video_processing(file_path)


def start_camera():
    home_frame.pack_forget()
    video_frame.pack()
    start_video_processing()

def switch_frame(target_frame):
    global video_on, cap, panel

    # Dừng video hoặc camera nếu đang chạy
    if video_on:
        video_on = False
        if isinstance(cap, cv2.VideoCapture):
            cap.release()
        elif isinstance(cap, VideoStream):
            cap.stop()
        cap = None  # Đặt về None để chuẩn bị cho lần chạy tiếp theo

    # Xóa panel hiển thị hình ảnh/video nếu có
    if panel:
        panel.pack_forget()
        panel = None

    # Ẩn tất cả các frame
    home_frame.pack_forget()
    video_frame.pack_forget()
    image_frame.pack_forget()

    # Hiển thị frame mục tiêu
    target_frame.pack()



# Load models
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")

# Initialize GUI
root = tk.Tk()
root.title("Face Mask Detector")

# Set full screen
root.attributes('-fullscreen', True)

video_on = False
cap = None
panel = None

home_frame = Frame(root)
video_frame = Frame(root)
image_frame = Frame(root)

Label(home_frame, text="Welcome to Face Mask Detector", font=("Time New Roman", 24)).pack(pady=20)
Button(home_frame, text="Bật camera", command=start_camera, font=("Helvetica", 16)).pack(pady=10)
Button(home_frame, text="Chọn Ảnh", command=select_image, font=("Helvetica", 16)).pack(pady=10)
Button(home_frame, text="Chọn Video", command=select_video, font=("Helvetica", 16)).pack(pady=10)
Button(home_frame, text="Thoát", command=root.quit, font=("Helvetica", 16)).pack(pady=10)
button_frame = Frame(video_frame)
button_frame.pack(pady=10)
Button(video_frame, text="Quay lại", command=lambda: switch_frame(home_frame), font=("Helvetica", 16)).pack(side=tk.LEFT, padx=5)
Button(video_frame, text="Pause/Resume Video", command=pause_video, font=("Helvetica", 16)).pack(side=tk.LEFT, padx=5)
Button(image_frame, text="Quay lại", command=lambda: switch_frame(home_frame), font=("Helvetica", 16)).pack(pady=10)


home_frame.pack()
root.mainloop()