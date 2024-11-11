import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Function to load and display an image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Convert image to grayscale
def convert_to_grayscale(img):
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return gray

# Apply Gaussian blur to image
def apply_gaussian_blur(img, kernel_size):
    img_cv = np.array(img)
    blurred = cv2.GaussianBlur(img_cv, (kernel_size, kernel_size), 0)
    return blurred

# Apply median blur
def apply_median_blur(img, kernel_size):
    img_cv = np.array(img)
    blurred = cv2.medianBlur(img_cv, kernel_size)
    return blurred

# Edge detection using Canny
def apply_canny_edge_detection(img, threshold1, threshold2):
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, threshold1, threshold2)
    return edges

# Thresholding for image segmentation
def apply_threshold(img, thresh_value):
    img_cv = np.array(img)
    _, thresholded_img = cv2.threshold(img_cv, thresh_value, 255, cv2.THRESH_BINARY)
    return thresholded_img

# Object detection with Haar cascade
def detect_objects(img, cascade_path):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    img_cv = np.array(img)
    for (x, y, w, h) in objects:
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img_cv

# Real-time webcam object detection
def real_time_object_detection():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        stframe.image(frame, channels="BGR", use_column_width=True)
        
        # Stop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

# Set up Streamlit layout
st.title("OpenCV Image Manipulation")

st.markdown("Upload an image or use real-time webcam feed for object detection and various OpenCV operations.")

# Upload image
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Display uploaded image
    img = load_image(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Select operation
    operation = st.selectbox("Select an operation", 
                              ["None", "Grayscale", "Gaussian Blur", "Median Blur", "Edge Detection", 
                               "Thresholding", "Object Detection"])

    if operation == "Grayscale":
        result = convert_to_grayscale(img)
        st.image(result, caption="Grayscale Image", use_column_width=True)

    elif operation == "Gaussian Blur":
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=21, step=2, value=5)
        result = apply_gaussian_blur(img, kernel_size)
        st.image(result, caption=f"Gaussian Blurred Image (Kernel: {kernel_size}x{kernel_size})", use_column_width=True)

    elif operation == "Median Blur":
        kernel_size = st.slider("Kernel Size", min_value=3, max_value=21, step=2, value=5)
        result = apply_median_blur(img, kernel_size)
        st.image(result, caption=f"Median Blurred Image (Kernel: {kernel_size})", use_column_width=True)

    elif operation == "Edge Detection":
        threshold1 = st.slider("Lower Threshold", min_value=50, max_value=200, value=100)
        threshold2 = st.slider("Upper Threshold", min_value=100, max_value=300, value=200)
        result = apply_canny_edge_detection(img, threshold1, threshold2)
        st.image(result, caption="Edge Detected Image (Canny)", use_column_width=True)

    elif operation == "Thresholding":
        thresh_value = st.slider("Threshold Value", min_value=0, max_value=255, value=127)
        result = apply_threshold(img, thresh_value)
        st.image(result, caption="Thresholded Image", use_column_width=True)

    elif operation == "Object Detection":
        cascade_path = st.selectbox("Select Cascade Classifier", 
                                    ["haarcascade_frontalface_default.xml", "haarcascade_eye.xml"])
        result = detect_objects(img, cascade_path)
        st.image(result, caption="Object Detected Image", use_column_width=True)

# Real-time Webcam Feed
if st.button("Start Webcam Feed with Object Detection"):
    real_time_object_detection()
