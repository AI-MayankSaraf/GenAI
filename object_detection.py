import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Page configuration for better UX
st.set_page_config(
    page_title='YOLO Model Detection',
    page_icon='🦾',
    layout='centered',  # Use wide layout for better space utilization
    initial_sidebar_state='expanded'
)

# App header and description
st.title('🔍 YOLO Object Detection Application')
st.markdown(
    """
    **Upload an image or video and select a YOLO model for object detection.**
    This tool supports multiple YOLO models and provides real-time processing.
    """
)

# Sidebar for model selection
st.sidebar.header('Model Selection and Settings')
model_options = ['yolo11s.pt', 'yolo11n.pt', 'yolov11n-seg.pt', 'yolov10n.pt', 'yolov10s.pt', 'yolov9s.pt', 'yolov8n.pt', 'yolov8s.pt']
selected_model = st.sidebar.selectbox('Choose a YOLO model', model_options)
model = YOLO(selected_model)  # Load the selected model
st.sidebar.success(f'Model Loaded: {selected_model}')

# File uploader for user input
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

# Function to process image with YOLO
def process_image(image):
    if len(image.shape) == 2 or image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = model(image)
    processed_image = results[0].plot()
    return processed_image

# Function to process video with YOLO
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# Handle file uploads
if uploaded_file:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.subheader('Uploaded Image')
        st.image(image, caption='Original Image', width=600)

        st.subheader('Detection Result')
        processed_image = process_image(image)
        st.image(processed_image, caption='Processed Image', use_container_width=True)

    elif uploaded_file.type == 'video/mp4':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        st.subheader('Uploaded Video')
        st.video(video_path)

        st.subheader('Detection Result')
        process_video(video_path)

# Sidebar footer
st.sidebar.info('Adjust settings as needed to explore different models.')
st.sidebar.markdown('Developed by **Mayank Saraf**, guided by **Mr. Prakash Senapati** ❤️')
