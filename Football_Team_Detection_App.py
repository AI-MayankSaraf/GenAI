import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

# Set up Streamlit page configuration
st.set_page_config(page_title='YOLO Model Detection', layout='wide')

# Streamlit app header
st.title('Object Detection Application')
st.markdown('Upload an image or video and select a YOLO model to detect objects.')

# Sidebar for model selection and configuration
st.sidebar.header('Model Selection and Settings')
model_options = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt','yolov11n.pt','yolov11n-seg.pt']
selected_model = st.sidebar.selectbox('Choose a YOLO model', model_options)
model = 'models\\'+selected_model
# Load the selected YOLO model
model = YOLO(model)
st.sidebar.success(f'Selected Model: {selected_model}')

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

# Function to process and display image detections
def process_image(image):
    results = model(image)
    processed_image = results[0].plot()  # Draws the detections on the image
    return processed_image

# Function to process and display video detections
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
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# Handling the uploaded file
if uploaded_file:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        image = np.array(image)

        st.subheader('Uploaded Image')
        st.image(image, caption='Original Image', use_column_width=True)

        # Process the image and display the result
        st.subheader('Detection Result')
        processed_image = process_image(image)
        st.image(processed_image, caption='Processed Image', use_column_width=True)

    elif uploaded_file.type == 'video/mp4':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        st.subheader('Uploaded Video')
        st.video(video_path)

        # Process the video
        st.subheader('Detection Result')
        process_video(video_path)

st.sidebar.info('Use the sidebar to select different YOLO models and settings.')
st.sidebar.markdown('Developed by Mayank Saraf, guidence of Mr. Prakash Senapati ❤️')