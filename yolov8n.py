import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load the model
model = YOLO('yolov8n.pt')

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
if uploaded_image:
    image = Image.open(uploaded_image)
    results = model(image)
    results[0].show()  # Show results in your local environment

    # Display image in Streamlit
    st.image(image, caption="Uploaded Image", use_container_width=True)
