import cv2
import numpy as np
from pyzbar.pyzbar import decode
import streamlit as st
from PIL import Image
import io

# Function to decode barcodes/QR codes and return bounding boxes
def decode_barcodes_qrcodes(image):
    # Decode barcodes and QR codes in the image
    decoded_objects = decode(image)
    
    # For storing bounding boxes and data for each object
    barcode_info = []
    
    for obj in decoded_objects:
        # Get the bounding box coordinates
        points = obj.polygon
        if len(points) == 4:
            pts = [tuple(point) for point in points]
            barcode_info.append({
                "type": obj.type,
                "data": obj.data.decode('utf-8'),
                "bbox": pts
            })
        else:
            # If the polygon is not 4 points, use the convex hull (e.g., QR codes)
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            barcode_info.append({
                "type": obj.type,
                "data": obj.data.decode('utf-8'),
                "bbox": hull
            })
    
    return barcode_info

# Function to draw bounding boxes around detected barcodes/QR codes
def draw_bounding_boxes(image, barcode_info):
    for info in barcode_info:
        # Draw the bounding box on the image
        cv2.polylines(image, [np.array(info['bbox'], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Place the decoded data next to the barcode/QR code
        text_position = (info['bbox'][0][0], info['bbox'][0][1] - 10)
        cv2.putText(image, info['data'], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Function to process image from Streamlit's uploaded file
def process_image(image_data):
    # Convert the image into an OpenCV format (BGR)
    image = np.array(image_data)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect and decode barcodes/QR codes
    barcode_info = decode_barcodes_qrcodes(image)
    
    # Draw bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image, barcode_info)
    
    return image_with_boxes, barcode_info

# Streamlit Web App

st.title("Barcode and QR Code Detector")

st.write("Upload an image to detect barcodes and QR codes.")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using PIL
    image_data = Image.open(uploaded_file)
    
    # Process the image and detect barcodes/QR codes
    image_with_boxes, barcode_info = process_image(image_data)
    
    # Convert image back to PIL format for display
    image_with_boxes_pil = Image.fromarray(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    
    # Display the processed image
    st.image(image_with_boxes_pil, caption="Processed Image with Barcodes/QR Codes Detected", use_column_width=True)
    
    # Display the detected barcode information
    if barcode_info:
        st.write("Detected Barcodes/QR Codes:")
        for info in barcode_info:
            st.write(f"Type: {info['type']}, Data: {info['data']}")
    else:
        st.write("No barcodes or QR codes detected.")
