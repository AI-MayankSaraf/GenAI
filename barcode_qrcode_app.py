import streamlit as st
import numpy as np
from pyzbar.pyzbar import decode
import qrcode
from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image

st.title("QR Code and Barcode Generator")

# Sidebar for selection
option = st.sidebar.selectbox(
    "Choose an option",
    ["Generate QR Code", "Generate Barcode", "Decode QR/Barcode"]
)

if option == "Generate QR Code":
    st.header("Generate QR Code")
    user_input = st.text_input("Enter text/data for QR Code:")
    
    if st.button("Generate QR Code"):
        if user_input:
            qr = qrcode.make(user_input)
            qr_image = np.array(qr)
            st.image(qr_image, caption="Generated QR Code", use_column_width=True)

            # Option to download the QR code
            qr.save("generated_qr_code.png")
            with open("generated_qr_code.png", "rb") as file:
                btn = st.download_button(
                    label="Download QR Code",
                    data=file,
                    file_name="generated_qr_code.png",
                    mime="image/png"
                )

elif option == "Generate Barcode":
    st.header("Generate Barcode")
    user_input = st.text_input("Enter text/data for Barcode:")
    
    if st.button("Generate Barcode"):
        if user_input:
            barcode = Code128(user_input, writer=ImageWriter())
            barcode.save("generated_barcode")
            barcode_image = Image.open("generated_barcode.png")
            st.image(barcode_image, caption="Generated Barcode", use_column_width=True)

            # Option to download the barcode
            with open("generated_barcode.png", "rb") as file:
                btn = st.download_button(
                    label="Download Barcode",
                    data=file,
                    file_name="generated_barcode.png",
                    mime="image/png"
                )

elif option == "Decode QR/Barcode":
    st.header("Decode QR/Barcode")
    uploaded_file = st.file_uploader("Upload an image containing QR/Barcode", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Decode using pyzbar
        decoded_objects = decode(image_array)

        if decoded_objects:
            for obj in decoded_objects:
                st.write(f"Type: {obj.type}")
                st.write(f"Data: {obj.data.decode('utf-8')}")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            st.write("No QR/Barcode detected in the image.")
