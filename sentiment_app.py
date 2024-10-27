import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Streamlit app layout
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter a sentence:", key="unique_key_1")

if st.button("Analyze", key="btn_key_1"):
    if user_input:
        result = sentiment_pipeline(user_input)
        st.write(f"Sentiment: {result[0]['label']} with score {result[0]['score']:.2f}")
    else:
        st.write("Please enter a sentence.")

        
        
 