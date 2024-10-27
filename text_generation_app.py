import streamlit as st
from transformers import pipeline

# Load a pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

st.title("Text Generation App")
prompt = st.text_area("Enter a text:", key="unique_key_2")

if st.button("Analyze", key="btn_key_2"):
    if prompt:
        #generated_text = generator(prompt, max_length=50, num_return_sequences=1)
        generated_text = generator(prompt, max_new_tokens=50, num_return_sequences=1)
        st.write(f"Sentiment: {generated_text[0]['generated_text']}")
    else:
        st.write("Please enter a text.")
        
        
 