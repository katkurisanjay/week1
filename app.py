import streamlit as st  
# from wasteclassification import load_images  # Import your functions from the Jupyter notebook  
import pandas as pd
import numpy as np
import time


def main():  
    # Set the title and layout  
    st.set_page_config(page_title="Waste Classification using CNN", layout="centered")  
    st.title("♻️ Waste Classification using CNN")  
    st.markdown("Upload an image for classification")  

    # Image upload section  
    uploaded_file = st.file_uploader(  
        "Drag and drop file here or Browse files",   
        type=["jpg", "png", "jpeg"],   
        help="Limit 200MB per file"  
    )  

    # URL input section  
    image_url = st.text_input("Or enter an Image URL for Classification:")  

    # Classification button  
    if st.button("Classify Image"):  
        if uploaded_file is not None:  
            # Code to process the uploaded image  
            st.success("Image uploaded successfully!")  
            # You can add your model prediction code here  
        elif image_url:  
            # Code to process the image URL  
            st.success("Image URL accepted!")  
            # You can add your model prediction code here  
        else:  
            st.error("Please upload an image or enter a URL.")  

    # Trainer credits  
    st.markdown("Trainer Credits: RMS")  
    st.markdown("Developed with ❤ for AICTE Internship Cycle 3 from RMS")
    
if __name__ == "__main__":  
    main()