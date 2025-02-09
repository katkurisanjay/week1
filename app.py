import streamlit as st  
# from wasteclassification import load_images  # Import your functions from the Jupyter notebook  
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from huggingface_hub import hf_hub_download
import tensorflow as tf
# Define the repository ID and filename
repo_id = "sanjaykatkuri/wasteclassification"
filename = "my_waste_classification_model.keras"
# Download the model file
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
# Load the model
model = tf.keras.models.load_model(model_path)

def predict_fun(img):
    plt.figure(figsize=(6,4))
    # Change 'cv2.Color' to 'cv2.cvtColor'
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img,(224,224))
    img = np.reshape(img, [-1, 224, 224,3])
    result = np.argmax(model.predict(img))
    if result == 0:
        print("the image is Recyclable waste")
        return "The image is Recyclable waste"
    elif result == 1:
        print("the image show is Organic waste")
        return "The image show is Organic waste"

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
            result = predict_fun(uploaded_file)
            st.success(result)  
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