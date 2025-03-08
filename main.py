import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

from util import set_background
set_background('BGImage.png')

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/maize-leaf-disease-model.h5"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image and Return Confidence Score
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)

    # Get confidence score and predicted class
    max_confidence = np.max(predictions)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]

    # Return both predicted class and confidence score
    return predicted_class_name, max_confidence

# Function to Get Medicine Recommendation
def get_medicine_recommendation(disease):
    medicine_recommendations = {
        "Gray Leaf Spot": "Apply fungicides like Azoxystrobin or Mancozeb. Ensure proper crop rotation and use resistant varieties.",
        "Common Rust": "Use fungicides such as Propiconazole. Rotate crops and ensure proper irrigation management.",
        "Northern Leaf Blight": "Apply fungicides like Pyraclostrobin. Use disease-resistant seeds and manage crop debris.",
        "Healthy": "No treatment needed. Maintain regular crop monitoring and optimal growing conditions.",
        "Not Maize Leaf": "Unknown"
    }
    return medicine_recommendations.get(disease, "No specific recommendation available.")

# Streamlit App
st.title('AgriGuide: Disease Identification in Maize Leaf')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Predict the class and confidence score of the uploaded image
            prediction, confidence = predict_image_class(model, uploaded_image, class_indices)

            # Display class and confidence score
            st.success(f'Prediction: {prediction} (Confidence: {confidence * 100:.2f}%)')

            # Get and display medicine recommendation
            recommendation = get_medicine_recommendation(prediction)
            st.info(f'Medicine Recommendation: {recommendation}')