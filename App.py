import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model and cache it
@st.cache_resource
def load_gan_model():
    return load_model('gan_generator_model.h5')

model = load_gan_model()

def preprocess_image(image):
    # Preprocess image to fit model's input requirements
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (model.input_shape[1], model.input_shape[2]))  # Resize
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

def predict_next_frame(image, original_size):
    # Predict the next frame using the loaded model
    processed_image = preprocess_image(image)
    predicted_frame = model.predict(processed_image)
    
    # Rescale pixel values and resize to original dimensions
    predicted_frame = (predicted_frame[0] * 255).astype(np.uint8)  # Rescale to 0-255
    predicted_frame = cv2.resize(predicted_frame, (original_size[1], original_size[0]))  # Resize back to original size
    return predicted_frame

def main():
    st.title("Satellite Image Next Frame Prediction Website")
    
    st.markdown("Upload a satellite image, and this app will predict the next frame based on your input using a GAN model.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB for correct color display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display image details
        st.write(f"Image shape: {image.shape}")
        
        # Add progress bar
        if st.button('Predict Next Frame'):
            with st.spinner('Predicting...'):
                original_size = image.shape[:2]  # Store original size
                predicted_frame = predict_next_frame(image, original_size)
                
                # Display images side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
                
                with col2:
                    st.image(predicted_frame, caption='Predicted Next Frame', use_column_width=True)

if __name__ == "__main__":
    main()
