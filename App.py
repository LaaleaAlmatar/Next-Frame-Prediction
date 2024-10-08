import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model (update the model path)
model = load_model('gan_generator_model.h5')


def preprocess_image(image):
    # Preprocess the image to fit the model's input requirements
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (model.input_shape[1], model.input_shape[2]))  # Resize to model input size
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension


def predict_next_frame(image):
    # Predict the next frame using the loaded model
    processed_image = preprocess_image(image)
    predicted_frame = model.predict(processed_image)
    return (predicted_frame[0] * 255).astype(np.uint8)  # Scale back to 0-255


def main():
    st.title("Next Frame Prediction App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Predict the next frame
        if st.button('Predict Next Frame'):
            predicted_frame = predict_next_frame(image)
            st.image(predicted_frame, caption='Predicted Next Frame', use_column_width=True)


if __name__ == "__main__":
    main()
