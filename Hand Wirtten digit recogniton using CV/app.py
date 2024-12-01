import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the pre-trained model
model = load_model("F:/Deep learning Projects/Hand Wirtten digit recogniton using CV/MNIST.keras")

# Title of the app
st.title("Digit Recognition App")
st.write("Upload an image of a handwritten digit (0-9) or draw one using the canvas below.")

# Sidebar
st.sidebar.header("How it Works")
st.sidebar.write("""
1. Upload an image of a digit or draw one.
2. The app preprocesses the image and sends it to a neural network.
3. The network predicts the digit.
""")

# Function to preprocess the image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (black background)
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img).astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

# Option 1: Upload an image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    digit = np.argmax(prediction)
    st.write(f"Predicted Digit: **{digit}**")

# Option 2: Draw a digit
st.write("---")
st.write("Or draw a digit below:")

canvas = st.empty()  # Create an empty widget to act as a canvas
draw_btn = st.button("Draw Digit")

if draw_btn:
    with canvas.container():
        import streamlit_drawable_canvas as dc

        drawing = dc.st_canvas(
            fill_color="#000000",  # Background color
            stroke_width=10,
            stroke_color="#FFFFFF",  # Draw color
            background_color="#000000",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas",
        )

        if drawing.image_data is not None:
            # Convert canvas image to 28x28 grayscale
            canvas_img = drawing.image_data
            gray_image = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)
            resized = cv2.resize(gray_image, (28, 28))
            normalized = resized.astype('float32') / 255.0
            input_data = np.expand_dims(normalized, axis=(0, -1))

            # Predict digit
            prediction = model.predict(input_data)
            digit = np.argmax(prediction)
            st.write(f"Predicted Digit: **{digit}**")
            st.image(resized, caption="Processed Image", width=150)


    
    
    