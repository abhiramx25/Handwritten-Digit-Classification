import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = load_model("mnist_cnn_model.h5")

st.set_page_config(page_title="Handwritten Digit Classifier", layout="centered")
st.title("✍️ Handwritten Digit Classifier")
st.markdown("Draw a digit below (0–9), then click **Predict**.")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # White stroke
    stroke_width=15,
    stroke_color="#FFFFFF",              # White ink
    background_color="#000000",          # Black canvas
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction logic
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    if st.button("Predict"):
        # Convert to grayscale
        img_pil = Image.fromarray(img.astype(np.uint8)).convert("L")

        # Resize to 28x28 (as required by MNIST model)
        img_resized = img_pil.resize((28, 28), Image.LANCZOS)

        # Show what the model will see
        st.image(img_resized, caption="Processed Input Image", width=100)

        # Normalize and reshape
        img_np = np.array(img_resized) / 255.0
        img_np = img_np.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img_np)
        digit = np.argmax(pred)

        # Display result
        st.subheader(f"🧾 Predicted Digit: **{digit}**")
        st.bar_chart(pred[0])