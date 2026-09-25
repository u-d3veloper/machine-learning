"""Streamlit demo: draw a digit and let the from-scratch NumPy MLP recognise it.

streamlit run app.py
"""

from pathlib import Path

import cv2
import streamlit as st
from numpy_mlp import load_parameters, predict
from streamlit_drawable_canvas import st_canvas

CANVAS_SIZE = 280  # 10x the MNIST resolution: comfortable to draw on

st.title("Handwritten digit recognition")
st.markdown(
    "Draw a digit (0-9) below. The model is a 784-64-10 neural network written "
    "with NumPy only and trained on MNIST (about 94% test accuracy)."
)


@st.cache_resource
def get_parameters():
    return load_parameters()


left, right = st.columns([2, 1])
with left:
    canvas = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",  # white on black, like MNIST
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )
    clicked = st.button("Predict")

with right:
    st.markdown("### Prediction")
    if clicked:
        if canvas.image_data is None:
            st.warning("Draw a digit first.")
        else:
            gray = canvas.image_data[:, :, 0]  # white strokes: any RGB channel works
            small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA) / 255
            digit = predict(small.reshape(784, 1), get_parameters())[0]
            st.success(f"## {digit}")

st.markdown("---")
st.markdown(
    """
    **How it works.** The 28x28 image is flattened to a 784-vector, multiplied by a
    64x784 weight matrix, passed through a ReLU, then through a 10x64 matrix and a
    softmax. The prediction is the class with the highest probability.

    $$ReLU(z) = \\max(0, z) \\qquad softmax(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

    The drawing is converted to grayscale, resized to 28x28 and scaled to [0, 1]
    before being fed to the network.
    """
)
st.image(
    str(Path(__file__).parent / "assets" / "model-schema.png"),
    caption="Network architecture",
)
