"""Gradio web demo for Dog Vision: upload a dog photo, get the top-5 predicted breeds.

Run locally:   python app.py
Needs the files exported at the end of dog_vision.ipynb (section 11):
    models/dog_vision_full.keras
    models/breeds.json
"""
import json
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
IMG_SIZE = 224

model = keras.models.load_model(os.path.join(MODEL_DIR, "dog_vision_full.keras"))
with open(os.path.join(MODEL_DIR, "breeds.json")) as f:
    breeds = json.load(f)


def preprocess(image: np.ndarray) -> tf.Tensor:
    # Same steps as preprocess_image() in the notebook, starting from decoded RGB pixels.
    image = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)
    return tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])


def predict(image: np.ndarray) -> dict:
    probs = model.predict(tf.expand_dims(preprocess(image), axis=0), verbose=0)[0]
    top5 = np.argsort(probs)[-5:][::-1]
    return {breeds[i].replace("_", " ").title(): float(probs[i]) for i in top5}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Dog photo"),
    outputs=gr.Label(num_top_classes=5, label="Predicted breed"),
    title="🐶 Dog Vision",
    description="Identifies 120 dog breeds using a ResNet101V2 transfer-learning model trained on the Kaggle Dog Breed Identification dataset.",
)

if __name__ == "__main__":
    demo.launch()
