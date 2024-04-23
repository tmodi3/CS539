import numpy as np
import cv2
from tensorflow.keras.models import load_model, save_model

def load_keras_model(model_path):
    """
    Load a Keras model from a given file path.
    """
    return load_model(model_path)

def save_keras_model(model, model_path):
    """
    Save a Keras model to a given file path.
    """
    save_model(model, model_path)

def resize_images(images, size=(224, 224)):
    """
    Resize a list of images to a specified size.
    """
    resized_images = np.array([cv2.resize(image, size) for image in images])
    return resized_images

def normalize_images(images):
    """
    Normalize image data to 0-1 range.
    """
    return images.astype('float32') / 255.0
