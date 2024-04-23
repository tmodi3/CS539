import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images(image_directory):
    """
    Load images from a directory, assuming each file is an image.
    """
    images = []
    labels = []  # Labels are assumed to be prefixed in the filename
    for filename in os.listdir(image_directory):
        img = cv2.imread(os.path.join(image_directory, filename))
        if img is not None:
            images.append(img)
            labels.append(filename.split('_')[0])  # Assuming label is part of filename
    return np.array(images), np.array(labels)

def preprocess_images(images):
    """
    Normalizes and resizes images to a fixed dimension.
    """
    # Normalize pixel values
    images = images.astype('float32') / 255.0
    # Resize images to 224x224
    images = np.array([cv2.resize(img, (224, 224)) for img in images])
    return images

def augment_data():
    """
    Returns an ImageDataGenerator object configured for data augmentation.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

def main():
    image_directory = './data/images'
    images, labels = load_images(image_directory)
    images = preprocess_images(images)
    
    # Example usage of ImageDataGenerator
    datagen = augment_data()
    # Fit parameters from data
    datagen.fit(images)
    
    print("Data preprocessing is set up and ready.")

if __name__ == '__main__':
    main()
