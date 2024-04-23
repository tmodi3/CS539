import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

def load_sample_data():
    """
    Load or generate sample data to visualize model predictions.
    """
    # Dummy sample data for demonstration
    images = np.random.rand(10, 224, 224, 3)  # 10 sample images
    return images

def display_predictions(model_path, sample_data):
    """
    Display model predictions on sample data.
    """
    model = load_model(model_path)
    predictions = model.predict(sample_data)
    predicted_classes = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Predictions')
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_data[i])
        ax.set_title(f'Predicted: {predicted_classes[i]}')
        ax.axis('off')
    plt.show()

def main():
    sample_data = load_sample_data()
    model_path = 'model_best.h5'  # Path to your model file
    display_predictions(model_path, sample_data)

if __name__ == '__main__':
    main()
