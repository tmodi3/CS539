from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def load_test_data():
    """
    Load your test dataset here.
    This function is a placeholder and should be replaced with the actual code to load your test data.
    """
    # Dummy test data generation for demonstration:
    # Assume X_test is an array of image data and y_test is labels
    X_test = np.random.rand(100, 224, 224, 3)  # 100 test images
    y_test = np.random.randint(0, 10, 100)  # 100 test labels for 10 classes
    return X_test, y_test

def evaluate_model(model_path, test_data, test_labels):
    """
    Load a model, evaluate it on the test set, and print classification metrics.
    """
    model = load_model(model_path)
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predicted_classes)
    print(f"Accuracy: {accuracy:.2f}")

    # Detailed classification report
    report = classification_report(test_labels, predicted_classes)
    print("Classification Report:\n", report)

def main():
    X_test, y_test = load_test_data()
    model_path = 'model_best.h5'  # Path to your model file
    evaluate_model(model_path, X_test, y_test)

if __name__ == '__main__':
    main()
