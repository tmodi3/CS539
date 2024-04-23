import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():
    """
    Load your dataset here.
    This function is a placeholder and should be replaced with the actual code to load your dataset.
    """
    # Dummy data generation for demonstration:
    # Assume X is an array of image data and y is labels
    X = np.random.rand(1000, 224, 224, 3)  # 1000 images, 224x224 size, 3 color channels
    y = np.random.randint(0, 10, 1000)  # 1000 labels for 10 classes
    return X, y

def partition_data(X, y, test_size=0.20, val_size=0.25):
    """
    Partitions data into training, validation, and testing sets.
    Args:
        X: Feature dataset.
        y: Labels for the dataset.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the training dataset to include in the validation split.
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: Partitioned data sets.
    """
    # First split to carve out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Second split to carve out the validation set from the remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = partition_data(X, y)

    # Optionally, print shapes of the datasets to verify everything is as expected
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)

if __name__ == '__main__':
    main()
