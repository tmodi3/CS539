import tensorflow as tf
from tensorflow.keras.models import load_model

def convert_to_saved_model(model_path, export_path):
    """
    Convert a Keras model to TensorFlow SavedModel format.
    """
    model = load_model(model_path)
    tf.saved_model.save(model, export_path)
    print(f"Model saved in SavedModel format at: {export_path}")

def convert_to_tflite(model_path, output_path):
    """
    Convert a Keras model to TensorFlow Lite format.
    """
    model = load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved at: {output_path}")

def main():
    model_path = 'model_best.h5'  # Path to your Keras model
    save_model_path = './deploy/saved_model'
    tflite_model_path = './deploy/model.tflite'

    convert_to_saved_model(model_path, save_model_path)
    convert_to_tflite(model_path, tflite_model_path)

if __name__ == '__main__':
    main()
