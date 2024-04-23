from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model (adjust the path to your saved model directory)
model = tf.keras.models.load_model('./deploy/saved_model')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that expects a JSON array of images encoded as lists of pixel values.
    """
    data = request.get_json()
    # Assuming incoming data is already normalized and properly shaped
    images = np.array(data['images'])
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    return jsonify({'predictions': predicted_classes.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
