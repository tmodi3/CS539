from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras import layers, models

def create_efficient_net_model(num_classes):
    base_model = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model initially

    # Add custom layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model

if __name__ == '__main__':
    model = create_efficient_net_model(10)  # Example: 10 classes
    model.summary()
