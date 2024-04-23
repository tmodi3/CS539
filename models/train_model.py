import numpy as np
from efficient_net_model import create_efficient_net_model
from yolo_model import create_yolo_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, train_data, val_data, epochs=10):
    model.compile(optimizer=Adam(lr=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('model_best.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[checkpoint, early_stopping])
    return history

def main():
    num_classes = 10  # Adjust based on your dataset
    efficient_net_model = create_efficient_net_model(num_classes)
    yolo_model = create_yolo_model()

    # Dummy data for example purposes
    train_data = np.random.rand(100, 224, 224, 3), np.random.randint(0, num_classes, 100)
    val_data = np.random.rand(20, 224, 224, 3), np.random.randint(0, num_classes, 20)

    print("Training EfficientNet Model")
    train_model(efficient_net_model, train_data, val_data, epochs=10)

    print("Training YoLo Model")
    train_model(yolo_model, train_data, val_data, epochs=10)

if __name__ == '__main__':
    main()
