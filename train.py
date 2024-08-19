import tensorflow as tf
import data_loader
import model
import os

def train_model(model_save_path, epochs=10):
    # Load data
    (x_train, y_train), (x_test, y_test), datagen = data_loader.load_data()

    # Build model
    cnn_model = model.build_model(input_shape=(28, 28, 1))

    # Train the model
    history = cnn_model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(x_test, y_test)
    )

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    cnn_model.save(model_save_path)

    return cnn_model, history

#