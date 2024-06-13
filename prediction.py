import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def load_and_prep_image(img_path):
    # Load the image with grayscale and resize to 28x28
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array /= 255.0
    
    return img_array

def predict_image(model, img_path):
    # Load and preprocess the image
    img_array = load_and_prep_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted label
    predicted_label = np.argmax(predictions[0])
    
    return predicted_label

if __name__ == "__main__":
    model_save_path = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\saved models\model.keras")
    image_to_predict = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\image\image.png")
    
    # Load the model
    cnn_model = tf.keras.models.load_model(model_save_path)
    
    # Make a prediction
    predicted_label = predict_image(cnn_model, image_to_predict)
    print(f'The predicted label is: {predicted_label}')
