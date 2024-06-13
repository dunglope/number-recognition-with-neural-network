import train
import prediction
import os
import tensorflow as tf

def main():
    # Define paths
    model_save_path = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\saved models\model.keras")
    image_to_predict = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\image\image.png")

    # check if model_save_path exists
    if os.path.exists(model_save_path):
        print("Model already exists, loading model...")
        cnn_model = tf.keras.models.load_model(model_save_path)
    else:
        print("training new model...")
        cnn_model = train.train_model(model_save_path)

    # Predict the class of a new image
    predicted_character = prediction.predict_image(cnn_model, image_to_predict)
    print(f'The predicted character is: {predicted_character}')
    
if __name__ == "__main__":
    main()
