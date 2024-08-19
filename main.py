import train
import prediction
import os
import tensorflow as tf
import tkinter as tk
from gui import DrawingApp

def main():
    # Define paths
    model_save_path = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\saved models\model.keras")
    image_to_predict = (r"C:\Users\Admin\OneDrive\Máy tính\New folder\hand write\image\image.png")
    
    model = tf.keras.models.load_model(model_save_path)
    
     # Initialize the Tkinter root window
    root = tk.Tk()
    
    # Initialize the DrawingApp with the root window and the model
    app = DrawingApp(root, model)
    
    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
