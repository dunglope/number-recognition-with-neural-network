import train
import os
import tensorflow as tf
import tkinter as tk
from gui import DrawingApp
import cfg.config

def main():
    # Define paths
    model_save_path = cfg.config.model_save_path
    
    if os.path.exists(model_save_path):
        cnn_model = tf.keras.models.load_model(model_save_path)
    else:
        cnn_model, _ = train.train_model(model_save_path)
        
    model = tf.keras.models.load_model(model_save_path)
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    cnn_model.save(model_save_path)
    
     # Initialize the Tkinter root window
    root = tk.Tk()
    
    # Initialize the DrawingApp with the root window and the model
    app = DrawingApp(root, model)
    
    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
