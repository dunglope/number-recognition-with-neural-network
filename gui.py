import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import numpy as np

class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Digit Drawing and Prediction")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='black')
        self.canvas.pack(side=tk.RIGHT)  # Place canvas on the right side

        self.label_frame = tk.Frame(root)
        self.label_frame.pack(side=tk.LEFT, padx=20)  # Frame to hold progress bars and labels

        self.progressbars = []
        self.labels = []

        for i in range(1, 10):  # From 1 to 9
            # Digit label
            label = tk.Label(self.label_frame, text=f"{i}", font=("Helvetica", 14))
            label.pack(anchor=tk.W, pady=5)
            self.labels.append(label)

            # Progress bar
            frame = tk.Frame(self.label_frame)
            frame.pack(anchor=tk.W)

            progressbar = ttk.Progressbar(frame, orient="horizontal", length=150, mode="determinate")
            progressbar.pack(side=tk.LEFT)

            progress_label = tk.Label(frame, text="0%", font=("Helvetica", 10), width=5)
            progress_label.pack(side=tk.LEFT, padx=5)

            self.progressbars.append((progressbar, progress_label))

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)

        self.image1 = Image.new("L", (280, 280), 0)  # Black background, larger canvas size
        self.draw = ImageDraw.Draw(self.image1)

        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 20))
        self.prediction_label.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.clear_canvas)
        self.reset_button.pack()

        self.last_x, self.last_y = None, None
        self.is_drawing = False

    def paint(self, event):
        if not self.is_drawing:
            self.is_drawing = True
            self.last_x, self.last_y = event.x, event.y
            return
        
        current_x, current_y = event.x, event.y
        r = 10  # Radius of the brush

        # Draw on the canvas
        self.canvas.create_line(self.last_x, self.last_y, current_x, current_y, fill="white", width=20)

        # Draw on the PIL image
        self.draw.line([self.last_x, self.last_y, current_x, current_y], fill=255, width=20)

        self.last_x, self.last_y = current_x, current_y

    def reset_coords(self, event):
        self.is_drawing = False
        self.update_prediction()
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (280, 280), 0)  # Reset PIL image
        self.draw = ImageDraw.Draw(self.image1)
        self.prediction_label.config(text="")
        for progressbar, progress_label in self.progressbars:
            progressbar["value"] = 0
            progress_label.config(text="0%")

    def update_prediction(self):
        img_array = np.array(self.image1.resize((28, 28)))  # Resize to model input size (28x28)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        predicted_label = np.argmax(prediction[0])
        self.prediction_label.config(text=f"Predicted Digit: {predicted_label}")

        # Update progress bars for each digit
        for i in range(9):  # Update from digit 1 to 9
            confidence = prediction[0][i + 1]  # Adjust index to start from digit 1
            percent_confidence = confidence * 100
            self.progressbars[i][0]["value"] = percent_confidence
            self.progressbars[i][1].config(text=f"{percent_confidence:.1f}%")

if __name__ == "__main__":
    import tensorflow as tf
    model_save_path = 'saved_model/model.h5'
    model = tf.keras.models.load_model(model_save_path)
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()
