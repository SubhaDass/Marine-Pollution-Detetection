import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

# Load trained model
model = tf.keras.models.load_model("model.h5")

# Binary classes
classes = [
    "No Trash Detected",
    "Trash Detected"
]

# Get image path
img_path = input("Enter image path: ").strip().replace('"', '')

# Load image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Print probabilities
print("\nProbabilities:")
for cls, prob in zip(classes, prediction[0]):
    print(f"{cls}: {prob:.4f}")

# Get prediction
predicted_class = classes[np.argmax(prediction)]

# Confidence
confidence = np.max(prediction) * 100

print(f"\nPrediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")