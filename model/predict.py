import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("model.keras")

classes = ["No Trash Detected", "Trash Detected"]


img_path = input("Enter image path: ").strip().replace('"', '')

img = image.load_img(img_path, target_size=(224, 224))


img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

# Predict
prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction[0])

predicted_class = classes[predicted_index]

confidence = prediction[0][predicted_index] * 100

# Show probabilities
print("\nProbabilities:")

for cls, prob in zip(classes, prediction[0]):
    print(f"{cls}: {prob:.4f}")

print("\nPrediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")