import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "models/resnet50_transfer_final.keras"

CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled_In_Scale",
    "Scratches"
]

IMG_SIZE = (200, 200)

print("\n==========================================")
print("   MANUFACTURING DEFECT DETECTION")
print("==========================================")

if len(sys.argv) != 2:
    print("\nUsage:")
    print("python src/predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

print("\nLoading ResNet50 model...")
model = tf.keras.models.load_model(MODEL_PATH)

img = image.load_img(
    image_path,
    target_size=IMG_SIZE,
    color_mode="rgb"
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array, verbose=0)[0]

predicted_index = np.argmax(predictions)
predicted_class = CLASS_NAMES[predicted_index]
confidence = predictions[predicted_index] * 100

print("\n------------------------------------------")
print("RESULT")
print("------------------------------------------")
print(f"Predicted Defect : {predicted_class}")
print(f"Confidence       : {confidence:.2f}%")
print("------------------------------------------")

print("\nClass Probabilities:")
for name, probability in zip(CLASS_NAMES, predictions):
    print(f"{name:20s}: {probability * 100:.2f}%")

print("\n==========================================")
